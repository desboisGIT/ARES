"""Actual truncated BPTT and clipped PPO training for the flight policy.

The environment owns differentiable rigid-body transitions. PPO uses sampled
observations and rewards, without gradients through dynamics or an oracle.
"""
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import torch
from torch.nn.utils import clip_grad_norm_

from .flight_policy import FlightPolicy, quaternion_rotate


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 20
    rollout_steps: int = 32
    learning_rate: float = 3e-4
    seed: int = 0
    max_grad_norm: float = 1.0
    gamma: float = .99
    gae_lambda: float = .95
    clip_ratio: float = .2
    value_coefficient: float = .5
    entropy_coefficient: float = .001
    residual_penalty: float = 0.
    epochs: int = 4
    minibatch_size: int = 256

    def validate(self):
        if (any(type(x) is not int or x < 1 for x in
                (self.steps, self.rollout_steps, self.epochs, self.minibatch_size))
                or type(self.seed) is not int
                or not all(math.isfinite(x) for x in (self.learning_rate, self.max_grad_norm,
                    self.gamma, self.gae_lambda, self.clip_ratio, self.value_coefficient,
                    self.entropy_coefficient, self.residual_penalty))
                or self.learning_rate <= 0 or self.max_grad_norm <= 0
                or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1
                or not 0 < self.clip_ratio < 1
                or self.value_coefficient < 0 or self.entropy_coefficient < 0
                or self.residual_penalty < 0):
            raise ValueError('invalid training configuration')
        return self


def seed_training(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _config(config):
    return (config if isinstance(config, TrainingConfig) else TrainingConfig(**(config or {}))).validate()


def _copy_observation(observation):
    return {name: value.detach().clone() for name, value in observation.items() if torch.is_tensor(value)}


def _gradient_norm(module):
    values = [p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def _update(optimizer, policy, loss, config):
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite training loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = clip_grad_norm_(policy.parameters(), config.max_grad_norm, error_if_nonfinite=True)
    # Record pre-clipping overall gradient and post-clipping sensor-encoder gradient.
    cnn_gradient = _gradient_norm(policy.depth_encoder)
    optimizer.step()
    return float(gradient), cnn_gradient


def _active_transitions(info, done):
    return torch.as_tensor(info.get('active', torch.ones_like(done)),
                           device=done.device, dtype=torch.bool).expand_as(done)


def _reset_finished(env, done, observation):
    """Return the actual reset boundary, including legacy batch-reset fixtures."""
    if not bool(done.any()):
        return observation, done.clone()
    if hasattr(env, 'reset_idx'):
        return env.reset_idx(done), done.clone()
    return env.reset(), torch.ones_like(done)


def _active_mean(value, active):
    if value.ndim == 0:
        return value
    return value[active].mean()


def _residual_l2(policy, observation, latent):
    """Squared world residual norm per transition, using public policy outputs."""
    body = policy.config.residual_acceleration * latent.tanh()
    return quaternion_rotate(observation['q'], body).square().sum(-1)


def _deterministic_action_and_residual(policy, observation):
    # This is the existing FlightPolicy.forward path with its already computed
    # actor mean retained. No second depth/state encoding or stochastic draw.
    distribution, _ = policy.distribution(observation)
    latent = distribution.mean
    acceleration = policy.acceleration_from_latent(observation, latent)
    action = policy.action_from_acceleration(observation, acceleration)
    return action, _residual_l2(policy, observation, latent)


def _residual_reward_dt(env, config):
    if config.residual_penalty == 0:
        return 0.  # Historical fixture environments need not expose a timestep.
    dt = getattr(env, 'dt', None)
    if dt is None or not math.isfinite(dt) or dt <= 0:
        raise ValueError('PPO residual penalty requires finite positive public env.dt')
    return float(dt)


def train_bptt(env, policy, config=None, *, progress=None):
    """Optimize losses through full dynamics over explicit truncation windows.

    Selected episode resets are graph boundaries. Optimizer windows detach state and
    actuator memory *after* backward, preserving all transitions within a window.
    """
    config = _config(config)
    seed_training(config.seed)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    observation = env.reset()
    previous = torch.zeros(observation['features'].shape[0], 4,
                           device=observation['features'].device)
    curve, transitions = [], 0
    started = time.monotonic()
    for update in range(config.steps):
        start_update = getattr(env, 'start_update', None)
        staged_observation = start_update(update) if callable(start_update) else None
        if staged_observation is not None:
            observation = staged_observation
            previous = torch.zeros_like(previous)
        losses, rewards, terms = [], [], {}
        residual_terms = []
        resets, episode_resets = 0, 0
        for _ in range(config.rollout_steps):
            action, residual_l2 = _deterministic_action_and_residual(policy, observation)
            observation, reward, done, info = env.step(action)
            components = env.differentiable_loss(action, previous)
            if 'total' not in components:
                raise ValueError('environment differentiable_loss must include total')
            active = _active_transitions(info, done)
            if not bool(active.any()):
                observation, _ = _reset_finished(env, done, observation)
                previous = torch.zeros_like(action)
                continue
            physical_loss = _active_mean(components['total'], active)
            residual_mean = _active_mean(residual_l2, active)
            losses.append(physical_loss + config.residual_penalty * residual_mean
                          if config.residual_penalty > 0 else physical_loss)
            residual_terms.append(residual_mean.detach())
            rewards.append(_active_mean(reward.detach(), active))
            for name, value in components.items():
                if name != 'total':
                    terms.setdefault(name, []).append(_active_mean(value.detach(), active))
            transitions += int(active.sum())
            previous = action
            if bool(done.any()):
                observation, boundary = _reset_finished(env, done, observation)
                # Survivor action queues and previous commands retain their graph.
                previous = torch.where(boundary[:, None], torch.zeros_like(action), action)
                resets += 1
                episode_resets += int(boundary.sum())
        loss = torch.stack(losses).mean()
        gradient, cnn_gradient = _update(optimizer, policy, loss, config)
        env.detach()
        observation = env.observation()
        previous = previous.detach()
        row = {'update': update + 1, 'transitions': transitions,
               'loss': float(loss.detach()), 'reward': float(torch.stack(rewards).mean()),
               'gradient_norm': gradient, 'cnn_gradient_norm': cnn_gradient,
               'batch_resets': resets, 'episode_resets': episode_resets, 'elapsed_s': time.monotonic() - started}
        row.update({name: float(torch.stack(values).mean()) for name, values in terms.items()})
        row['residual_l2'] = float(torch.stack(residual_terms).mean())
        row['residual_l2_weighted'] = config.residual_penalty * row['residual_l2']
        curve.append(row)
        if progress is not None:
            progress(dict(row))
    return {'algorithm': 'bptt', 'config': asdict(config), 'seed': config.seed,
            'training_steps': transitions, 'optimizer_updates': config.steps,
            'training_curve': curve, 'architecture': policy.architecture}


def generalized_advantage_estimate(rewards, values, next_values, terminated, boundaries,
                                   gamma=.99, gae_lambda=.95, valid=None):
    """GAE with separate bootstrap and trace masks, including time limits.

    A time limit bootstraps the final state's value but ends its trace. A true
    terminal transition neither bootstraps nor leaks advantage into a reset.
    """
    valid = torch.ones_like(terminated) if valid is None else valid
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[-1])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]).to(rewards.dtype) - values[t]
        carry = delta + gamma * gae_lambda * (~boundaries[t]).to(rewards.dtype) * carry
        carry = torch.where(valid[t], carry, torch.zeros_like(carry))
        advantages[t] = carry
    return advantages, advantages + values


def train_ppo(env, policy, config=None, *, progress=None):
    """On-policy clipped PPO on squashed Gaussian acceleration residuals.

    Stored latents make the old/new probability ratio exact for the same fixed
    tanh map. All transitions and observations are detached during collection.
    """
    config = _config(config)
    seed_training(config.seed)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    observation = env.reset()
    _residual_reward_dt(env, config)  # Validate before collecting any transition.
    curve, transitions, optimizer_updates = [], 0, 0
    started = time.monotonic()
    for update in range(config.steps):
        start_update = getattr(env, 'start_update', None)
        staged_observation = start_update(update) if callable(start_update) else None
        if staged_observation is not None:
            observation = staged_observation
        observations, latents, old_log_probabilities = [], [], []
        rewards, values, next_values, terminals, boundaries, validity = [], [], [], [], [], []
        raw_rewards, residual_terms, residual_costs = [], [], []
        resets, episode_resets = 0, 0
        with torch.no_grad():
            for _ in range(config.rollout_steps):
                observations.append(_copy_observation(observation))
                residual_dt = _residual_reward_dt(env, config)
                action, latent, log_probability, value = policy.sample(observation)
                residual_l2 = _residual_l2(policy, observation, latent)
                next_observation, reward, done, info = env.step(action)
                raw_rewards.append(reward)
                residual_terms.append(residual_l2)
                cost = config.residual_penalty * residual_dt * residual_l2
                residual_costs.append(cost)
                if config.residual_penalty > 0:
                    reward = reward - cost
                # Evaluate terminal observation before reset: necessary for truncations.
                next_value = policy.value(next_observation)
                truncated = torch.as_tensor(info.get('truncated', torch.zeros_like(done)),
                                            device=done.device, dtype=torch.bool).expand_as(done)
                terminal = torch.as_tensor(info.get('terminated', done & ~truncated),
                                           device=done.device, dtype=torch.bool).expand_as(done)
                active = _active_transitions(info, done)
                observation, boundary = _reset_finished(env, done, next_observation)
                validity.append(active)
                latents.append(latent)
                old_log_probabilities.append(log_probability)
                rewards.append(reward)
                values.append(value)
                next_values.append(next_value)
                terminals.append(terminal)
                boundaries.append(boundary)
                transitions += int(active.sum())
                resets += int(bool(done.any()))
                episode_resets += int(boundary.sum())
        reward_tensor, value_tensor = torch.stack(rewards), torch.stack(values)
        advantage, returns = generalized_advantage_estimate(
            reward_tensor, value_tensor, torch.stack(next_values), torch.stack(terminals),
            torch.stack(boundaries), config.gamma, config.gae_lambda, valid=torch.stack(validity))
        valid = torch.stack(validity).flatten()
        flat_observations = {key: torch.cat([obs[key] for obs in observations], dim=0)[valid]
                             for key in observations[0]}
        flat_latents = torch.cat(latents, dim=0)[valid]
        old_log_probability = torch.cat(old_log_probabilities, dim=0)[valid]
        advantage, returns = advantage.flatten()[valid], returns.flatten()[valid]
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
        count = len(advantage)
        metrics = []
        for _ in range(config.epochs):
            permutation = torch.randperm(count, device=advantage.device)
            for start in range(0, count, config.minibatch_size):
                index = permutation[start:start + config.minibatch_size]
                minibatch = {key: value[index] for key, value in flat_observations.items()}
                log_probability, entropy, value = policy.evaluate_latent(minibatch, flat_latents[index])
                log_ratio = log_probability - old_log_probability[index]
                ratio = log_ratio.exp()
                surrogate = ratio * advantage[index]
                clipped = ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * advantage[index]
                actor_loss = -torch.minimum(surrogate, clipped).mean()
                value_loss = .5 * (value - returns[index]).square().mean()
                entropy_mean = entropy.mean()
                loss = actor_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy_mean
                gradient, cnn_gradient = _update(optimizer, policy, loss, config)
                optimizer_updates += 1
                with torch.no_grad():
                    approximate_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = ((ratio - 1).abs() > config.clip_ratio).float().mean()
                metrics.append([float(loss.detach()), float(actor_loss.detach()), float(value_loss.detach()),
                                float(entropy_mean.detach()), gradient, cnn_gradient,
                                float(approximate_kl), float(clip_fraction)])
        means = torch.tensor(metrics, dtype=torch.float64).mean(0).tolist()
        names = ['loss', 'actor_loss', 'value_loss', 'residual_entropy', 'gradient_norm',
                 'cnn_gradient_norm', 'approximate_kl', 'clip_fraction']
        row = dict(zip(names, means))
        row.update(update=update + 1, transitions=transitions, reward=float(reward_tensor.flatten()[valid].mean()),
                   batch_resets=resets, episode_resets=episode_resets, elapsed_s=time.monotonic() - started)
        row['env_reward'] = float(torch.stack(raw_rewards).flatten()[valid].mean())
        row['residual_l2'] = float(torch.stack(residual_terms).flatten()[valid].mean())
        row['residual_l2_weighted'] = config.residual_penalty * row['residual_l2']
        row['residual_cost'] = float(torch.stack(residual_costs).flatten()[valid].mean())
        curve.append(row)
        if progress is not None:
            progress(dict(row))
        env.detach()
        observation = env.observation()
    return {'algorithm': 'ppo', 'config': asdict(config), 'seed': config.seed,
            'training_steps': transitions, 'optimizer_updates': optimizer_updates,
            'training_curve': curve, 'architecture': policy.architecture}


def train_policy(env, algorithm='bptt', config=None, policy=None, *, progress=None):
    config = _config(config)
    if algorithm not in ('bptt', 'ppo'):
        raise ValueError('algorithm must be bptt or ppo')
    seed_training(config.seed)
    if policy is None:
        policy = FlightPolicy().to(env.device)
    result = (train_bptt if algorithm == 'bptt' else train_ppo)(env, policy, config, progress=progress)
    return policy, result


def save_checkpoint(path, policy, *, algorithm, seed, training_steps, training_curve,
                    metadata=None):
    """Atomically publish an exclusively created, weights-only-safe checkpoint."""
    if algorithm not in ('bptt', 'ppo'):
        raise ValueError('algorithm must be bptt or ppo')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON round trip rejects executable/nonportable metadata and tensor objects.
    plain = json.loads(json.dumps({'seed': seed, 'training_steps': training_steps,
                                  'training_curve': training_curve, 'metadata': metadata or {}},
                                 allow_nan=False))
    payload = {'schema_version': 'ares.flight_checkpoint.v1', 'algorithm': algorithm,
               'policy_config': policy.configuration(), 'architecture': policy.architecture,
               'state_dict': {name: value.detach().cpu() for name, value in policy.state_dict().items()},
               **plain}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.flight-', suffix='.tmp', delete=False) as handle:
            temporary = handle.name
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # link is atomic and refuses an existing destination, including races.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return path


def load_checkpoint(path, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get('schema_version') != 'ares.flight_checkpoint.v1':
        raise ValueError('unsupported flight checkpoint schema')
    if payload.get('algorithm') not in ('bptt', 'ppo'):
        raise ValueError('unsupported flight checkpoint algorithm')
    policy_config = dict(payload['policy_config'])
    policy_config.setdefault('controller_version', 'nominal_v1')
    policy_config.setdefault('guidance_version', 'legacy_v1')
    policy_config.setdefault('max_yaw_rate', policy_config.get('attitude_gain',6.)*math.pi)
    policy = FlightPolicy(policy_config).to(device)
    policy.load_state_dict(payload['state_dict'], strict=True)
    policy.eval()
    return policy, payload
