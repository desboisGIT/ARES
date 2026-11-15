"""Residual costs use public outputs and retain real-dynamics learning paths."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional torch dependency unavailable')
class FlightRegularizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        from ares.flight_policy import FlightPolicy
        self.policy = FlightPolicy({'residual_acceleration': .5})
        with torch.no_grad():
            self.policy.actor.bias.copy_(torch.tensor([.8, .4, -.3]))
        self.profile = json.loads((Path(__file__).parents[1]/'ares/data/configs/drone_sim_v2.json').read_text())

    def environment(self, dt=.02):
        from ares.diffaero_adapter import DiffAeroAdapter
        profile = self.profile
        class Environment:
            device = 'cpu'
            def __init__(self):
                self.dt = dt
                self.adapter = DiffAeroAdapter(profile, batch_size=2, dt=.02)
            def reset(self):
                self.adapter.reset()
                return self.observation()
            def observation(self):
                return {'depth': torch.full((2, 2, 24, 32), .3), 'features': torch.full((2, 43), .2),
                        'p': self.adapter.position, 'v': self.adapter.velocity, 'q': self.adapter.quaternion,
                        'wp1': torch.tensor([[0., 2., 0.]]).expand(2, -1),
                        'wp2': torch.tensor([[0., 4., 0.]]).expand(2, -1),
                        'gaze': torch.tensor([[0., 4., 0.]]).expand(2, -1),
                        'speedlimit': torch.ones(2, 1)}
            def step(self, action):
                self.adapter.step(action)
                reward = -self.adapter.velocity.sub(torch.tensor([0., 1., 0.])).square().sum(-1).detach()*self.adapter.dt
                done = torch.zeros(2, dtype=torch.bool)
                return self.observation(), reward, done, {'terminated': done, 'truncated': done}
            def differentiable_loss(self, action, previous_action):
                # The physical state alone supplies this gradient; no action
                # penalty or teacher target can replace the rigid-body path.
                physical = (self.adapter.velocity-torch.tensor([0., 1., 0.])).square().sum(-1)
                return {'total': physical, 'physical_velocity': physical}
            def detach(self):
                self.adapter.detach()
        return Environment()

    def config(self, **kwargs):
        from ares.flight_training import TrainingConfig
        return replace(TrainingConfig(steps=1, rollout_steps=4, epochs=1,
                       minibatch_size=8, seed=91, learning_rate=.001), **kwargs)

    def test_finite_nonnegative_coefficient_validation(self):
        for invalid in (-.1, math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                self.config(residual_penalty=invalid).validate()
        self.assertEqual(self.config().validate().residual_penalty, 0.)
        self.config(residual_penalty=1.).validate()

    def test_zero_bptt_is_exact_old_forward_objective_and_single_encoding(self):
        import ares.flight_training as training
        policy = deepcopy(self.policy); old = deepcopy(self.policy)
        config = self.config()
        encodings = []
        handle = policy.depth_encoder.register_forward_hook(lambda *args: encodings.append(1))
        result = training.train_bptt(self.environment(), policy, config)
        handle.remove()
        self.assertEqual(len(encodings), config.rollout_steps)
        # Historical deterministic BPTT path, including identical Adam and loss
        # ordering, checked against actual rigid-body transitions.
        training.seed_training(config.seed)
        optimizer = torch.optim.Adam(old.parameters(), lr=config.learning_rate)
        env = self.environment(); obs = env.reset(); previous = torch.zeros(2, 4); losses = []
        for _ in range(config.rollout_steps):
            action = old(obs)
            obs, _, _, _ = env.step(action)
            losses.append(env.differentiable_loss(action, previous)['total'].mean())
            previous = action
        loss = torch.stack(losses).mean()
        training._update(optimizer, old, loss, config)
        for name, value in old.state_dict().items():
            torch.testing.assert_close(policy.state_dict()[name], value, atol=0, rtol=0)
        self.assertEqual(result['training_curve'][0]['loss'], float(loss.detach()))
        self.assertEqual(result['training_curve'][0]['residual_l2_weighted'], 0.)

    def test_bptt_penalty_adds_sustained_bias_gradient_to_actual_dynamics(self):
        import ares.flight_training as training
        gradients, rows = [], []
        original_update = training._update
        def record(optimizer, policy, loss, config):
            gradients.append(torch.autograd.grad(loss, policy.actor.bias, retain_graph=True)[0].detach())
            return original_update(optimizer, policy, loss, config)
        for coefficient in (0., 1.):
            with patch('ares.flight_training._update', side_effect=record):
                result = training.train_bptt(self.environment(), deepcopy(self.policy),
                                             self.config(residual_penalty=coefficient))
            rows.append(result['training_curve'][0])
            self.assertEqual(result['config']['residual_penalty'], coefficient)
        self.assertGreater(float(gradients[0].norm()), 0.)
        bias = self.policy.actor.bias.detach()
        expected = 2*self.policy.config.residual_acceleration**2*bias.tanh()*(1-bias.tanh().square())
        torch.testing.assert_close(gradients[1]-gradients[0], expected, atol=2e-5, rtol=1e-4)
        expected_cost = float((self.policy.config.residual_acceleration*bias.tanh()).square().sum())
        self.assertAlmostEqual(rows[1]['residual_l2'], expected_cost, places=6)
        self.assertAlmostEqual(rows[1]['residual_l2_weighted'], expected_cost, places=6)
        self.assertAlmostEqual(rows[1]['loss']-rows[0]['loss'], expected_cost, places=6)

    def test_ppo_penalty_shapes_rewards_without_changing_latents_or_density(self):
        from ares.flight_training import train_ppo
        collections, rows = [], []
        for coefficient in (0., 1.):
            policy = deepcopy(self.policy); saved = []
            original_sample = policy.sample
            def sample(obs):
                values = original_sample(obs)
                saved.append(tuple(value.detach().clone() for value in values[:3]))
                return values
            with patch.object(policy, 'sample', side_effect=sample):
                result = train_ppo(self.environment(), policy, self.config(residual_penalty=coefficient))
            collections.append(saved); rows.append(result['training_curve'][0])
        for before, after in zip(*collections):
            for left, right in zip(before, after):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        row = rows[1]
        self.assertAlmostEqual(row['residual_cost'], .02*row['residual_l2_weighted'], places=7)
        self.assertAlmostEqual(row['env_reward']-row['reward'], row['residual_cost'], places=7)
        self.assertEqual(rows[0]['env_reward'], rows[0]['reward'])
        self.assertEqual(row['approximate_kl'], 0.)  # Stored latent density unchanged before update.

    def test_zero_ppo_objective_rng_and_weights_match_without_cost_computation(self):
        from ares.flight_training import train_ppo
        policies, results, random_states = [], [], []
        for skip_cost in (False, True):
            policy = deepcopy(self.policy)
            # Removing the new metric-only tensor operations reproduces the old
            # zero-cost rollout objective and stochastic draw sequence exactly.
            if skip_cost:
                with patch('ares.flight_training._residual_l2', side_effect=lambda p, o, l: torch.zeros(l.shape[0])):
                    result = train_ppo(self.environment(), policy, self.config())
            else:
                result = train_ppo(self.environment(), policy, self.config())
            policies.append(policy); results.append(result); random_states.append(torch.random.get_rng_state())
        for name, value in policies[0].state_dict().items():
            torch.testing.assert_close(value, policies[1].state_dict()[name], atol=0, rtol=0)
        torch.testing.assert_close(random_states[0], random_states[1], atol=0, rtol=0)
        for key in ('loss', 'actor_loss', 'value_loss', 'residual_entropy', 'reward', 'approximate_kl'):
            self.assertEqual(results[0]['training_curve'][0][key], results[1]['training_curve'][0][key])

    def test_positive_ppo_penalty_requires_public_timestep_zero_does_not(self):
        from ares.flight_training import train_ppo
        for invalid in (None, 0., -.02, math.inf, math.nan):
            env = self.environment(); env.dt = invalid
            with self.assertRaisesRegex(ValueError, 'env.dt'):
                train_ppo(env, deepcopy(self.policy), self.config(residual_penalty=1.))
        env = self.environment(); del env.dt
        result = train_ppo(env, deepcopy(self.policy), self.config())
        self.assertEqual(result['training_curve'][0]['residual_cost'], 0.)


if __name__ == '__main__':
    unittest.main()
