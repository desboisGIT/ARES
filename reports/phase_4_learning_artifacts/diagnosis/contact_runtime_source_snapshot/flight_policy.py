"""Depth/state neural residual over a nominal geometric flight controller.

Both learners use this identical architecture. Inputs contain measured state,
mission targets and sensor observations only; sampled simulator parameters and
occupancy are deliberately not inputs. Actions are thrust N and body rad/s.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from .flight_control import ControllerConfig, acceleration_to_action
from .flight_guidance import guidance_velocity


@dataclass(frozen=True)
class PolicyConfig:
    controller_version: str = 'adaptive_v2'
    guidance_version: str = 'braking_v2'
    feature_count: int = 43
    hidden_size: int = 64
    residual_acceleration: float = 2.0
    velocity_gain: float = 2.5
    attitude_gain: float = 6.0
    nominal_mass: float = .5
    gravity: float = 9.80665
    max_thrust: float = 24.5
    max_body_rate: float = 6.0
    max_yaw_rate: float = 1.0
    initial_log_std: float = -.7


def quaternion_rotate(q, vector):
    """Rotate body vector to world with an xyzw quaternion."""
    xyz = q[..., :3]
    cross = torch.linalg.cross(xyz, vector, dim=-1)
    return vector + 2 * (q[..., 3:] * cross + torch.linalg.cross(xyz, cross, dim=-1))


def quaternion_inverse_rotate(q, vector):
    return quaternion_rotate(torch.cat((-q[..., :3], q[..., 3:]), -1), vector)


def _unit(vector):
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-6)


class FlightPolicy(nn.Module):
    """Small depth CNN + state MLP, 3-D acceleration residual and value head."""

    @property
    def architecture(self):
        prefix = 'observed-state adaptive control' if self.config.controller_version=='adaptive_v2' else 'nominal geometric PD'
        return prefix + ' + learned depth/state acceleration residual'

    def __init__(self, config=None):
        super().__init__()
        self.config = config if isinstance(config, PolicyConfig) else PolicyConfig(**(config or {}))
        c = self.config
        numeric = (c.residual_acceleration,c.velocity_gain,c.attitude_gain,c.nominal_mass,
                   c.gravity,c.max_thrust,c.max_body_rate,c.max_yaw_rate,c.initial_log_std)
        if (not all(math.isfinite(value) for value in numeric)
                or c.controller_version not in ('nominal_v1','adaptive_v2') or c.guidance_version not in ('legacy_v1','braking_v2') or c.feature_count != 43 or c.hidden_size < 8 or c.residual_acceleration <= 0
                or min(c.velocity_gain, c.attitude_gain, c.nominal_mass, c.gravity,
                       c.max_thrust, c.max_body_rate, c.max_yaw_rate) <= 0):
            raise ValueError("invalid flight policy configuration")
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(2, 8, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(),
            nn.Linear(16 * 3 * 4, c.hidden_size), nn.SiLU())
        self.memory_feature_count = 11 if c.controller_version=='adaptive_v2' else 0
        self.state_encoder = nn.Sequential(nn.Linear(c.feature_count+self.memory_feature_count, c.hidden_size), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(c.hidden_size * 2, c.hidden_size), nn.SiLU(),
                                    nn.Linear(c.hidden_size, c.hidden_size), nn.SiLU())
        self.actor = nn.Linear(c.hidden_size, 3)
        self.critic = nn.Linear(c.hidden_size, 1)
        # Begin with the declared PD reference, allowing a useful first rollout.
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)
        self.log_std = nn.Parameter(torch.full((3,), c.initial_log_std))

    def encode(self, observation):
        features=observation['features']
        if self.memory_feature_count:
            zero=torch.zeros_like(observation['v'])
            flags=torch.zeros_like(zero[:,:1])
            memory=torch.cat((
                quaternion_inverse_rotate(observation['q'],observation.get('controller_bias',zero)),
                quaternion_inverse_rotate(observation['q'],observation.get('controller_velocity',zero)),
                quaternion_inverse_rotate(observation['q'],observation.get('controller_hold_anchor',observation['p'])-observation['p']),
                observation.get('controller_holding',flags).reshape(-1,1).to(features.dtype),
                observation.get('controller_stopping',flags).reshape(-1,1).to(features.dtype)), -1)
            features=torch.cat((features,memory),-1)
        return self.fusion(torch.cat((self.depth_encoder(observation['depth']),
                                      self.state_encoder(features)), -1))

    def distribution(self, observation):
        encoded = self.encode(observation)
        return Normal(self.actor(encoded), self.log_std.clamp(-5, 1).exp()), self.critic(encoded).squeeze(-1)

    def value(self, observation):
        return self.critic(self.encode(observation)).squeeze(-1)

    def reference_acceleration(self, observation):
        """Nominal PD reference; every operand comes from public observations."""
        p, v = observation['p'], observation['v']
        wp1, wp2 = observation['wp1'], observation['wp2']
        if self.config.guidance_version == 'braking_v2':
            target_velocity = guidance_velocity(p, wp1, wp2, observation['speedlimit'])
            return self.config.velocity_gain * (target_velocity - v)
        offset = wp1 - p
        distance = offset.norm(dim=-1, keepdim=True)
        continuation = wp2 - wp1
        has_next = continuation.norm(dim=-1, keepdim=True) > 1e-4
        blend = (1 - distance / .6).clamp(0, 1) * has_next
        direction = _unit((1 - blend) * _unit(offset) + blend * _unit(continuation))
        speed_limit = observation['speedlimit'].reshape(-1, 1).clamp_min(0)
        speed = torch.minimum(speed_limit, distance * 2 + blend * speed_limit)
        return self.config.velocity_gain * (direction * speed - v)

    def acceleration_from_latent(self, observation, latent):
        body_residual = self.config.residual_acceleration * latent.tanh()
        return self.reference_acceleration(observation) + quaternion_rotate(observation['q'], body_residual)

    def desired_acceleration(self, observation):
        distribution, _ = self.distribution(observation)
        return self.acceleration_from_latent(observation, distribution.mean)

    def action_from_acceleration(self, observation, acceleration):
        """Differentiable nominal attitude tracking, with physical action bounds."""
        c = self.config
        if c.controller_version=='adaptive_v2':
            return acceleration_to_action(observation,acceleration,ControllerConfig(
                nominal_mass=c.nominal_mass,gravity=c.gravity,max_thrust=c.max_thrust,
                max_body_rate=c.max_body_rate,max_yaw_rate=c.max_yaw_rate,attitude_gain=c.attitude_gain))
        q = observation['q']
        z = torch.zeros_like(acceleration)
        z[:, 2] = 1
        body_z = quaternion_rotate(q, z)
        force_per_mass = acceleration + c.gravity * z
        desired_z = _unit(force_per_mass)
        world_tilt_error = torch.linalg.cross(body_z, desired_z, dim=-1)
        rates = c.attitude_gain * quaternion_inverse_rotate(q, world_tilt_error)
        # Heading points toward the mission gaze; vertical gaze retains heading.
        gaze = observation['gaze'] - observation['p']
        heading = quaternion_rotate(q, torch.tensor([1., 0., 0.], device=q.device,
                                                   dtype=q.dtype).expand_as(acceleration))
        heading_xy, gaze_xy = heading[:, :2], gaze[:, :2]
        valid_heading = (gaze_xy.norm(dim=-1) > 1e-4) & (heading_xy.norm(dim=-1) > 1e-4)
        yaw_cross = heading_xy[:, 0] * gaze_xy[:, 1] - heading_xy[:, 1] * gaze_xy[:, 0]
        yaw_dot = (heading_xy * gaze_xy).sum(-1)
        # Mask atan2 operands first: masking its output still backpropagates
        # undefined 0/0 derivatives for exactly vertical mission targets.
        yaw_error = torch.atan2(torch.where(valid_heading, yaw_cross, torch.zeros_like(yaw_cross)),
                                torch.where(valid_heading, yaw_dot, torch.ones_like(yaw_dot)))
        yaw_world = z * (c.attitude_gain * yaw_error)[:, None]
        rates = rates + quaternion_inverse_rotate(q, yaw_world)
        thrust = c.nominal_mass * (force_per_mass * body_z).sum(-1)
        return torch.cat((thrust.clamp(0, c.max_thrust)[:, None],
                          rates.clamp(-c.max_body_rate, c.max_body_rate)), -1)

    def forward(self, observation):
        return self.action_from_acceleration(observation, self.desired_acceleration(observation))

    def _log_probability(self, distribution, latent):
        # Stable exact Jacobian for residual = max_accel * tanh(latent).
        log_jacobian = math.log(self.config.residual_acceleration) + 2 * (
            math.log(2) - latent - F.softplus(-2 * latent))
        return (distribution.log_prob(latent) - log_jacobian).sum(-1)

    def sample(self, observation):
        distribution, value = self.distribution(observation)
        latent = distribution.sample()
        acceleration = self.acceleration_from_latent(observation, latent)
        return (self.action_from_acceleration(observation, acceleration), latent,
                self._log_probability(distribution, latent), value)

    def evaluate_latent(self, observation, latent):
        distribution, value = self.distribution(observation)
        log_probability = self._log_probability(distribution, latent)
        # Monte Carlo entropy of the squashed residual, with reparameterization.
        fresh_latent = distribution.rsample()
        entropy = -self._log_probability(distribution, fresh_latent)
        return log_probability, entropy, value

    def configuration(self):
        return asdict(self.config)
