"""Observed-state flight controller; no sampled airframe parameters enter this API.

The observer is advanced once per physical step. Its explicit tensor snapshot is
part of an observation; action conversion never advances it, including during PPO
re-evaluation. This is a simulation controller, not a flight certification.
"""
from dataclasses import dataclass, fields
import math

import torch


@dataclass(frozen=True)
class ControllerConfig:
    nominal_mass: float = .5
    gravity: float = 9.80665
    max_thrust: float = 24.5
    max_body_rate: float = 6.
    max_yaw_rate: float = 1.
    attitude_gain: float = 6.
    nominal_motor_lag_s: float = .025
    nominal_delay_s: float = .0125
    observer_bandwidth: float = 10.
    max_disturbance: float = 4.
    stop_velocity_gain: float = 3.
    hold_position_gain: float = 1.5
    hold_velocity_gain: float = 2.5
    hold_enter_speed: float = .10
    hold_exit_speed: float = .25

    def __post_init__(self):
        values = [getattr(self, f.name) for f in fields(self)]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError('controller parameters must be finite and positive')
        if self.hold_exit_speed <= self.hold_enter_speed:
            raise ValueError('hold exit speed must exceed entry speed')


def _config(config):
    return config if isinstance(config, ControllerConfig) else ControllerConfig(**(config or {}))


def _rotate(q, vector):
    cross = torch.linalg.cross(q[..., :3], vector, dim=-1)
    return vector + 2 * (q[..., 3:] * cross + torch.linalg.cross(q[..., :3], cross, dim=-1))


def _inverse_rotate(q, vector):
    return _rotate(torch.cat((-q[..., :3], q[..., 3:]), -1), vector)


def _bounded(vector, bound):
    return vector * (bound / vector.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp_max(1)


@dataclass(frozen=True)
class ControllerMemory:
    velocity: torch.Tensor
    bias: torch.Tensor
    delayed_thrust: torch.Tensor
    motor_thrust: torch.Tensor
    previous_quaternion: torch.Tensor
    hold_anchor: torch.Tensor
    holding: torch.Tensor
    stopping: torch.Tensor

    def observation_state(self):
        """Return detached snapshots, safe to retain in a rollout buffer."""
        return {name: value.detach().clone() for name, value in (
            ('controller_bias', self.bias), ('controller_velocity', self.velocity),
            ('controller_hold_anchor', self.hold_anchor),
            ('controller_holding', self.holding), ('controller_stopping', self.stopping))}


@torch.no_grad()
def initialize_memory(observation, config=None):
    """Initialize from public p/v/q only; hover thrust starts at nominal weight."""
    c = _config(config)
    velocity = observation['v'].detach().clone()
    weight = torch.full_like(velocity[:, 0], c.nominal_mass * c.gravity)
    false = torch.zeros_like(weight, dtype=torch.bool)
    return ControllerMemory(velocity, torch.zeros_like(velocity), weight, weight.clone(),
                            observation['q'].detach().clone(),
                            observation['p'].detach().clone(), false, false.clone())


def _reset_lanes(memory, observation, reset_mask, config):
    fresh = initialize_memory(observation, config)
    reset = torch.as_tensor(reset_mask, device=memory.velocity.device, dtype=torch.bool)
    return ControllerMemory(*(torch.where(
        reset.reshape((-1,) + (1,) * (getattr(memory, f.name).ndim - 1)),
        getattr(fresh, f.name), getattr(memory, f.name)) for f in fields(memory)))


@torch.no_grad()
def update_memory(memory, observation, previous_action, dt, config=None, *,
                  reset_mask=None, stop_requested=None):
    """Advance an innovation observer using the command sent over this interval.

    ``previous_action`` is the public SI command, never the plant's applied
    thrust or its initial sampled weight. A masked reset replaces every lane's
    observer and hold state. Stop mode persists when stop_requested is omitted;
    passing False explicitly releases it. Time intervals are public elapsed time.
    """
    c = _config(config)
    if not math.isfinite(dt) or dt < 0 or (dt == 0 and reset_mask is None):
        raise ValueError('positive finite observer interval required (zero only for masked reset)')
    if dt == 0:
        return _reset_lanes(memory, observation, reset_mask, c)
    velocity, q, p = observation['v'], observation['q'], observation['p']
    command = previous_action[:, 0].detach().clamp(0, c.max_thrust)
    # Two public nominal first-order actuator states approximate transport delay
    # and motor lag. Integrate the motor's mean thrust, not just its endpoint.
    delay_decay = math.exp(-dt / c.nominal_delay_s)
    delayed = command + (memory.delayed_thrust - command) * delay_decay
    delayed_mean = command + (memory.delayed_thrust - command) * (
        c.nominal_delay_s / dt * (1 - delay_decay))
    motor_decay = math.exp(-dt / c.nominal_motor_lag_s)
    motor = delayed_mean + (memory.motor_thrust - delayed_mean) * motor_decay
    mean_thrust = delayed_mean + (memory.motor_thrust - delayed_mean) * (
        c.nominal_motor_lag_s / dt * (1 - motor_decay))
    up = torch.zeros_like(velocity); up[:, 2] = 1
    # Average observed orientation over the interval; no angular truth needed.
    body_z = (_rotate(memory.previous_quaternion, up) + _rotate(q, up)) / 2
    prediction = memory.velocity + dt * (
        mean_thrust[:, None] / c.nominal_mass * body_z - c.gravity * up + memory.bias)
    innovation = velocity.detach() - prediction
    omega = c.observer_bandwidth
    alpha = 1 - math.exp(-2 * omega * dt)
    beta = omega * omega * dt / (1 + omega * dt)
    filtered_v = prediction + alpha * innovation
    # Bounded disturbance estimation supplies integral trim. There is no
    # requested-acceleration integrator to wind up behind actuator saturation.
    bias = _bounded(memory.bias + beta * innovation, c.max_disturbance)
    requested = memory.stopping if stop_requested is None else torch.as_tensor(
        stop_requested, device=velocity.device, dtype=torch.bool).expand_as(memory.stopping)
    speed = filtered_v.norm(dim=-1)
    holding = requested & torch.where(memory.holding,
                                      speed < c.hold_exit_speed, speed < c.hold_enter_speed)
    capture = holding & ~memory.holding
    anchor = torch.where(capture[:, None], p.detach(), memory.hold_anchor)
    result = ControllerMemory(filtered_v, bias, delayed, motor, q.detach().clone(),
                              anchor, holding, requested.clone())
    if reset_mask is not None:
        result = _reset_lanes(result, observation, reset_mask, c)
    return result


def acceleration_to_action(observation, acceleration, config=None):
    """Pure differentiable net world acceleration -> thrust N / body rad/s.

    The optional controller_bias snapshot estimates actual minus nominal
    acceleration. Subtracting it adapts hover and disturbance rejection without
    exposing mass, drag, gravity samples or actuator truth to the controller.
    """
    c = _config(config)
    q = observation['q']
    up = torch.zeros_like(acceleration); up[:, 2] = 1
    body_z = _rotate(q, up)
    force = acceleration + c.gravity * up - observation.get('controller_bias', torch.zeros_like(acceleration))
    desired_z = force / force.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    rates = c.attitude_gain * _inverse_rotate(q, torch.linalg.cross(body_z, desired_z, dim=-1))
    gaze = observation.get('gaze', observation['p']) - observation['p']
    forward = torch.zeros_like(acceleration); forward[:, 0] = 1
    heading = _rotate(q, forward)
    valid = (gaze[:, :2].norm(dim=-1) > 1e-4) & (heading[:, :2].norm(dim=-1) > 1e-4)
    cross = heading[:, 0] * gaze[:, 1] - heading[:, 1] * gaze[:, 0]
    dot = (heading[:, :2] * gaze[:, :2]).sum(-1)
    yaw = torch.atan2(torch.where(valid, cross, torch.zeros_like(cross)),
                      torch.where(valid, dot, torch.ones_like(dot)))
    # Limit only the world-vertical heading contribution. Geometric tilt
    # feedback keeps its authority before the final physical body-rate limits.
    yaw_rate = (c.attitude_gain * yaw).clamp(-c.max_yaw_rate, c.max_yaw_rate)
    rates = rates + _inverse_rotate(q, up * yaw_rate[:, None])
    thrust = c.nominal_mass * (force * body_z).sum(-1)
    return torch.cat((thrust.clamp(0, c.max_thrust)[:, None],
                      rates.clamp(-c.max_body_rate, c.max_body_rate)), -1)


def stop_acceleration(observation, max_acceleration=2., config=None):
    """Bounded smooth braking and captured position hold from public memory.

    Above the low-speed band this opposes filtered velocity. At low speed it
    smoothly tends to zero rather than issuing a unit brake with noisy sign.
    Once captured, position feedback keeps the stop anchor with hysteresis.
    """
    c = _config(config)
    if not math.isfinite(max_acceleration) or max_acceleration <= 0:
        raise ValueError('positive finite braking acceleration required')
    velocity = observation.get('controller_velocity', observation['v'])
    braking = -c.stop_velocity_gain * velocity
    holding = observation.get('controller_holding', torch.zeros_like(velocity[:, 0], dtype=torch.bool))
    anchor = observation.get('controller_hold_anchor', observation['p'])
    hold = c.hold_position_gain * (anchor - observation['p']) - c.hold_velocity_gain * velocity
    return _bounded(torch.where(holding[:, None], hold, braking), max_acceleration)
