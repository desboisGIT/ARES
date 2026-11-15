"""Pure mission guidance from public position, two waypoints and a speed limit.

The braking budget is a developer assumption, not a measurement of free space
or a guarantee about any particular airframe. See docs/flight_guidance.md.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GuidanceConfig:
    minimum_deceleration: float = .8
    reaction_time: float = .4
    waypoint_radius: float = .10
    lookahead_distance: float = .20
    corner_speed: float = .20
    reversal_speed: float = .10

    def __post_init__(self):
        values = (self.minimum_deceleration, self.reaction_time, self.waypoint_radius,
                  self.lookahead_distance, self.corner_speed, self.reversal_speed)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError('guidance parameters must be finite and positive')
        if self.reversal_speed > self.corner_speed:
            raise ValueError('reversal speed must not exceed right-angle corner speed')


def guidance_velocity(p, wp1, wp2, speed_limit, config=None):
    """Return differentiable world velocity without changing controller state.

    Equal waypoints mean there is no following leg. Incoming direction is kept
    until mission handoff, avoiding shortcuts across a corner. Corner speed is
    reduced before the waypoint acceptance radius, using the next leg's angle.
    Scalar, B, and Bx1 speed limits broadcast over matching ...x3 pose tensors.
    """
    c = config if isinstance(config, GuidanceConfig) else GuidanceConfig(**(config or {}))
    if p.shape != wp1.shape or p.shape != wp2.shape or p.shape[-1:] != (3,):
        raise ValueError('position and waypoints must have matching ...x3 shapes')
    limit = torch.as_tensor(speed_limit, device=p.device, dtype=p.dtype)
    if limit.ndim == p.ndim-1:
        limit = limit.unsqueeze(-1)
    limit = torch.broadcast_to(limit, p.shape[:-1]+(1,)).clamp_min(0)
    incoming, outgoing = wp1-p, wp2-wp1
    distance = incoming.norm(dim=-1, keepdim=True)
    next_distance = outgoing.norm(dim=-1, keepdim=True)
    direction = incoming/distance.clamp_min(1e-6)
    next_direction = outgoing/next_distance.clamp_min(1e-6)
    cosine = (direction*next_direction).sum(-1, keepdim=True).clamp(-1, 1)
    # A bounded rational angle rule avoids acos/sqrt singularities at exact
    # straight, reversing, vertical and coincident waypoint configurations.
    angle_ratio = (1+cosine)/(1-cosine+1e-6)
    turn_speed = c.reversal_speed+(c.corner_speed-c.reversal_speed)*angle_ratio
    turn_speed = torch.minimum(turn_speed, limit)
    endpoint = next_distance <= 1e-6
    turn_speed = torch.where(endpoint, torch.zeros_like(turn_speed), turn_speed)
    available = torch.where(endpoint, distance,
                            (distance-c.waypoint_radius-c.lookahead_distance).clamp_min(0))
    reaction = c.minimum_deceleration*c.reaction_time
    braking_cap = (reaction*reaction+turn_speed.square()+
                   2*c.minimum_deceleration*available).sqrt()-reaction
    # At or below handoff speed no further incoming deceleration is requested.
    # A small reversal floor lets an ideal tracker cross the acceptance radius.
    speed = torch.minimum(limit, torch.maximum(turn_speed, braking_cap))
    return direction*speed
