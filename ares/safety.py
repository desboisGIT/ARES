"""Observed-map, bounded-acceleration stopping shield (SI units, xyz, z up).

Certificates are conditional mathematical checks, not flight certification. See
``docs/safety_shield.md`` for the map, state-estimation and controller assumptions.
No ground-truth scene, simulator, tensor library or flight device is used here.
"""
from dataclasses import dataclass
import math

import numpy as np

SAFETY_VERSION = "ares.safety.v2"


@dataclass(frozen=True)
class ShieldConfig:
    body_radius_m: float = .15
    position_uncertainty_m: float = .03
    velocity_uncertainty_mps: float = .05
    control_interval_s: float = .02
    latency_s: float = .26
    max_acceleration_mps2: float = 5.
    min_braking_acceleration_mps2: float = 1.
    max_speed_mps: float = 4.
    acceleration_error_mps2: float = 0.
    terminal_speed_mps: float = 0.
    terminal_stop_radius_m: float = 0.

    def __post_init__(self):
        for key, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("body_radius_m", "control_interval_s", "max_acceleration_mps2",
                    "min_braking_acceleration_mps2", "max_speed_mps"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.terminal_speed_mps > 0 and self.terminal_stop_radius_m <= 0:
            raise ValueError("positive terminal speed requires a positive terminal stop radius")
        if self.min_braking_acceleration_mps2 > self.max_acceleration_mps2:
            raise ValueError("minimum braking acceleration exceeds acceleration bound")


@dataclass(frozen=True)
class SweptSegment:
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius_m: float


@dataclass(frozen=True)
class ShieldDecision:
    safe_acceleration: np.ndarray
    accepted: bool
    certified: bool
    state_viable: bool
    already_unsafe: bool
    emergency: bool
    reason: str
    stopping_distance_m: float
    checked_segments: tuple[SweptSegment, ...]
    candidate_fraction: float

    @property
    def desired_acceleration(self):
        """Net world-frame acceleration, before adding gravity in the controller."""
        return self.safe_acceleration.copy()


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return result


class SafetyShield:
    """Check one control period followed by a robust stop/terminal-hold envelope.

    Acceleration candidates blend the proposal toward active braking. When no
    candidate fits, return the braking request with ``certified=False``; never
    substitute hover for braking a moving vehicle. ``already_unsafe`` means the
    immediate braking envelope is uncertified, not that every escape is impossible.
    """

    def __init__(self, config=None):
        self.config = config or ShieldConfig()

    def _emergency_acceleration(self, velocity):
        speed = float(np.linalg.norm(velocity))
        c = self.config
        if c.terminal_speed_mps > 0:
            gain = min(c.min_braking_acceleration_mps2 / c.terminal_speed_mps,
                       1 / c.control_interval_s)
            return -velocity * min(gain, c.min_braking_acceleration_mps2 / max(speed, 1e-12))
        return (-velocity * c.min_braking_acceleration_mps2 / speed
                if speed > 1e-12 else np.zeros(3))

    def _envelope(self, position, velocity, command=None):
        c = self.config
        p, v = position.copy(), velocity.copy()
        ep, ev = c.position_uncertainty_m, c.velocity_uncertainty_mps
        segments = []
        peak_speed = float(np.linalg.norm(v)) + ev

        def segment(end, radius):
            # The legacy complete-stop tube contains every active-braking
            # prefix. Minkowski expansion by R additionally encloses every
            # allowed terminal trajectory within R of its prefix endpoint.
            segments.append(SweptSegment(tuple(p), tuple(end),
                                         c.body_radius_m + radius + c.terminal_stop_radius_m))

        def delay():
            nonlocal p, ep, ev, peak_speed
            t = c.latency_s
            end = p + v*t
            segment(end, ep + ev*t + .5*c.max_acceleration_mps2*t*t)
            ep += ev*t + .5*c.max_acceleration_mps2*t*t
            ev += c.max_acceleration_mps2*t
            p = end
            peak_speed = max(peak_speed, float(np.linalg.norm(v)) + ev)

        delay()
        if command is not None:
            t, error = c.control_interval_s, c.acceleration_error_mps2
            end = p + v*t + .5*command*t*t
            # The entire parabola lies within |a| t²/8 of its chord.
            segment(end, ep + ev*t + .5*error*t*t + np.linalg.norm(command)*t*t/8)
            ep += ev*t + .5*error*t*t
            ev += error*t
            v = v + command*t
            p = end
            peak_speed = max(peak_speed, float(np.linalg.norm(v)) + ev)
            # If the next check fails, braking itself may have another full delay.
            delay()
        speed = float(np.linalg.norm(v))
        alpha = (speed + ev) / (2*c.min_braking_acceleration_mps2)
        segment(p + v*alpha, ep + ev*alpha)
        # Includes state/latency uncertainty, but not body radius.
        distance = (float(np.linalg.norm(p-position)) + ep
                    + (speed+ev)**2/(2*c.min_braking_acceleration_mps2)
                    + c.terminal_stop_radius_m)
        return tuple(segments), peak_speed, distance

    @staticmethod
    def _segment_free(grid, origin, resolution, segment):
        a, b = np.asarray(segment.start), np.asarray(segment.end)
        radius = segment.radius_m
        lo = np.minimum(a, b)-radius
        hi = np.maximum(a, b)+radius
        bound = origin + np.asarray(grid.shape)*resolution
        # Contact is blocked, including contact with the edge of map support.
        if np.any(lo <= origin) or np.any(hi >= bound):
            return False
        first = np.maximum(0, np.floor(np.nextafter((lo-origin)/resolution, -np.inf)).astype(int))
        last = np.minimum(np.asarray(grid.shape)-1, np.floor((hi-origin)/resolution).astype(int))
        slices = tuple(slice(int(x), int(y)+1) for x, y in zip(first, last))
        blocked = np.argwhere(grid[slices] != 1) + first
        if len(blocked) == 0:
            return True
        # Minkowski inflate each blocked voxel by an axis-aligned cube containing
        # the physical/uncertainty sphere. Slab intersection checks the ENTIRE
        # segment; neither thin walls nor diagonal corner contacts can be skipped.
        lower = origin + blocked*resolution-radius
        upper = lower + resolution+2*radius
        direction = b-a
        near = np.zeros(len(blocked))
        far = np.ones(len(blocked))
        for axis in range(3):
            if abs(direction[axis]) < 1e-15:
                invalid = (a[axis] < lower[:, axis]) | (a[axis] > upper[:, axis])
                far[invalid] = -1.
            else:
                ta = (lower[:, axis]-a[axis])/direction[axis]
                tb = (upper[:, axis]-a[axis])/direction[axis]
                near = np.maximum(near, np.minimum(ta, tb))
                far = np.minimum(far, np.maximum(ta, tb))
        return not bool(np.any(near <= far + 1e-12))

    def filter(self, observed_grid, position, velocity, proposed_acceleration, *,
               origin=(0., 0., 0.), voxel_size=.2, assumptions_valid=True):
        """Return a conditional decision using only a 0 unknown/1 free/2 occupied map.

        ``assumptions_valid`` must be false when the platform cannot enforce the
        configured tracking, latency, braking/terminal-hold or estimation bounds (for
        example during an unvalidated attitude recovery). It prevents certification
        even if geometry fits. The caller must log this and retain a failsafe.
        """
        grid = np.asarray(observed_grid)
        if grid.ndim != 3 or any(n == 0 for n in grid.shape) or not np.isin(grid, (0, 1, 2)).all():
            raise ValueError("observed_grid must be a nonempty xyz grid with values 0, 1, 2")
        if not math.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError("voxel_size must be finite and positive")
        p, v = _vector(position, "position"), _vector(velocity, "velocity")
        a, o = _vector(proposed_acceleration, "proposed_acceleration"), _vector(origin, "origin")
        c = self.config
        brake = self._emergency_acceleration(v)
        immediate, peak, distance = self._envelope(p, v)
        viable = peak <= c.max_speed_mps + 1e-12 and all(
            self._segment_free(grid, o, voxel_size, s) for s in immediate)
        if not assumptions_valid or not viable:
            return ShieldDecision(brake, False, False, bool(viable), not viable, True,
                                  "assumptions_invalid" if not assumptions_valid else "no_braking_certificate",
                                  distance, immediate, 0.)
        for fraction in (1., .75, .5, .25):
            candidate = fraction*a + (1-fraction)*brake
            if np.linalg.norm(candidate) + c.acceleration_error_mps2 > c.max_acceleration_mps2 + 1e-12:
                continue
            segments, peak, _ = self._envelope(p, v, candidate)
            if peak <= c.max_speed_mps + 1e-12 and all(
                    self._segment_free(grid, o, voxel_size, s) for s in segments):
                return ShieldDecision(candidate.copy(), fraction == 1., True, True, False, False,
                                      "accepted" if fraction == 1. else "limited", distance, segments, fraction)
        return ShieldDecision(brake, False, True, True, False, True,
                              "emergency_brake", distance, immediate, 0.)
