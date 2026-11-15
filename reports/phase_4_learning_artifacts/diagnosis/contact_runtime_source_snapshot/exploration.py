"""Classical exploration from an observed voxel map, without simulator access.

The policy uses frontier-adjacent viewpoints, ray-based information gain and
six-connected routing.  These are independent implementations of classical
planning ideas; no FUEL/FALCON implementation is included.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class ExplorationConfig:
    """Public, scene-independent planner settings (SI units)."""

    cruise_speed_m_s: float = 1.8
    viewpoint_spacing_m: float = .6
    maximum_candidates: int = 100
    minimum_view_gain: float = 5.
    max_viewpoint_visits: int = 2
    stagnation_scans: int = 16
    minimum_new_voxels: int = 3
    scan_rate_rad_s: float = 1.4
    mission_budget_s: float = 3300.
    landing_dwell_s: float = 1.1
    gain_azimuth_samples: int = 64
    gain_elevation_samples: int = 9

    def __post_init__(self):
        for name in ("cruise_speed_m_s", "viewpoint_spacing_m", "scan_rate_rad_s",
                     "mission_budget_s", "landing_dwell_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("maximum_candidates", "max_viewpoint_visits", "gain_azimuth_samples",
                     "gain_elevation_samples", "stagnation_scans", "minimum_new_voxels"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.minimum_view_gain) or self.minimum_view_gain < 0:
            raise ValueError("minimum_view_gain must be finite and nonnegative")


def body_clear_mask(known_grid, resolution, body_radius_m):
    """Center positions whose complete spherical hull is observed free.

    Inflation uses voxel AABBs, not center distances. At 20 cm resolution a
    15 cm sphere intersects 19 cells, including the twelve edge neighbors.
    Unknown cells and space outside the grid are blocked.
    """
    grid = np.asarray(known_grid)
    if grid.ndim != 3 or min(grid.shape) < 1:
        raise ValueError("known_grid must be a nonempty XYZ array")
    if resolution <= 0 or body_radius_m < 0:
        raise ValueError("invalid resolution or radius")
    extent = int(math.ceil(body_radius_m / resolution + .5))
    offsets = np.indices((2 * extent + 1,) * 3).transpose(1, 2, 3, 0) - extent
    distance = np.maximum(np.abs(offsets) - .5, 0) * resolution
    structure = np.square(distance).sum(axis=-1) <= body_radius_m ** 2 + 1e-14
    return ndimage.binary_erosion(grid == 1, structure=structure, border_value=0)


def segment_is_known_free(known_grid, start, end, resolution, origin, body_radius_m):
    """Exact swept-sphere/voxel test, treating unknown and out-of-map as blocked."""
    grid = np.asarray(known_grid)
    start, end, origin = (np.asarray(value, dtype=float) for value in (start, end, origin))
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in (start, end, origin)):
        return False
    if resolution <= 0 or body_radius_m < 0:
        return False
    lo = np.floor((np.minimum(start, end) - body_radius_m - origin) / resolution).astype(int)
    hi = np.floor((np.maximum(start, end) + body_radius_m - origin) / resolution).astype(int)
    if np.any(lo < 0) or np.any(hi >= grid.shape):
        return False
    cells = np.indices(tuple(hi - lo + 1)).reshape(3, -1).T + lo
    cells = cells[grid[tuple(cells.T)] != 1]
    if not len(cells):
        return True
    lower = origin + cells * resolution
    upper = lower + resolution
    direction = end - start
    # Squared distance to a box is piecewise quadratic. Box-face crossing
    # parameters partition the segment into intervals with fixed active faces.
    crossings = np.zeros((len(cells), 8))
    crossings[:, 1] = 1.
    for axis in range(3):
        if abs(direction[axis]) > 1e-14:
            crossings[:, 2 + axis * 2] = (lower[:, axis] - start[axis]) / direction[axis]
            crossings[:, 3 + axis * 2] = (upper[:, axis] - start[axis]) / direction[axis]
    crossings = np.sort(np.clip(crossings, 0., 1.), axis=1)
    left, right = crossings[:, :-1], crossings[:, 1:]
    middle = (left + right) * .5
    points = start + middle[..., None] * direction
    below, above = points < lower[:, None], points > upper[:, None]
    active = below | above
    intercept = np.where(below, start - lower[:, None], np.where(above, start - upper[:, None], 0.))
    slope = active * direction
    denominator = np.square(slope).sum(axis=-1)
    optimum = np.divide(-(intercept * slope).sum(axis=-1), denominator,
                        out=np.zeros_like(denominator), where=denominator > 0)
    optimum = np.clip(optimum, left, right)
    points = start + optimum[..., None] * direction
    delta = np.maximum(np.maximum(lower[:, None] - points, points - upper[:, None]), 0.)
    distance2 = np.square(delta).sum(axis=-1).min(axis=1)
    # Tangent contact is conservatively blocked, consistent with hull inflation.
    return bool(np.all(distance2 > body_radius_m ** 2 + 1e-14))


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _reachable_paths(safe, start):
    """Six-connected, equal-edge-cost global routing with parent pointers."""
    shape = safe.shape
    parents = np.full(shape, -1, dtype=np.int64)
    distances = np.full(shape, -1, dtype=np.int32)
    if not safe[tuple(start)]:
        return distances, parents
    queue = deque([tuple(int(v) for v in start)])
    distances[tuple(start)] = 0
    while queue:
        cell = queue.popleft()
        cost = distances[cell] + 1
        flat = np.ravel_multi_index(cell, shape)
        for axis in range(3):
            for sign in (-1, 1):
                neighbor = list(cell)
                neighbor[axis] += sign
                if not 0 <= neighbor[axis] < shape[axis]:
                    continue
                neighbor = tuple(neighbor)
                if safe[neighbor] and distances[neighbor] < 0:
                    parents[neighbor] = flat
                    distances[neighbor] = cost
                    queue.append(neighbor)
    return distances, parents


class ClassicalExplorer:
    """Observed-map-only explorer with the fast-world five-value action API.

    Construction accepts configuration only. ``act`` reads only documented
    sensor/map and estimated-pose fields; scene labels, seed, ground truth and
    scorer coverage are neither accepted by the constructor nor inspected.
    """

    policy_name = "classical_frontier_viewgain"
    policy_version = "ares.classical_explorer.v1"

    def __init__(self, config: ExplorationConfig | None = None):
        self.config = ExplorationConfig() if config is None else config
        if not isinstance(self.config, ExplorationConfig):
            raise TypeError("config must be ExplorationConfig")
        self.state = "initial_scan"
        self._scan_angle = 0.
        self._last_yaw = None
        self._route = []
        self._goal_yaw = 0.
        self._visits = {}
        self._active_view_key = None
        self._last_observed_count = 0
        self._stagnant_scans = 0
        self._land_since = None
        self._last_plan_s = -math.inf
        self.diagnostics = {"plans": 0, "viewpoints_scanned": 0, "return_reason": None}

    def _view_key(self, point):
        return tuple(np.floor(np.asarray(point) / self.config.viewpoint_spacing_m + .5).astype(int))

    def _gain(self, grid, cell, resolution, camera):
        """Unknown ray samples before observed surfaces, across gaze directions."""
        count = self.config.gain_azimuth_samples
        azimuth = np.arange(count) * (2 * math.pi / count)
        elevation = np.linspace(-camera[1] * .49, camera[1] * .49,
                                self.config.gain_elevation_samples)
        horizontal = np.stack((np.cos(azimuth), np.sin(azimuth)), axis=-1)
        rays = np.empty((count, len(elevation), 3))
        rays[..., :2] = horizontal[:, None]
        rays[..., 2] = np.tan(elevation)[None, :]
        rays = rays.reshape(-1, 3)
        sector = np.repeat(np.arange(count), len(elevation))
        coordinates = np.tile(np.asarray(cell, dtype=int), (len(rays), 1))
        step = np.sign(rays).astype(int)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.where(np.abs(rays) > 1e-12, resolution / np.abs(rays), np.inf)
        crossing = delta * .5
        distance = np.zeros(len(rays))
        active = np.ones(len(rays), dtype=bool)
        unknown_hits = []
        # Voxel DDA is important here: fixed-distance samples can skip a thin
        # intersection at a wall corner, predicting gain through solid walls.
        for _ in range(sum(grid.shape) + 3):
            active &= (distance <= camera[2]) & np.all((coordinates >= 0) & (coordinates < grid.shape), axis=1)
            ids = np.flatnonzero(active)
            if not len(ids):
                break
            labels = grid[tuple(coordinates[ids].T)]
            unknown = ids[labels == 0]
            if len(unknown):
                flat = np.ravel_multi_index(tuple(coordinates[unknown].T), grid.shape)
                unknown_hits.append(sector[unknown] * grid.size + flat)
            active[ids[labels == 2]] = False
            ids = ids[labels != 2]
            if not len(ids):
                continue
            next_distance = crossing[ids].min(axis=1)
            axes = crossing[ids] <= next_distance[:, None] + 1e-10
            coordinates[ids] += axes * step[ids]
            crossing[ids] += np.where(axes, delta[ids], 0.)
            distance[ids] = next_distance
        sector_gain = np.zeros(count)
        if unknown_hits:
            unique_hits = np.unique(np.concatenate(unknown_hits))
            sector_gain = np.bincount(unique_hits // grid.size, minlength=count).astype(float)
        half = max(1, int(camera[0] * count / (4 * math.pi)))
        gains = sum(np.roll(sector_gain, shift) for shift in range(-half, half + 1))
        best = int(np.argmax(gains))
        return float(gains[best]), float(azimuth[best])

    def _plan(self, observation, returning=False):
        grid = np.asarray(observation["known_grid"])
        position = np.asarray(observation["position"], dtype=float)
        resolution, origin = float(observation["resolution"]), np.asarray(observation["origin"], dtype=float)
        radius = float(observation["body_radius_m"])
        safe = body_clear_mask(grid, resolution, radius)
        cell = np.floor((position - origin) / resolution).astype(int)
        if np.any(cell < 0) or np.any(cell >= grid.shape):
            return False
        if not safe[tuple(cell)]:
            # A noncentral start can be safe while its voxel center is not.
            # Attach only to a nearby center with a completely known-safe link.
            centers = np.argwhere(safe)
            if not len(centers):
                return False
            ordered = np.argsort(np.square(origin + (centers + .5) * resolution - position).sum(axis=1))[:32]
            for index in ordered:
                candidate = centers[index]
                endpoint = origin + (candidate + .5) * resolution
                if segment_is_known_free(grid, position, endpoint, resolution, origin, radius):
                    cell = candidate
                    break
            else:
                return False
        distances, parents = _reachable_paths(safe, cell)
        self.diagnostics["plans"] += 1
        self._last_plan_s = float(observation["time_s"])
        if returning:
            home = np.asarray(observation["home"], dtype=float)
            target = np.floor((home - origin) / resolution).astype(int)
            if np.any(target < 0) or np.any(target >= grid.shape) or distances[tuple(target)] < 0:
                return False
            best = target
        else:
            candidates = np.argwhere(distances >= 0)
            if not len(candidates):
                return False
            stride = max(1, int(round(self.config.viewpoint_spacing_m / resolution)))
            # One deterministic center per spatial bin, including every height.
            bins = candidates // stride
            _, indices = np.unique(bins, axis=0, return_index=True)
            candidates = candidates[indices]
            unknown_distance = ndimage.distance_transform_edt(grid != 0) * resolution
            candidates = candidates[unknown_distance[tuple(candidates.T)] <= 4.8]
            if not len(candidates):
                return False
            points = origin + (candidates + .5) * resolution
            visits = np.array([self._visits.get(self._view_key(point), 0) for point in points])
            keep = visits < self.config.max_viewpoint_visits
            candidates, points, visits = candidates[keep], points[keep], visits[keep]
            if not len(candidates):
                return False
            costs = distances[tuple(candidates.T)] * resolution
            # Preserve spatial diversity instead of truncating only to nearby
            # frontiers. A deterministic evenly-spaced shortlist covers the map.
            order = np.argsort(costs + visits * 2., kind="stable")
            near_count = self.config.maximum_candidates // 2
            near = order[:near_count]
            rest = order[near_count:]
            if len(rest):
                rest = rest[np.linspace(0, len(rest) - 1, min(len(rest), self.config.maximum_candidates - len(near)), dtype=int)]
            shortlist = np.concatenate((near, rest))
            camera_info = observation.get("camera", {})
            camera = (float(camera_info.get("hfov_rad", math.radians(87))),
                      float(camera_info.get("vfov_rad", math.radians(58))),
                      float(camera_info.get("max_range_m", 4.)))
            best, best_score = None, -math.inf
            for index in shortlist:
                candidate = candidates[index]
                gain, yaw = self._gain(grid, candidate, resolution, camera)
                if gain < self.config.minimum_view_gain:
                    continue
                score = gain / (2. + costs[index]) / (1. + visits[index] * 3.)
                if score > best_score:
                    best, best_score, self._goal_yaw = candidate, score, yaw
            if best is None:
                return False
            self._active_view_key = self._view_key(origin + (best + .5) * resolution)
        route = [tuple(int(value) for value in best)]
        while route[-1] != tuple(cell):
            parent = int(parents[route[-1]])
            if parent < 0:
                return False
            route.append(np.unravel_index(parent, grid.shape))
        route.reverse()
        # Keep only ends of straight runs; no diagonal corner cutting.
        compressed = [route[0]]
        for index in range(1, len(route) - 1):
            if tuple(np.subtract(route[index], route[index - 1])) != tuple(np.subtract(route[index + 1], route[index])):
                compressed.append(route[index])
        if len(route) > 1:
            compressed.append(route[-1])
        self._route = [origin + (np.asarray(value) + .5) * resolution for value in compressed]
        if returning:
            self._route.append(np.asarray(observation["home"], dtype=float).copy())
        return True

    def _scan(self, observation):
        yaw = float(observation["yaw"])
        if self._last_yaw is not None:
            self._scan_angle += abs(_wrap(yaw - self._last_yaw))
        self._last_yaw = yaw
        if self._scan_angle >= 2 * math.pi - .04:
            self._scan_angle = 0.
            self._last_yaw = None
            key = self._active_view_key or self._view_key(observation["position"])
            self._visits[key] = self._visits.get(key, 0) + 1
            self._active_view_key = None
            self.diagnostics["viewpoints_scanned"] += 1
            observed_count = int(np.count_nonzero(observation["known_grid"]))
            if observed_count - self._last_observed_count < self.config.minimum_new_voxels:
                self._stagnant_scans += 1
            else:
                self._stagnant_scans = 0
            self._last_observed_count = observed_count
            return None
        return np.array([0., 0., 0., min(self.config.scan_rate_rad_s, float(observation["max_yaw_rate"])), 0.])

    def act(self, observation: dict) -> np.ndarray:
        """Return world vx/vy/vz, yaw rate, land flag from public observation."""
        position = np.asarray(observation["position"], dtype=float)
        now = float(observation["time_s"])
        if self.state == "landed":
            return np.array([0., 0., 0., 0., 1.])
        if now >= self.config.mission_budget_s and self.state not in ("return", "landing"):
            self.state = "return"
            self._route = []
            self.diagnostics["return_reason"] = "policy_time_budget"
        if self.state in ("initial_scan", "scan"):
            action = self._scan(observation)
            if action is not None:
                return action
            self.state = "explore"
            if self._stagnant_scans >= self.config.stagnation_scans:
                self.state = "return"
                self.diagnostics["return_reason"] = "observed_map_stagnation"
        if self.state == "explore" and not self._route:
            if not self._plan(observation):
                self.state = "return"
                self.diagnostics["return_reason"] = "observed_viewpoints_exhausted"
        if self.state == "return" and not self._route:
            if np.linalg.norm(position - np.asarray(observation["home"])) < .025:
                self.state = "landing"
            elif not self._plan(observation, returning=True):
                # New observations from a full scan may reconnect a conservative
                # known-free route; never move blindly to the landing point.
                return np.array([0., 0., 0., min(self.config.scan_rate_rad_s, float(observation["max_yaw_rate"])), 0.])
        if self.state == "landing":
            if np.linalg.norm(np.asarray(observation["velocity"])) > .03:
                self._land_since = None
            elif self._land_since is None:
                self._land_since = now
            elif now - self._land_since >= self.config.landing_dwell_s:
                self.state = "landed"
                return np.array([0., 0., 0., 0., 1.])
            return np.zeros(5)
        while self._route and np.linalg.norm(self._route[0] - position) < .008:
            self._route.pop(0)
        if not self._route:
            if self.state == "return":
                self.state = "landing"
            else:
                self.state = "scan"
                self._scan_angle = 0.
                self._last_yaw = None
            return np.zeros(5)
        offset = self._route[0] - position
        distance = float(np.linalg.norm(offset))
        dt = float(observation["dt"])
        speed = min(self.config.cruise_speed_m_s, float(observation["max_speed"]), distance / dt)
        desired_yaw = math.atan2(offset[1], offset[0]) if np.linalg.norm(offset[:2]) > .025 else self._goal_yaw
        yaw_error = _wrap(desired_yaw - float(observation["yaw"]))
        yaw_rate = float(np.clip(yaw_error / dt, -float(observation["max_yaw_rate"]), float(observation["max_yaw_rate"])))
        # Give the forward camera time to reveal the next corridor before flight.
        speed *= max(.15, math.cos(min(abs(yaw_error), math.pi / 2)))
        velocity = offset / distance * speed
        if not segment_is_known_free(observation["known_grid"], position, position + velocity * dt,
                                     float(observation["resolution"]), observation["origin"],
                                     float(observation["body_radius_m"])):
            self._route = []
            if self.state == "explore":
                self.state = "scan"
                self._scan_angle = 0.
                self._last_yaw = None
            return np.array([0., 0., 0., yaw_rate or min(1., float(observation["max_yaw_rate"])), 0.])
        return np.array([*velocity, yaw_rate, 0.])


def extract_rooms(known_grid, resolution, origin):
    """Infer axis-aligned room footprints from observed above-door free space.

    The 2.3 m layer is an explicit dataset prior, not hidden scene metadata.
    Separating walls above doors give independent free-space components. Bounding
    rectangles are geometric inferences only; this function never fills unknown
    voxels in a report. Components lacking sufficient evidence are omitted.
    """
    grid = np.asarray(known_grid)
    origin = np.asarray(origin, dtype=float)
    if grid.ndim != 3 or resolution <= 0 or origin.shape != (3,):
        raise ValueError("invalid observed map geometry")
    layer = int(math.floor(2.3 / resolution))
    if layer >= grid.shape[2]:
        return []
    labels, count = ndimage.label(grid[:, :, layer] == 1)
    rooms = []
    for component, slices in enumerate(ndimage.find_objects(labels), 1):
        if slices is None:
            continue
        x, y = slices
        mask = labels[x, y] == component
        if mask.sum() * resolution ** 2 < .5 or min(x.stop - x.start, y.stop - y.start) < 3:
            continue
        xmin, ymin = origin[:2] + np.array([x.start, y.start]) * resolution
        xmax, ymax = origin[:2] + np.array([x.stop, y.stop]) * resolution
        rooms.append({"room_id": f"observed_room_{len(rooms)}", "polygon":
                      [[float(xmin), float(ymin)], [float(xmax), float(ymin)],
                       [float(xmax), float(ymax)], [float(xmin), float(ymax)]]})
    return rooms
