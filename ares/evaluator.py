"""Evaluator-owned ground truth and an idealized Phase 0 kinematic harness.

Policies get reports; this module owns the scene and observation ledger. This is
an in-process trust boundary, not a sandbox for hostile arbitrary Python code.
It is intentionally not an onboard mapper, flight simulator, or depth camera.
"""

from collections import deque
from dataclasses import dataclass
from itertools import product
import copy
import hashlib
import json
import math

from .contracts import validate_scene, validate_profile, point_in_polygon

AXES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def segment_box_distance(a, b, lower, upper):
    """Exact Euclidean minimum distance between a segment and a closed AABB."""
    delta = [b[i] - a[i] for i in range(3)]
    breaks = {0.0, 1.0}
    for i in range(3):
        if delta[i]:
            for plane in (lower[i], upper[i]):
                t = (plane - a[i]) / delta[i]
                if 0 < t < 1:
                    breaks.add(t)
    ordered = sorted(breaks)

    def distance_sq(t):
        return sum(max(lower[i] - (a[i] + t * delta[i]),
                       (a[i] + t * delta[i]) - upper[i], 0.0) ** 2
                   for i in range(3))

    best = min(distance_sq(t) for t in ordered)
    for left, right in zip(ordered, ordered[1:]):
        middle = (left + right) / 2
        numerator = denominator = 0.0
        for i in range(3):
            value = a[i] + middle * delta[i]
            plane = lower[i] if value < lower[i] else upper[i] if value > upper[i] else None
            if plane is not None:
                numerator += delta[i] * (a[i] - plane)
                denominator += delta[i] ** 2
        if denominator:
            t = min(right, max(left, -numerator / denominator))
            best = min(best, distance_sq(t))
    return math.sqrt(best)


@dataclass(frozen=True)
class GroundTruthSample:
    timestamp_s: float
    position: tuple
    scan: bool


@dataclass(frozen=True)
class GroundTruthEpisode:
    samples: tuple
    observations: tuple
    collisions: int
    left_bounds: bool
    minimum_clearance_m: float
    landed_safely: bool
    seed: int
    world_digest: str


class GroundTruthWorld:
    """All ground truth access is explicit in this evaluator-only class name."""

    def __init__(self, scene, profile):
        validate_scene(scene)
        validate_profile(profile)
        self.scene = copy.deepcopy(scene)
        self.profile = copy.deepcopy(profile)
        self.digest = hashlib.sha256(json.dumps(
            {"scene": scene, "profile": profile}, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        grid = scene["grid"]
        self.resolution = grid["resolution"]
        self.origin = tuple(grid["origin"])
        self.shape = tuple(grid["shape"])
        self.occupied = frozenset(map(tuple, grid["occupied"]))
        self.radius = profile["parameters"]["body_radius_m"]["nominal"]
        self.sensor_range = profile["parameters"]["depth_max_range_m"]["nominal"]
        self.spawn = tuple(scene["spawn"])
        if not self.body_in_bounds(self.spawn) or self.clearance(self.spawn, self.spawn) < 0:
            raise ValueError("spawn does not have drone body clearance")
        self.reachable = self._reachable()
        # A denominator fixed before any policy runs. Six axial center rays are
        # an idealized scan, not a claim of D435i or full Phase 2 depth fidelity.
        self.observable = {}
        for cell in sorted(self.reachable):
            self.observable.update(self.axial_scan(cell))
        self.reachable_rooms = tuple(sorted(
            room["room_id"] for room in scene["rooms"]
            if any(point_in_polygon(self.center(cell)[:2], room["polygon"])
                   for cell in self.reachable)))

    def in_grid(self, cell):
        return all(0 <= cell[i] < self.shape[i] for i in range(3))

    def index(self, position):
        return tuple(math.floor((position[i] - self.origin[i]) / self.resolution) for i in range(3))

    def center(self, cell):
        return tuple(self.origin[i] + (cell[i] + 0.5) * self.resolution for i in range(3))

    def body_in_bounds(self, position):
        return all(self.origin[i] + self.radius <= position[i]
                   <= self.origin[i] + self.shape[i] * self.resolution - self.radius for i in range(3))

    def clearance(self, a, b):
        """Signed swept-sphere clearance, considering every occupied voxel.

        Exact geometry, no discrete-step tunnelling. The broad phase expands
        until excluded cells cannot contain a nearer obstacle.
        """
        pad = max(self.resolution * 2, self.radius)
        best = math.inf
        while True:
            lo = [max(0, math.floor((min(a[i], b[i]) - pad - self.origin[i]) / self.resolution)) for i in range(3)]
            hi = [min(self.shape[i] - 1, math.floor((max(a[i], b[i]) + pad - self.origin[i]) / self.resolution)) for i in range(3)]
            for cell in product(*(range(lo[i], hi[i] + 1) for i in range(3))):
                if cell in self.occupied:
                    lower = tuple(self.origin[i] + cell[i] * self.resolution for i in range(3))
                    upper = tuple(v + self.resolution for v in lower)
                    best = min(best, segment_box_distance(a, b, lower, upper))
            # Any excluded voxel is at least pad away along an axis. Once a
            # closer obstacle is found this is the global exact minimum.
            if best <= pad or all(lo[i] == 0 and hi[i] == self.shape[i] - 1 for i in range(3)):
                return best - self.radius
            pad *= 2

    def _reachable(self):
        start = self.index(self.spawn)
        if self.clearance(self.spawn, self.center(start)) < 0:
            raise ValueError("spawn cannot reach its voxel center")
        queue = deque([start])
        found = {start}
        while queue:
            here = queue.popleft()
            for direction in AXES:
                nxt = tuple(here[i] + direction[i] for i in range(3))
                if nxt in found or not self.in_grid(nxt) or nxt in self.occupied:
                    continue
                a, b = self.center(here), self.center(nxt)
                if self.body_in_bounds(b) and self.clearance(a, b) >= -1e-12:
                    found.add(nxt)
                    queue.append(nxt)
        return frozenset(found)

    def axial_scan(self, cell):
        if not self.in_grid(cell) or cell in self.occupied:
            return {}
        result = {cell: 1}
        for direction in AXES:
            for distance in range(1, math.floor(self.sensor_range / self.resolution) + 1):
                nxt = tuple(cell[i] + direction[i] * distance for i in range(3))
                if not self.in_grid(nxt):
                    break
                result[nxt] = 2 if nxt in self.occupied else 1
                if nxt in self.occupied:
                    break
        return result

    def record(self, samples, *, seed, land=False):
        """Create trusted telemetry from evaluator-executed kinematic actions.

        Every sample is checked, including the complete segment between samples.
        A landing event requires a stationary terminal dwell near the spawn pad;
        Phase 0 uses a virtual home landing station at spawn height.
        """
        if type(seed) is not int or type(land) is not bool or not samples:
            raise ValueError("record requires an integer seed, boolean land, and samples")
        previous = None
        observed = {}
        collisions = 0
        left_bounds = False
        min_clearance = math.inf
        checked = []
        for sample in samples:
            if not isinstance(sample, GroundTruthSample):
                raise ValueError("samples must be GroundTruthSample instances")
            t, pos = sample.timestamp_s, sample.position
            if (isinstance(t, bool) or not isinstance(t, (float, int)) or not math.isfinite(t)
                    or t < 0 or len(pos) != 3 or type(sample.scan) is not bool
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in pos)):
                raise ValueError("invalid telemetry sample")
            sample = GroundTruthSample(float(t), tuple(pos), sample.scan)
            if previous is None:
                if t != 0 or math.dist(pos, self.spawn) > 1e-9:
                    raise ValueError("trace must start at spawn at t=0")
                a = pos
            else:
                if t <= previous.timestamp_s:
                    raise ValueError("timestamps must strictly increase")
                a = previous.position
            clearance = self.clearance(a, pos)
            collisions += int(clearance <= 0)
            min_clearance = min(min_clearance, clearance)
            left_bounds |= not self.body_in_bounds(pos)
            if sample.scan and clearance > 0 and self.body_in_bounds(pos):
                # This ideal scanner accepts cell-centered samples only, to
                # avoid quantization credit through an obstacle.
                cell = self.index(pos)
                if math.dist(pos, self.center(cell)) <= 1e-8:
                    observed.update(self.axial_scan(cell))
            previous = sample
            checked.append(sample)
        end = checked[-1]
        dwell = [sample for sample in checked if end.timestamp_s - sample.timestamp_s <= 1.0]
        stationary = (len(dwell) >= 2 and end.timestamp_s - dwell[0].timestamp_s >= 1 - 1e-9
                      and all(math.dist(sample.position, self.spawn) <= 0.10 for sample in dwell))
        landed = bool(land and stationary and not collisions and not left_bounds)
        return GroundTruthEpisode(tuple(checked), tuple(sorted(observed.items())), collisions,
                                  left_bounds, min_clearance, landed, seed, self.digest)

    def witness(self, seed):
        """Evaluator-only successful fixture, NOT an autonomous exploration baseline."""
        start = self.index(self.spawn)
        seen = {start}
        route = [start]
        stack = [(start, iter(AXES))]
        while stack:
            here, directions = stack[-1]
            direction = next(directions, None)
            if direction is None:
                stack.pop()
                if stack:
                    route.append(stack[-1][0])
                continue
            nxt = tuple(here[i] + direction[i] for i in range(3))
            if nxt in self.reachable and nxt not in seen:
                seen.add(nxt)
                route.append(nxt)
                stack.append((nxt, iter(AXES)))
        samples = [GroundTruthSample(0.0, self.spawn, True)]
        t = 0.0
        for cell in route:
            pos = self.center(cell)
            if math.dist(samples[-1].position, pos) < 1e-9:
                continue
            t += math.dist(samples[-1].position, pos) / 0.5
            samples.append(GroundTruthSample(t, pos, True))
        if math.dist(samples[-1].position, self.spawn) > 1e-9:
            t += math.dist(samples[-1].position, self.spawn) / 0.5
            samples.append(GroundTruthSample(t, self.spawn, True))
        samples.append(GroundTruthSample(t + 1, self.spawn, True))
        return self.record(samples, seed=seed, land=True)
