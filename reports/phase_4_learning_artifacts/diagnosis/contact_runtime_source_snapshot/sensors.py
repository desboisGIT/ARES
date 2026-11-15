"""Explicit, seeded sensor approximations; importing this module does not require NumPy."""
from dataclasses import dataclass
import hashlib
import math

try:
    import numpy as np
except ImportError:  # Keep the dependency-free ARES tools usable.
    np = None


def _numpy():
    if np is None:
        raise RuntimeError("Sensor simulation requires NumPy")


def _rng(seed, name):
    _numpy()
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    digest = hashlib.sha256(f"ares.sensor.v1:{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def _param(profile, name, default=None):
    params = profile.get("parameters", profile)
    if name not in params:
        if profile.get("schema_version", "ares.profile.v1") != "ares.profile.v1" or default is None:
            raise ValueError(f"missing sensor parameter {name}")
        return default
    value = params[name]
    return value["nominal"] if isinstance(value, dict) else value


def _nonnegative(value, name):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return result


def _dt(dt):
    if isinstance(dt, bool) or not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("dt must be finite and positive")


class SensorSuite:
    """IMU specific force, body XY velocity/range, and bounded distance drift."""
    def __init__(self, profile, seed):
        _numpy()
        self.gravity = _nonnegative(_param(profile, "gravity_mps2", 9.80665), "gravity")
        self.accel_std = _nonnegative(_param(profile, "imu_accel_noise_std"), "accel noise")
        self.gyro_std = _nonnegative(_param(profile, "imu_gyro_noise_std"), "gyro noise")
        self.flow_std = _nonnegative(_param(profile, "optical_flow_noise_std_mps", .02), "flow noise")
        self.range_std = _nonnegative(_param(profile, "range_noise_std_m", .01), "range noise")
        self.drift_rate = _nonnegative(_param(profile, "odometry_drift_per_m"), "drift rate")
        self.min_range = _nonnegative(_param(profile, "depth_min_range_m", 0.), "min range")
        self.max_range = _nonnegative(_param(profile, "depth_max_range_m"), "max range")
        self.imu_rate_hz = _nonnegative(_param(profile, "imu_rate_hz", 200.), "IMU rate")
        if self.max_range <= self.min_range or self.imu_rate_hz == 0:
            raise ValueError("max range and IMU rate must be positive")
        self._imu_rng = _rng(seed, "imu")
        self._flow_rng = _rng(seed, "flow")
        self._drift_rng = _rng(seed, "odometry")
        bias = self._drift_rng.normal(size=3)
        self._bias = bias / np.linalg.norm(bias)
        self._error = np.zeros(3)
        self._previous_position = None

    def imu(self, acceleration_world, angular_velocity_body, rotation_body_to_world, dt):
        _dt(dt)
        acceleration = _vector(acceleration_world, "acceleration_world")
        omega = _vector(angular_velocity_body, "angular_velocity_body")
        rotation = np.asarray(rotation_body_to_world, dtype=float)
        if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)) or
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or
                not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
            raise ValueError("rotation_body_to_world must be a proper rotation")
        force = rotation.T @ (acceleration - np.array([0., 0., -self.gravity]))
        return {"accel_mps2": force + self._imu_rng.normal(0, self.accel_std, 3),
                "gyro_rps": omega + self._imu_rng.normal(0, self.gyro_std, 3)}

    def optical_flow(self, velocity_body, height_m, dt):
        _dt(dt)
        velocity = _vector(velocity_body, "velocity_body")
        height = float(height_m)
        valid = math.isfinite(height) and self.min_range < height <= self.max_range
        if not valid:
            return {"velocity_body_mps2d": np.full(2, np.nan), "range_m": float("nan"), "valid": False}
        measured_range = height + self._flow_rng.normal(0, self.range_std)
        if not self.min_range < measured_range <= self.max_range:
            return {"velocity_body_mps2d": np.full(2, np.nan), "range_m": float("nan"), "valid": False}
        return {"velocity_body_mps2d": velocity[:2] + self._flow_rng.normal(0, self.flow_std, 2),
                "range_m": float(measured_range), "valid": True}

    def drift(self, pose_position, dt):
        _dt(dt)
        position = _vector(pose_position, "pose_position")
        if self._previous_position is not None:
            distance = np.linalg.norm(position - self._previous_position)
            direction = self._drift_rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            self._error += self.drift_rate * distance * (.5 * self._bias + .5 * direction)
        self._previous_position = position.copy()
        return position + self._error


@dataclass(frozen=True)
class DepthNoiseModel:
    base_std_m: float = .01
    quadratic_std_per_m2: float = .001
    max_range_m: float = 4.
    min_range_m: float = 0.
    edge_width_px: int = 1
    edge_threshold_m: float = .2
    max_range_dropout_probability: float = .1
    material_hole_probability: tuple = (0., .15, .1, .7, .9)

    def __post_init__(self):
        for name in ("base_std_m", "quadratic_std_per_m2", "max_range_m", "min_range_m", "edge_threshold_m"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must be positive")
        if type(self.edge_width_px) is not int or not 0 <= self.edge_width_px <= 32:
            raise ValueError("edge_width_px must be an integer in [0, 32]")
        probabilities = tuple(self.material_hole_probability)
        if len(probabilities) != 5:
            raise ValueError("material_hole_probability requires five probabilities")
        for value in probabilities + (self.max_range_dropout_probability,):
            if not math.isfinite(float(value)) or not 0 <= value <= 1:
                raise ValueError("probabilities must be in [0, 1]")
        object.__setattr__(self, "material_hole_probability", probabilities)

    @classmethod
    def from_profile(cls, profile):
        return cls(base_std_m=_param(profile, "depth_noise_std_m"),
                   quadratic_std_per_m2=_param(profile, "depth_noise_quadratic_per_m2", .001),
                   max_range_m=_param(profile, "depth_max_range_m"),
                   min_range_m=_param(profile, "depth_min_range_m", 0.),
                   edge_width_px=_param(profile, "depth_edge_width_px", 1),
                   edge_threshold_m=_param(profile, "depth_edge_threshold_m", .2),
                   max_range_dropout_probability=_param(profile, "max_range_dropout_probability", .1),
                   material_hole_probability=_param(profile, "material_hole_probability", (0., .15, .1, .7, .9)))


def apply_depth_noise(depth_m, materials, model, seed, frame_index):
    """Return optical-axis Z metres (float32); all missing returns are NaN."""
    _numpy()
    if type(frame_index) is not int or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    depth = np.asarray(depth_m, dtype=float)
    material = np.asarray(materials)
    if depth.ndim not in (2, 3) or material.shape != depth.shape:
        raise ValueError("depth/materials must have matching (H,W) or (B,H,W) shapes")
    if not np.all(np.isin(material, [-1, 0, 1, 2, 3, 4])):
        raise ValueError("material IDs must be -1 or 0..4")
    valid = np.isfinite(depth) & (depth > model.min_range_m) & (depth <= model.max_range_m) & (material >= 0)
    # Detect discontinuities on image axes only, including hit/no-hit boundaries.
    edge = np.zeros(depth.shape, dtype=bool)
    for axis in (-2, -1):
        left = [slice(None)] * depth.ndim
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        pair = (valid[left] != valid[right]) | (valid[left] & valid[right] &
                (np.abs(depth[left] - depth[right]) > model.edge_threshold_m))
        edge[left] |= pair
        edge[right] |= pair
    if model.edge_width_px:
        expanded = edge.copy()
        for dy in range(-min(model.edge_width_px - 1, depth.shape[-2] - 1),
                        min(model.edge_width_px - 1, depth.shape[-2] - 1) + 1):
            for dx in range(-min(model.edge_width_px - 1, depth.shape[-1] - 1),
                            min(model.edge_width_px - 1, depth.shape[-1] - 1) + 1):
                src_y = slice(max(0, -dy), depth.shape[-2] - max(0, dy))
                dst_y = slice(max(0, dy), depth.shape[-2] - max(0, -dy))
                src_x = slice(max(0, -dx), depth.shape[-1] - max(0, dx))
                dst_x = slice(max(0, dx), depth.shape[-1] - max(0, -dx))
                expanded[..., dst_y, dst_x] |= edge[..., src_y, src_x]
        valid &= ~expanded
    z = np.where(valid, depth, 0.)
    sigma = np.sqrt(model.base_std_m ** 2 + (model.quadratic_std_per_m2 * z ** 2) ** 2)
    noisy = z + _rng(seed, f"depth:frame:{frame_index}:gaussian").normal(size=depth.shape) * sigma
    probability = np.asarray(model.material_hole_probability)[np.maximum(material.astype(int), 0)]
    holes = _rng(seed, f"depth:frame:{frame_index}:material").random(depth.shape) < probability
    range_holes = _rng(seed, f"depth:frame:{frame_index}:range").random(depth.shape) < (
        model.max_range_dropout_probability * (z / model.max_range_m) ** 2)
    valid &= ~holes & ~range_holes & (noisy > model.min_range_m) & (noisy <= model.max_range_m)
    return np.where(valid, noisy, np.nan).astype(np.float32)
