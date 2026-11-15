"""Strict recorded sensor data, with no runtime access to ground truth.

NumPy is an optional Phase 2 dependency; importing the Phase 0/1 stack does not
import this module. Depth is optical camera Z, in metres, with NaN for holes.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .contracts import ContractError, load_json


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.width, self.height)):
            raise ValueError("camera dimensions must be positive integers")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("camera intrinsics must be finite numbers")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")
        if not (0 <= self.cx < self.width and 0 <= self.cy < self.height):
            raise ValueError("principal point must be inside the image")

    @classmethod
    def from_profile(cls, profile):
        return from_profile(profile)


def from_profile(profile):
    """Derive a centred pinhole camera from nominal profile dimensions/FOV."""
    params = profile["parameters"]
    def nominal(name):
        return params[name]["nominal"]
    width, height = nominal("depth_width"), nominal("depth_height")
    hfov, vfov = nominal("depth_horizontal_fov_rad"), nominal("depth_vertical_fov_rad")
    if not (0 < hfov < math.pi and 0 < vfov < math.pi):
        raise ValueError("camera field of view must be between zero and pi")
    return CameraIntrinsics(width, height, width / (2 * math.tan(hfov / 2)),
                            height / (2 * math.tan(vfov / 2)),
                            (width - 1) / 2, (height - 1) / 2)


def validate_metadata(metadata):
    required = {"schema_version", "recording_id", "source", "seed", "profile_id", "camera", "frames_file"}
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ContractError("recording metadata: missing or unknown fields")
    if metadata["schema_version"] != "ares.sensor_recording.v1":
        raise ContractError("recording metadata: unsupported schema_version")
    if metadata["frames_file"] != "frames.npz":
        raise ContractError("recording metadata: frames_file must be frames.npz")
    if metadata["source"] not in ("synthetic", "real"):
        raise ContractError("recording metadata: source must be synthetic or real")
    if type(metadata["seed"]) is not int:
        raise ContractError("recording metadata: seed must be an integer")
    for name in ("recording_id", "profile_id"):
        if not isinstance(metadata[name], str) or not metadata[name].strip():
            raise ContractError(f"recording metadata: {name} must be a nonempty string")
    camera = metadata["camera"]
    if not isinstance(camera, dict) or set(camera) != set(CameraIntrinsics.__dataclass_fields__):
        raise ContractError("recording metadata: invalid camera fields")
    try:
        return CameraIntrinsics(**camera)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"recording metadata: {exc}") from exc


def validate_arrays(arrays, camera):
    required = {"timestamps_s", "depth_m", "imu_timestamps_s", "imu_accel_mps2", "imu_gyro_rps"}
    flow = {"flow_timestamps_s", "flow_velocity_body_mps", "flow_range_m"}
    if not isinstance(arrays, dict) or not required <= arrays.keys() or arrays.keys() - required - flow:
        raise ContractError("recording arrays: missing or unknown arrays")
    if arrays.keys() & flow and not flow <= arrays.keys():
        raise ContractError("recording arrays: optical flow requires all three arrays")
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "fi":
            raise ContractError(f"{name}: expected numeric ndarray without objects")
        if name != "depth_m" and not np.isfinite(value).all():
            raise ContractError(f"{name}: all values must be finite")
    def times(name, nonempty=False):
        value = arrays[name]
        if value.ndim != 1 or (nonempty and len(value) == 0) or np.any(np.diff(value) <= 0):
            raise ContractError(f"{name}: expected strictly increasing one-dimensional timestamps")
        return len(value)
    frames, imu = times("timestamps_s", True), times("imu_timestamps_s")
    depth = arrays["depth_m"]
    if depth.dtype != np.dtype("float32") or depth.shape != (frames, camera.height, camera.width):
        raise ContractError("depth_m: expected float32 array shaped (F, height, width)")
    # Captures may be disk-backed, minutes-long arrays. Limit validation
    # temporaries to a few frames instead of allocating a whole-recording mask.
    for first in range(0, frames, 8):
        block = depth[first:first + 8]
        if np.isinf(block).any() or np.any(np.isfinite(block) & (block <= 0)):
            raise ContractError("depth_m: use positive optical Z or NaN for invalid depth")
    for name in ("imu_accel_mps2", "imu_gyro_rps"):
        if arrays[name].shape != (imu, 3):
            raise ContractError(f"{name}: expected shape (M, 3)")
    if flow <= arrays.keys():
        count = times("flow_timestamps_s")
        if arrays["flow_velocity_body_mps"].shape != (count, 2) or arrays["flow_range_m"].shape != (count,):
            raise ContractError("optical flow: expected velocity (K,2) and range (K,)")
        if np.any(arrays["flow_range_m"] <= 0):
            raise ContractError("flow_range_m: ranges must be positive")


def read_recording(path):
    """Load only metadata.json and frames.npz; ground_truth.npz is never opened."""
    directory = Path(path).resolve()
    metadata = load_json(directory / "metadata.json")
    camera = validate_metadata(metadata)
    frames_path = directory / "frames.npz"
    if frames_path.resolve().parent != directory:
        raise ContractError("recording arrays: frames.npz may not escape recording directory")
    try:
        with np.load(frames_path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ContractError("recording arrays: duplicate names")
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, EOFError) as exc:
        raise ContractError(f"recording arrays: {exc}") from exc
    validate_arrays(arrays, camera)
    return metadata, arrays


def write_recording(path, metadata, arrays):
    """Write a validated recording directory; never accept ground-truth arrays."""
    camera = validate_metadata(metadata)
    validate_arrays(arrays, camera)
    directory = Path(path).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "frames.npz"):
        if (directory / name).exists() or (directory / name).is_symlink():
            raise ContractError(f"recording: refusing existing output {name}")
    np.savez_compressed(directory / "frames.npz", **arrays)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
