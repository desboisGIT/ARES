"""Versioned Phase 0 log export with an explicit evaluator ground-truth block.

Phase 0 has no actual flight sensor or command fidelity: raw frames, IMU,
commands, policy outputs, shield interventions, and map snapshots are null.
Pose estimates are idealized harness positions with an identity orientation;
these logs do not demonstrate real odometry or hardware performance.
"""

import math

from .contracts import ContractError, _list, _number, _object, _string, _vector

SCHEMA_VERSION = "ares.flight_log.v1"
NULL_FIELDS = (
    "depth_frame", "camera_frame", "imu", "command", "policy_output",
    "shield_intervention", "map_snapshot",
)


def _samples(samples, *, ground_truth=False):
    _list(samples, "samples")
    if not samples:
        raise ContractError("samples: expected nonempty trace starting at t=0")
    previous = -1.0
    for index, sample in enumerate(samples):
        path = f"samples[{index}]"
        keys = {"timestamp_s", "position"} if ground_truth else {
            "timestamp_s", "pose_estimate", *NULL_FIELDS,
        }
        _object(sample, keys, path)
        timestamp = _number(sample["timestamp_s"], f"{path}.timestamp_s")
        if (index == 0 and timestamp != 0) or timestamp < 0 or timestamp <= previous:
            raise ContractError(f"{path}.timestamp_s: expected nonnegative strictly increasing time")
        previous = timestamp
        if ground_truth:
            _vector(sample["position"], 3, f"{path}.position")
            continue
        pose = sample["pose_estimate"]
        _object(pose, {"position", "quaternion_xyzw"}, f"{path}.pose_estimate")
        _vector(pose["position"], 3, f"{path}.pose_estimate.position")
        quaternion = _vector(pose["quaternion_xyzw"], 4, f"{path}.pose_estimate.quaternion_xyzw")
        if not math.isclose(sum(v * v for v in quaternion), 1, abs_tol=1e-9):
            raise ContractError(f"{path}.pose_estimate.quaternion_xyzw: expected unit quaternion")
        for field in NULL_FIELDS:
            if sample[field] is not None:
                raise ContractError(f"{path}.{field}: Phase 0 requires null")


def validate_flight_log(log, *, allow_ground_truth=False):
    """Validate strict v1 JSON-shaped data; reject evaluator truth by default."""
    if not isinstance(allow_ground_truth, bool):
        raise ContractError("allow_ground_truth: expected bool")
    keys = {"schema_version", "scene_id", "profile_id", "seed", "samples"}
    if isinstance(log, dict) and "ground_truth" in log:
        if not allow_ground_truth:
            raise ContractError("ground_truth: evaluator-only data is not allowed")
        keys.add("ground_truth")
    _object(log, keys, "flight_log")
    if log["schema_version"] != SCHEMA_VERSION:
        raise ContractError("schema_version: unsupported flight log version")
    _string(log["scene_id"], "scene_id")
    _string(log["profile_id"], "profile_id")
    _number(log["seed"], "seed", integer=True)
    _samples(log["samples"])
    if "ground_truth" in log:
        truth = log["ground_truth"]
        _object(truth, {"samples", "observed_voxels"}, "ground_truth")
        _samples(truth["samples"], ground_truth=True)
        _list(truth["observed_voxels"], "ground_truth.observed_voxels")
        seen = set()
        for index, voxel in enumerate(truth["observed_voxels"]):
            path = f"ground_truth.observed_voxels[{index}]"
            values = _vector(voxel, 4, path, integer=True)
            if any(v < 0 for v in values[:3]) or values[3] not in (0, 1, 2):
                raise ContractError(f"{path}: expected nonnegative indices and occupancy 0, 1, or 2")
            if values[:3] in seen:
                raise ContractError(f"{path}: duplicate voxel")
            seen.add(values[:3])
    return log


def make_flight_log(episode, *, scene_id, profile_id, include_ground_truth=False):
    """Copy harness samples into a log, optionally adding isolated evaluator data."""
    if not isinstance(include_ground_truth, bool):
        raise ContractError("include_ground_truth: expected bool")
    result = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "profile_id": profile_id,
        "seed": episode.seed,
        "samples": [
            {"timestamp_s": sample.timestamp_s,
             "pose_estimate": {"position": list(sample.position),
                               "quaternion_xyzw": [0, 0, 0, 1]},
             **dict.fromkeys(NULL_FIELDS)}
            for sample in episode.samples
        ],
    }
    if include_ground_truth:
        result["ground_truth"] = {
            "samples": [{"timestamp_s": sample.timestamp_s,
                         "position": list(sample.position)} for sample in episode.samples],
            "observed_voxels": [[*cell, value] for cell, value in episode.observations],
        }
    return validate_flight_log(result, allow_ground_truth=include_ground_truth)
