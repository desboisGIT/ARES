"""Offline evaluator comparison using explicitly registered camera poses."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .depth import GroundTruthDepthRenderer
from .odometry import validate_pose
from .recording import CameraIntrinsics, read_recording


def compare_registered_depth(recording_path, scene, profile, poses_path, *, backend="cpu", pairs_output=None):
    """Render an imported scene at external reference poses, never hidden GT.

    Registration/reference correctness is the operator's responsibility. The
    estimator is not run here, and this is not evidence of real-room agreement
    until actual captures and independent registration have been provided.
    """
    metadata, arrays = read_recording(recording_path)
    with np.load(poses_path, allow_pickle=False) as archive:
        if len(archive.files) != 2 or set(archive.files) != {"timestamps_s", "camera_to_world"}:
            raise ValueError("registered poses require exactly timestamps_s and camera_to_world")
        times, poses = archive["timestamps_s"], archive["camera_to_world"]
    if (times.shape != arrays["timestamps_s"].shape or not np.isfinite(times).all()
            or not np.allclose(times, arrays["timestamps_s"], atol=1e-9, rtol=0)
            or poses.shape != (len(times), 4, 4)):
        raise ValueError("registered poses must match every depth frame and timestamp")
    poses = np.asarray([validate_pose(pose) for pose in poses])
    camera = CameraIntrinsics(**metadata["camera"])
    renderer = GroundTruthDepthRenderer(scene, profile, backend=backend, seed=0, camera=camera)
    reference, material = [], []
    for index, pose in enumerate(poses):
        result = renderer.render(pose[:3, 3], pose[:3, :3], frame_index=index, noise=False)
        reference.append(renderer.to_host(result.depth_m)[0])
        material.append(renderer.to_host(result.ground_truth["material_ids"])[0])
    reference, material = np.asarray(reference), np.asarray(material)
    measured = arrays["depth_m"]
    eligible = np.isfinite(reference) & (reference > 0)
    returned = eligible & np.isfinite(measured) & (measured > 0)
    residual = measured[returned] - reference[returned]
    count = int(returned.sum())
    if pairs_output is not None:
        output = Path(pairs_output)
        if output.exists() or output.is_symlink():
            raise ValueError("refusing to overwrite existing comparison pairs")
        output.parent.mkdir(parents=True, exist_ok=True)
        # A file object prevents NumPy silently appending .npz, and exclusive
        # creation prevents races or following a dangling output symlink.
        with output.open("xb") as stream:
            np.savez_compressed(stream, measured_m=measured, reference_m=reference, material_ids=material)
    return {"schema_version": "ares.registered_depth_comparison.v1", "source": metadata["source"],
            "recording_id": metadata["recording_id"], "scene_id": scene["scene_id"], "frames": len(times),
            "camera": asdict(camera), "reference_pixels": int(eligible.sum()), "compared_pixels": count,
            "missing_measured_fraction": float((eligible & ~returned).sum() / eligible.sum()) if eligible.any() else None,
            "rmse_m": float(np.sqrt(np.mean(residual**2))) if count else None,
            "mae_m": float(np.mean(np.abs(residual))) if count else None,
            "bias_m": float(np.mean(residual)) if count else None,
            "p90_absolute_error_m": float(np.percentile(np.abs(residual), 90)) if count else None,
            "registration_status": "external_poses_supplied_accuracy_not_verified",
            "notes": ["Uses actual recording intrinsics, overriding the centred profile camera model.",
                      "Reported residuals conflate sensor, imported-map, motion and registration errors.",
                      "Independent reference quality and masks must be checked before calibration.",
                      "A source flag declares provenance; it is not authentication of a real capture."]}
