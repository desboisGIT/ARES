"""CPU runtime replay of sensor data. This module never loads evaluator truth."""

from dataclasses import asdict
import platform
import time

import numpy as np

from .depth import OPTICAL_TO_BODY
from .contracts import validate_profile
from .mapping import OccupancyMapper
from .odometry import DepthInertialOdometry
from .recording import CameraIntrinsics, read_recording


def timing_summary(durations, target_hz):
    if not durations:
        return {"samples": 0, "target_hz": target_hz, "meets_target": False}
    values = np.asarray(durations, dtype=float)
    return {"samples": len(values), "median_s": float(np.median(values)),
            "p90_s": float(np.percentile(values, 90)), "p99_s": float(np.percentile(values, 99)),
            "hz": float(len(values) / values.sum()), "target_hz": target_hz,
            "meets_target": bool(np.percentile(values, 99) <= 1 / target_hz)}


def replay_recording(path, *, max_frames=None, mapper_max_rays=64, profile=None):
    """Same estimator/mapper for synthetic and real recordings, no pose oracle.

    The initial origin is the first camera position, held level facing world +X.
    World yaw is a local convention, not a globally measured heading. A real rig
    must start upright or supply a future gravity-based initializer.
    """
    metadata, arrays = read_recording(path)
    camera = CameraIntrinsics(**metadata["camera"])
    if max_frames is not None and (type(max_frames) is not int or max_frames < 2):
        raise ValueError("max_frames must be an integer >=2")
    limit = len(arrays["timestamps_s"]) if max_frames is None else min(max_frames, len(arrays["timestamps_s"]))
    if limit < 2:
        raise ValueError("replay requires at least two depth frames")
    pose = np.eye(4)
    pose[:3, :3] = OPTICAL_TO_BODY
    options = {}
    if profile is not None:
        validate_profile(profile)
        if profile["profile_id"] != metadata["profile_id"]:
            raise ValueError("recording/profile identifiers do not match")
        options["gyro_noise_std_rps"] = max(1e-9, profile["parameters"]["imu_gyro_noise_std"]["nominal"])
    odometry = DepthInertialOdometry(camera, initial_pose=pose, **options)
    mapper = OccupancyMapper(max_rays=mapper_max_rays)
    records, odometry_times, map_times, policy_times, combined_times = [], [], [], [], []
    accepted = 0
    started = time.perf_counter()
    for index in range(limit):
        frame_start = time.perf_counter()
        timestamp = float(arrays["timestamps_s"][index])
        # No extrapolation beyond actual IMU coverage. A missing prior does not
        # manufacture IMU data; depth registration remains available.
        imu_time = arrays["imu_timestamps_s"]
        covered = bool(len(imu_time) and imu_time[0] <= timestamp <= imu_time[-1])
        accel = gyro = None
        if covered:
            accel = np.array([np.interp(timestamp, imu_time, arrays["imu_accel_mps2"][:, axis]) for axis in range(3)])
            gyro = np.array([np.interp(timestamp, imu_time, arrays["imu_gyro_rps"][:, axis]) for axis in range(3)])
        before = time.perf_counter()
        estimate = odometry.update(arrays["depth_m"][index], timestamp, accel, gyro)
        odometry_times.append(time.perf_counter() - before)
        if estimate["can_integrate"]:
            before = time.perf_counter()
            mapper.integrate(arrays["depth_m"][index], camera, estimate["pose"])
            map_times.append(time.perf_counter() - before)
            accepted += 1
        before = time.perf_counter()
        # Timing-only dummy, with no flight-controller interface or actuator.
        dummy = {"thrust": 0., "body_rates": [0., 0., 0.], "enabled": False}
        policy_times.append(time.perf_counter() - before)
        combined_times.append(time.perf_counter() - frame_start)
        records.append({"timestamp_s": timestamp, "pose": estimate["pose"].tolist(),
                        "status": estimate["status"], "reason": estimate["reason"],
                        "can_integrate": estimate["can_integrate"], "rmse_m": estimate["rmse_m"],
                        "correspondences": estimate["correspondences"], "rank": estimate["rank"],
                        "condition_number": estimate["condition_number"], "imu_covered": covered,
                        "normal_span_ratio": estimate.get("normal_span_ratio"),
                        "normal_span_rank": estimate.get("normal_span_rank"),
                        "dummy_policy_output": dummy})
    elapsed = time.perf_counter() - started
    return {"schema_version": "ares.sensor_replay.v1", "recording_id": metadata["recording_id"],
            "source": metadata["source"], "profile_id": metadata["profile_id"], "seed": metadata["seed"],
            "camera": asdict(camera), "frames": limit, "accepted_frames": accepted,
            "gyro_noise_source": "supplied_profile" if profile is not None else "documented_estimator_default",
            "tracking_fraction": accepted / limit, "elapsed_wall_s": elapsed,
            "replayed_sensor_duration_s": records[-1]["timestamp_s"] - records[0]["timestamp_s"],
            "terminal_displacement_m": float(np.linalg.norm(np.array(records[-1]["pose"])[:3, 3])),
            "loop_error_m": None, "loop_error_status": "requires_independent_closed_route_reference",
            "timings": {"odometry": timing_summary(odometry_times, 100),
                        "mapper": timing_summary(map_times, 15),
                        "dummy_policy": timing_summary(policy_times, 60),
                        "combined": timing_summary(combined_times, 30)},
            "measurement_host": {"kind": "current_host_not_verified_onboard", "platform": platform.platform(),
                                 "machine": platform.machine(), "processor": platform.processor()},
            "mapper_max_rays": mapper_max_rays, "map": mapper.export(), "map_counts": mapper.counts(),
            "trajectory": records,
            "limitations": ["Local depth ICP and gyro prior; not full stereo visual-inertial SLAM.",
                            "No accelerometer translation/bias fusion, optical-flow fusion, or loop closure.",
                            "Component timings do not prove real-time scheduling or onboard compute rates.",
                            "Initial rig orientation is assumed level; no absolute pose or gravity initializer."]}
