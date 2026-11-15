"""Phase 2 synthetic checks and measured preparation; hardware gates stay explicit."""

from dataclasses import asdict, replace
import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path
import platform
import tempfile
import time

import numpy as np

from .calibration import fit_depth_noise
from .contracts import load_json, validate_scene, validate_profile
from .depth import GroundTruthDepthRenderer, OPTICAL_TO_BODY, DEPTH_RENDERER_VERSION
from .recording import write_recording, validate_metadata
from .replay import replay_recording
from .sensors import SensorSuite, apply_depth_noise


def default_profile():
    return load_json(files("ares") / "data/configs/drone_sim_v2.json")


def _yaw(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def write_synthetic_recording(path, scene, profile, *, backend="cpu", frames=24, seed=1741):
    if type(frames) is not int or frames < 4:
        raise ValueError("synthetic fixture requires at least four frames")
    if profile.get("schema_version") != "ares.profile.v2":
        raise ValueError("synthetic sensor fixtures require the explicit ares.profile.v2 sensor parameters")
    renderer = GroundTruthDepthRenderer(scene, profile, backend=backend, seed=seed)
    fps = profile["parameters"]["depth_fps"]["nominal"]
    stride = max(1, math.ceil(2 * fps / (frames - 1)))
    indices = np.arange(frames) * stride
    times = indices / fps
    duration = float(times[-1])
    # Look obliquely so the positive witness sees front/side/floor surfaces.
    # A straight-on view with only two normal families is deliberately tested
    # as unobservable by the odometry tests, not hidden by depth noise.
    omega, radius, yaw_amplitude, initial_yaw = 2 * math.pi / duration, .20, .04, .35
    spawn = np.asarray(scene["spawn"], dtype=float)

    def state(t):
        angle = omega * t
        position = spawn + [radius*math.sin(angle), radius*(1-math.cos(angle)), 0.]
        rotation = _yaw(initial_yaw + yaw_amplitude * math.sin(angle))
        velocity = np.array([radius*omega*math.cos(angle), radius*omega*math.sin(angle), 0.])
        acceleration = np.array([-radius*omega**2*math.sin(angle), radius*omega**2*math.cos(angle), 0.])
        gyro = np.array([0., 0., yaw_amplitude*omega*math.cos(angle)])
        return position, rotation, velocity, acceleration, gyro

    depth, ideal, positions, rotations = [], [], [], []
    for index, t in zip(indices, times):
        position, rotation, _, _, _ = state(t)
        camera_rotation = rotation @ np.asarray(OPTICAL_TO_BODY)
        rendered = renderer.render(position, camera_rotation, frame_index=int(index), noise=True)
        depth.append(renderer.to_host(rendered.depth_m)[0])
        ideal.append(renderer.to_host(rendered.ground_truth["ideal_depth_m"])[0])
        positions.append(position)
        rotations.append(camera_rotation)
    suite = SensorSuite(profile, seed)
    imu_hz = profile["parameters"]["imu_rate_hz"]["nominal"]
    imu_times = np.arange(math.floor(duration * imu_hz) + 1) / imu_hz
    imu_accel, imu_gyro = [], []
    for t in imu_times:
        _, rotation, _, acceleration, gyro = state(t)
        sample = suite.imu(acceleration, gyro, rotation, 1 / imu_hz)
        imu_accel.append(sample["accel_mps2"])
        imu_gyro.append(sample["gyro_rps"])
    flow_times, flow_velocity, flow_range = [], [], []
    flow_hz = profile["parameters"]["optical_flow_rate_hz"]["nominal"]
    for t in np.arange(math.floor(duration * flow_hz) + 1) / flow_hz:
        position, rotation, velocity, _, _ = state(t)
        # Synthetic floor is the top face of the floor voxel at origin.z+res.
        height = position[2] - (scene["grid"]["origin"][2] + scene["grid"]["resolution"])
        sample = suite.optical_flow(rotation.T @ velocity, height, 1 / flow_hz)
        if sample["valid"]:
            flow_times.append(t)
            flow_velocity.append(sample["velocity_body_mps2d"])
            flow_range.append(sample["range_m"])
    metadata = {"schema_version": "ares.sensor_recording.v1", "recording_id": f"synthetic-depth-loop-{seed}",
                "source": "synthetic", "seed": seed, "profile_id": profile["profile_id"],
                "camera": asdict(renderer.camera), "frames_file": "frames.npz"}
    arrays = {"timestamps_s": times, "depth_m": np.asarray(depth, dtype=np.float32),
              "imu_timestamps_s": imu_times, "imu_accel_mps2": np.asarray(imu_accel),
              "imu_gyro_rps": np.asarray(imu_gyro), "flow_timestamps_s": np.asarray(flow_times),
              "flow_velocity_body_mps": np.asarray(flow_velocity).reshape(-1, 2), "flow_range_m": np.asarray(flow_range)}
    write_recording(path, metadata, arrays)
    # Runtime readers never open this separate evaluator-only file.
    np.savez_compressed(Path(path) / "ground_truth.npz", timestamps_s=times,
                        positions_world=np.asarray(positions), camera_to_world_rotation=np.asarray(rotations),
                        ideal_depth_m=np.asarray(ideal))
    scene_digest = hashlib.sha256(json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    manifest = {"schema_version": "ares.synthetic_sensor_fixture.v1", "source": "synthetic",
                "seed": seed, "scene_id": scene["scene_id"], "profile": profile, "frames": frames,
                "scene_version": scene["schema_version"], "scene_sha256": scene_digest,
                "renderer_version": DEPTH_RENDERER_VERSION,
                "nominal_depth_fps": fps, "recorded_frame_stride": stride,
                "circle_radius_m": radius, "initial_body_yaw_rad": initial_yaw,
                "note": "Sparse samples of a short analytic closed loop; not a real handheld log or flight.",
                "noise_calibration_status": "unfitted_assumptions", "backend": backend}
    (Path(path) / "simulation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def evaluate_synthetic_replay(replay, ground_truth_path):
    """Evaluator-only alignment: runtime estimated poses are already finalized."""
    with np.load(ground_truth_path, allow_pickle=False) as truth:
        initial_alignment = np.array(OPTICAL_TO_BODY) @ truth["camera_to_world_rotation"][0].T
        expected = (truth["positions_world"] - truth["positions_world"][0]) @ initial_alignment.T
        expected_rotation = initial_alignment @ truth["camera_to_world_rotation"]
    estimated = np.array([np.asarray(frame["pose"])[:3, 3] for frame in replay["trajectory"]])
    errors = np.linalg.norm(estimated - expected[:len(estimated)], axis=1)
    estimated_rotation = np.array([np.asarray(frame["pose"])[:3, :3] for frame in replay["trajectory"]])
    relative = estimated_rotation.transpose(0, 2, 1) @ expected_rotation[:len(estimated)]
    angles = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1))
    return {"schema_version": "ares.synthetic_replay_evaluation.v1", "source": "synthetic",
            "pose_rmse_m": float(np.sqrt(np.mean(errors ** 2))), "maximum_pose_error_m": float(errors.max()),
            "closed_loop_error_m": float(np.linalg.norm(estimated[-1] - expected[len(estimated)-1])),
            "rotation_rmse_rad": float(np.sqrt(np.mean(angles**2))), "maximum_rotation_error_rad": float(angles.max()),
            "alignment": "single initial orientation/origin alignment used only after runtime replay",
            "ground_truth_used_only_by": "evaluate_synthetic_replay", "real_place_loop_error_m": None}


def benchmark_depth(renderer, scene, *, batch_size=4096, repeats=7):
    if type(batch_size) is not int or batch_size < 1 or type(repeats) is not int or repeats < 2:
        raise ValueError("batch_size must be positive and repeats >=2")
    rng = np.random.default_rng(1741)
    positions = np.tile(np.array(scene["spawn"], dtype=float), (batch_size, 1))
    positions += rng.uniform(-.04, .04, positions.shape)
    angles = rng.uniform(-math.pi, math.pi, batch_size)
    rotations = np.array([_yaw(angle) @ np.array(OPTICAL_TO_BODY) for angle in angles])
    measurements = {}
    for noisy in (False, True):
        for i in range(2):
            renderer.render(positions, rotations, frame_index=i, noise=noisy)
            renderer.synchronize()
        durations = []
        for i in range(repeats):
            start = time.perf_counter()
            result = renderer.render(positions, rotations, frame_index=i+2, noise=noisy)
            renderer.synchronize()
            durations.append(time.perf_counter() - start)
        seconds = float(np.median(durations))
        measurements["sensor_depth" if noisy else "ideal_raycast"] = {
            "median_batch_s": seconds, "p90_batch_s": float(np.percentile(durations, 90)),
            "camera_frames_per_second": batch_size / seconds,
            "rays_per_second": batch_size * renderer.camera.width * renderer.camera.height / seconds,
            "per_environment_hz": 1 / seconds, "repeats": repeats,
            "valid_depth_fraction": float(np.isfinite(renderer.to_host(result.depth_m)).mean())}
    device = None
    if renderer.backend == "cuda":
        props = renderer.cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        device = {"name": name.decode() if isinstance(name, bytes) else name,
                  "memory_bytes": int(props["totalGlobalMem"]), "cupy": renderer.cp.__version__,
                  "cuda_runtime_version": renderer.cp.cuda.runtime.runtimeGetVersion(),
                  "cuda_driver_version": renderer.cp.cuda.runtime.driverGetVersion()}
    return {"schema_version": "ares.depth_benchmark.v1", "backend": renderer.backend,
            "batch_size": batch_size, "width": renderer.camera.width, "height": renderer.camera.height,
            "nominal_camera_fps": renderer.clock.fps, "seed": 1741, "device": device,
            "python": platform.python_version(), "numpy": np.__version__, "measurements": measurements,
            "scope": "One voxel scene, varying camera yaw/near-spawn positions; pose upload, validation, allocations, raycast, optional on-device noise, completion synchronization; excludes scene upload/JIT warmup and output download."}


def _build_phase2(scene, profile, output_dir, *, backend, batch_size, frames, seed):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    recording = directory / "synthetic_recording"
    manifest = write_synthetic_recording(recording, scene, profile, backend=backend, frames=frames, seed=seed)
    replay = replay_recording(recording, profile=profile)
    evaluation = evaluate_synthetic_replay(replay, recording / "ground_truth.npz")
    renderer = GroundTruthDepthRenderer(scene, profile, backend=backend, seed=seed)
    benchmark = benchmark_depth(renderer, scene, batch_size=batch_size)
    # Validate fitter using deliberately isolated assumptions; no edges or range
    # censoring confounded with material dropout. This remains synthetic evidence.
    model = replace(renderer.model, edge_width_px=0, max_range_dropout_probability=0.)
    reference = np.tile(np.linspace(.5, min(3.5, model.max_range_m-.2), 1000), (100, 1))
    material = np.random.default_rng(seed).integers(0, 5, size=reference.shape)
    measured = apply_depth_noise(reference, material, model, seed, 1000)
    fit = fit_depth_noise(measured, reference, material, source="synthetic")
    checks = {"synthetic_recording_roundtrip": replay["frames"] == frames,
              "synthetic_replay_has_tracking": replay["tracking_fraction"] >= .8,
              "synthetic_pose_error_under_10cm": evaluation["maximum_pose_error_m"] < .10,
              "synthetic_map_has_free_and_occupied": replay["map_counts"]["free_cells"] > 0 and replay["map_counts"]["occupied_cells"] > 0,
              "synthetic_noise_fit_recovers_base": abs(fit["base_std_m"]-model.base_std_m) < max(.002, model.base_std_m*.25),
              "synthetic_noise_fit_recovers_distance_term": abs(fit["quadratic_std_per_m2"]-model.quadratic_std_per_m2) < max(.0005, model.quadratic_std_per_m2*.35),
              "depth_simulation_measured": benchmark["measurements"]["sensor_depth"]["camera_frames_per_second"] > 0}
    hardware = {"real_handheld_places": {"passed": False, "status": "needs_rig_and_logs_from_three_real_places"},
                "real_noise_and_imported_room_fit": {"passed": False, "status": "needs_registered_real_depth_and_room_reference"},
                "onboard_rate_and_latency": {"passed": False, "status": "needs_actual_onboard_computer"}}
    gpu_gate = backend == "cuda" and batch_size >= 1024
    for name, value in (("replay", replay), ("synthetic_evaluation", evaluation), ("depth_benchmark", benchmark),
                        ("synthetic_noise_fit", fit)):
        (directory / (name + ".json")).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return {"schema_version": "ares.phase2.v1", "seed": seed, "profile_id": profile["profile_id"],
            "profile_version": profile["schema_version"], "profile": profile, "scene_id": scene["scene_id"],
            "scene_snapshot": json.loads(json.dumps(scene, allow_nan=False)),
            "scene_sha256": manifest["scene_sha256"], "renderer_version": DEPTH_RENDERER_VERSION,
            "software_checks": checks, "software_preparation_passed": all(checks.values()),
            "gpu_thousands_gate": {"passed": gpu_gate, "backend": backend, "batch_size": batch_size},
            "hardware_gates": hardware, "phase_passed": False,
            "synthetic_evaluation": evaluation, "replay_tracking_fraction": replay["tracking_fraction"],
            "desktop_component_timings": replay["timings"], "depth_benchmark": benchmark,
            "noise_calibration_status": "synthetic_fit_only_real_model_not_calibrated",
            "recording_manifest": manifest,
            "limitations": ["No real hardware or captures were supplied.",
                            "Depth ICP/gyro baseline is not full stereo visual-inertial odometry or loop-closing SLAM.",
                            "Optical flow is simulated but not fused into the estimator.",
                            "No desktop timing establishes the onboard ARM rate gate."]}


_OWNER = "phase2_gate_owner.json"
_OUTPUT_FILES = {"replay.json", "synthetic_evaluation.json", "depth_benchmark.json", "synthetic_noise_fit.json"}
_RECORDING_FILES = {"metadata.json", "frames.npz", "ground_truth.npz", "simulation_manifest.json"}


def _artifact_hashes(directory):
    """Strictly enumerate owned files without traversing any symlink."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("gate output must be a real directory, not a symlink")
    allowed = _OUTPUT_FILES | {"synthetic_recording", _OWNER}
    if {p.name for p in directory.iterdir()} - allowed:
        raise ValueError("gate output contains unrelated files; choose a fresh directory")
    record = directory / "synthetic_recording"
    if record.is_symlink() or not record.is_dir() or {p.name for p in record.iterdir()} != _RECORDING_FILES:
        raise ValueError("gate recording is not an owned regular synthetic recording directory")
    paths = [directory / name for name in _OUTPUT_FILES] + [record / name for name in _RECORDING_FILES]
    hashes = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("gate refuses symlink or missing artifact files")
        with path.open("rb") as stream:
            hashes[str(path.relative_to(directory))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def _owner_payload(directory, *, final_path=None):
    hashes = _artifact_hashes(directory)
    metadata = load_json(directory / "synthetic_recording/metadata.json")
    manifest = load_json(directory / "synthetic_recording/simulation_manifest.json")
    validate_metadata(metadata)
    if not {"profile", "seed", "schema_version", "source"} <= manifest.keys() or not isinstance(manifest["profile"], dict):
        raise ValueError("existing gate simulation manifest is malformed")
    validate_profile(manifest["profile"])
    if (manifest["schema_version"] != "ares.synthetic_sensor_fixture.v1" or manifest["source"] != "synthetic"
            or metadata["source"] != "synthetic" or metadata["seed"] != manifest["seed"]
            or metadata["recording_id"] != f"synthetic-depth-loop-{manifest['seed']}"
            or metadata["profile_id"] != manifest["profile"]["profile_id"]):
        raise ValueError("existing gate recording identity does not match its simulation manifest")
    return {"schema_version": "ares.phase2_gate_owner.v1", "directory": str((final_path or directory).resolve()),
            "artifacts_sha256": hashes}


def run_phase2(scene, profile, output_dir, *, backend="cpu", batch_size=4, frames=12, seed=1741):
    """Build in a fresh staging directory; preserve previous artifacts on failure."""
    if (backend not in ("cpu", "cuda") or type(batch_size) is not int or batch_size < 1
            or type(frames) is not int or frames < 4 or type(seed) is not int):
        raise ValueError("backend cpu/cuda, batch_size positive integer, frames integer >=4, seed integer required")
    validate_scene(scene)
    validate_profile(profile)
    if profile["schema_version"] != "ares.profile.v2":
        raise ValueError("Phase 2 fixtures require ares.profile.v2; v1 remains supported by earlier phases")
    directory = Path(output_dir).absolute()
    if directory.is_symlink():
        raise ValueError("gate output directory must not be a symlink")
    if directory.exists() and any(directory.iterdir()):
        owner = directory / _OWNER
        if owner.is_symlink() or not owner.is_file():
            raise ValueError("gate refuses to replace unowned output; choose a fresh directory")
        expected = load_json(owner)
        if expected != _owner_payload(directory):
            raise ValueError("gate artifacts were modified; preserving previous output")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ares-phase2-", dir=directory.parent) as temporary:
        staging, previous = Path(temporary) / "artifacts", Path(temporary) / "previous"
        result = _build_phase2(scene, profile, staging, backend=backend, batch_size=batch_size, frames=frames, seed=seed)
        payload = _owner_payload(staging, final_path=directory)
        (staging / _OWNER).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        if directory.exists():
            directory.rename(previous)
        try:
            staging.rename(directory)
        except OSError:
            if previous.exists():
                previous.rename(directory)
            raise
    return result
