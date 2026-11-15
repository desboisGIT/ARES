"""Integration evidence preserves the boundary between simulation and hardware."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(np is None, "NumPy optional")
class Phase2PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ares.contracts import load_json
        from ares.phase2 import default_profile, run_phase2
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.scene = load_json(ROOT / "ares/data/scenes/three_room.json")
        cls.profile = default_profile()
        cls.artifacts = cls.directory / "artifacts"
        cls.result = run_phase2(cls.scene, cls.profile, cls.artifacts,
                                backend="cpu", batch_size=1, frames=12)
        cls.recording = cls.artifacts / "synthetic_recording"
        with np.load(cls.recording / "ground_truth.npz", allow_pickle=False) as truth:
            cls.times = truth["timestamps_s"].copy()
            cls.poses = np.tile(np.eye(4), (len(cls.times), 1, 1))
            cls.poses[:, :3, 3] = truth["positions_world"]
            cls.poses[:, :3, :3] = truth["camera_to_world_rotation"]
        cls.poses_path = cls.directory / "registered_poses.npz"
        np.savez(cls.poses_path, timestamps_s=cls.times, camera_to_world=cls.poses)

    def test_software_pass_is_not_hardware_or_full_phase_pass(self):
        result = self.result
        self.assertTrue(result["software_preparation_passed"], result["software_checks"])
        self.assertTrue(all(result["software_checks"].values()))
        self.assertFalse(result["phase_passed"])
        self.assertFalse(result["gpu_thousands_gate"]["passed"])
        self.assertTrue(all(not gate["passed"] for gate in result["hardware_gates"].values()))
        self.assertEqual(result["profile"], self.profile)
        self.assertEqual(result["profile_version"], "ares.profile.v2")
        self.assertEqual(result["recording_manifest"]["source"], "synthetic")
        self.assertEqual(result["recording_manifest"]["profile"], self.profile)
        for path in self.artifacts.rglob("*.json"):
            with self.subTest(path=path.name):
                value = json.loads(path.read_text(), parse_constant=lambda token: self.fail(f"invalid JSON number {token}"))
                json.dumps(value, allow_nan=False)
        for path in self.artifacts.rglob("*.npz"):
            with np.load(path, allow_pickle=False) as archive:
                for name in archive.files:
                    self.assertNotEqual(archive[name].dtype.kind, "O")

    def test_runtime_replay_ignores_poisoned_evaluator_truth(self):
        from ares.replay import replay_recording
        baseline = json.loads((self.artifacts / "replay.json").read_text())
        truth = self.recording / "ground_truth.npz"
        original = truth.read_bytes()
        try:
            truth.write_bytes(b"poison: evaluator truth must never be read by replay")
            replay = replay_recording(self.recording, profile=self.profile)
        finally:
            truth.write_bytes(original)
        for name in ("trajectory", "map", "map_counts", "frames", "accepted_frames", "source", "profile_id", "seed"):
            self.assertEqual(replay[name], baseline[name], name)
        self.assertEqual(replay["source"], "synthetic")
        self.assertIsNone(replay["loop_error_m"])
        self.assertEqual(replay["gyro_noise_source"], "supplied_profile")

    def test_require_complete_cli_exit_and_real_fit_flag_do_not_pass_gates(self):
        from ares.calibration import fit_depth_noise
        from ares.cli import main
        for complete, expected in ((False, 0), (True, 1)):
            arguments = ["phase2", "--output", str(self.directory / f"cli-{complete}.json")]
            if complete:
                arguments.append("--require-complete")
            with patch("ares.phase2.run_phase2", return_value=self.result), redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), expected)
        reference = np.linspace(.5, 4, 2000)
        measured = reference + np.random.default_rng(4).normal(0, .01, len(reference))
        fit = fit_depth_noise(measured, reference, np.zeros_like(reference), source="real")
        self.assertEqual(fit["source"], "real")
        self.assertIn("not verified", fit["provenance"])
        self.assertNotIn("phase_passed", fit)
        self.assertFalse(self.result["phase_passed"])
        self.assertTrue(all(not gate["passed"] for gate in self.result["hardware_gates"].values()))

    def test_registered_comparison_pairs_and_bad_registration(self):
        from ares.depth_comparison import compare_registered_depth
        output = self.directory / "pairs.npz"
        stats = compare_registered_depth(self.recording, self.scene, self.profile,
                                         self.poses_path, pairs_output=output)
        self.assertEqual(stats["source"], "synthetic")
        self.assertEqual(stats["frames"], 12)
        self.assertGreater(stats["compared_pixels"], 100)
        self.assertLess(stats["rmse_m"], .1)
        self.assertGreaterEqual(stats["mae_m"], abs(stats["bias_m"]))
        self.assertTrue(0 <= stats["missing_measured_fraction"] <= 1)
        json.dumps(stats, allow_nan=False)
        with np.load(output, allow_pickle=False) as pairs:
            self.assertEqual(set(pairs.files), {"measured_m", "reference_m", "material_ids"})
            self.assertEqual(pairs["measured_m"].shape, (12, 24, 32))
            mask = np.isfinite(pairs["measured_m"]) & np.isfinite(pairs["reference_m"])
            error = pairs["measured_m"][mask] - pairs["reference_m"][mask]
            self.assertAlmostEqual(stats["rmse_m"], float(np.sqrt(np.mean(error ** 2))), places=7)
        for kind in ("time", "pose"):
            times, poses = self.times.copy(), self.poses.copy()
            if kind == "time":
                times[1] += .001
            else:
                poses[1, 0, 0] = 10
            invalid = self.directory / f"bad-{kind}.npz"
            np.savez(invalid, timestamps_s=times, camera_to_world=poses)
            with self.assertRaises(ValueError):
                compare_registered_depth(self.recording, self.scene, self.profile, invalid)

    def test_recording_intrinsics_override_profile_with_offcenter_camera(self):
        from ares.depth import GroundTruthDepthRenderer
        from ares.depth_comparison import compare_registered_depth
        from ares.recording import CameraIntrinsics, read_recording, write_recording
        metadata, arrays = read_recording(self.recording)
        metadata = copy.deepcopy(metadata)
        metadata["camera"]["cx"] -= 4
        metadata["camera"]["cy"] += 2
        camera = CameraIntrinsics(**metadata["camera"])
        renderer = GroundTruthDepthRenderer(self.scene, self.profile, backend="cpu", camera=camera)
        poses = self.poses[:2]
        ideal = []
        for index, pose in enumerate(poses):
            rendered = renderer.render(pose[:3, 3], pose[:3, :3], frame_index=index, noise=False)
            ideal.append(renderer.to_host(rendered.depth_m)[0])
        arrays["depth_m"] = np.asarray(ideal, dtype=np.float32)
        arrays["timestamps_s"] = self.times[:2]
        recording = self.directory / "offcenter"
        write_recording(recording, metadata, arrays)
        poses_path = self.directory / "offcenter-poses.npz"
        np.savez(poses_path, timestamps_s=self.times[:2], camera_to_world=poses)
        stats = compare_registered_depth(recording, self.scene, self.profile, poses_path)
        self.assertEqual(stats["camera"], metadata["camera"])
        self.assertEqual(stats["rmse_m"], 0.)
        self.assertGreater(stats["compared_pixels"], 100)

    def test_invalid_options_fail_before_overwriting_products(self):
        from ares.phase2 import run_phase2
        sentinel = self.recording / "metadata.json"
        original = sentinel.read_bytes()
        for overrides in ({"frames": -1}, {"frames": 3}, {"batch_size": -1}, {"batch_size": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                run_phase2(self.scene, self.profile, self.artifacts, **overrides)
            self.assertEqual(sentinel.read_bytes(), original)
