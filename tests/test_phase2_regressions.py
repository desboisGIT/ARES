"""Preserve user data and earlier gate evidence across failures and reruns."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Phase 2 regressions require optional numpy")


ROOT = Path(__file__).resolve().parents[1]


def snapshot(directory):
    """Capture bytes and link destinations without following directory links."""
    return {str(path.relative_to(directory)): ("link", str(path.readlink())) if path.is_symlink()
            else ("bytes", path.read_bytes())
            for path in directory.rglob("*") if path.is_symlink() or path.is_file()}


class Phase2PreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ares.contracts import load_json
        from ares.phase2 import default_profile, run_phase2
        from ares.recording import read_recording, write_recording
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name)
        cls.scene = load_json(ROOT / "ares/data/scenes/three_room.json")
        cls.profile = default_profile()
        cls.fixture = cls.base / "fixture"
        result = run_phase2(cls.scene, cls.profile, cls.fixture, backend="cpu", batch_size=1, frames=12)
        if not result["software_preparation_passed"]:
            raise AssertionError(result["software_checks"])
        metadata, arrays = read_recording(cls.fixture / "synthetic_recording")
        arrays["timestamps_s"], arrays["depth_m"] = arrays["timestamps_s"][:2], arrays["depth_m"][:2]
        cls.comparison_recording = cls.base / "comparison_recording"
        write_recording(cls.comparison_recording, metadata, arrays)
        with np.load(cls.fixture / "synthetic_recording/ground_truth.npz", allow_pickle=False) as truth:
            poses = np.tile(np.eye(4), (2, 1, 1))
            poses[:, :3, 3] = truth["positions_world"][:2]
            poses[:, :3, :3] = truth["camera_to_world_rotation"][:2]
        cls.poses_path = cls.base / "registered_poses.npz"
        np.savez(cls.poses_path, timestamps_s=arrays["timestamps_s"], camera_to_world=poses)

    def setUp(self):
        self.working = tempfile.TemporaryDirectory(dir=self.base)
        self.addCleanup(self.working.cleanup)
        self.directory = Path(self.working.name)

    def owned_copy(self, name="artifacts", *, bind_to_copy=True):
        destination = self.directory / name
        shutil.copytree(self.fixture, destination)
        if bind_to_copy:
            path = destination / "phase2_gate_owner.json"
            owner = json.loads(path.read_text())
            owner["directory"] = str(destination.resolve())
            path.write_text(json.dumps(owner))
        return destination

    def run_gate(self, output, *, profile=None):
        from ares.phase2 import run_phase2
        return run_phase2(self.scene, self.profile if profile is None else profile, output,
                          backend="cpu", batch_size=1, frames=12)

    def test_unowned_real_and_synthetic_recordings_are_preserved(self):
        for source in ("real", "synthetic"):
            output = self.directory / source
            recording = output / "synthetic_recording"
            recording.mkdir(parents=True)
            (recording / "metadata.json").write_text(json.dumps({"source": source}))
            (recording / "frames.npz").write_bytes(b"user recording bytes")
            before = snapshot(output)
            with self.assertRaisesRegex(ValueError, "unowned"):
                self.run_gate(output)
            self.assertEqual(snapshot(output), before)

    def test_output_and_recording_directory_symlinks_preserve_targets(self):
        target = self.owned_copy("target")
        before = snapshot(target)
        output_link = self.directory / "output-link"
        output_link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.run_gate(output_link)
        self.assertTrue(output_link.is_symlink())
        self.assertEqual(snapshot(target), before)

        owned = self.owned_copy("recording-link-owner")
        shutil.rmtree(owned / "synthetic_recording")
        (owned / "synthetic_recording").symlink_to(target / "synthetic_recording", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "recording"):
            self.run_gate(owned)
        self.assertTrue((owned / "synthetic_recording").is_symlink())
        self.assertEqual(snapshot(target), before)

    def test_malformed_profile_does_not_touch_previous_artifacts(self):
        output = self.owned_copy()
        before = snapshot(output)
        profile = copy.deepcopy(self.profile)
        profile["parameters"]["depth_fps"]["nominal"] = -1.
        with self.assertRaises(ValueError):
            self.run_gate(output, profile=profile)
        self.assertEqual(snapshot(output), before)

    def test_changed_metadata_manifest_and_known_artifact_are_refused(self):
        paths = ("synthetic_recording/metadata.json", "synthetic_recording/simulation_manifest.json",
                 "synthetic_recording/frames.npz", "replay.json")
        for index, relative in enumerate(paths):
            with self.subTest(artifact=relative):
                output = self.owned_copy(f"changed-{index}")
                path = output / relative
                if relative.endswith("metadata.json"):
                    metadata = json.loads(path.read_text())
                    metadata["recording_id"] = "unrelated-synthetic-recording"
                    path.write_text(json.dumps(metadata))
                elif relative.endswith("simulation_manifest.json"):
                    manifest = json.loads(path.read_text())
                    manifest["seed"] += 1
                    path.write_text(json.dumps(manifest))
                else:
                    with path.open("ab") as stream:
                        stream.write(b"modified")
                before = snapshot(output)
                with patch("ares.phase2._build_phase2", side_effect=AssertionError("must reject before rebuilding")):
                    with self.assertRaises(ValueError):
                        self.run_gate(output)
                self.assertEqual(snapshot(output), before)

    def test_ownership_is_bound_to_exact_directory_and_file_set(self):
        wrong_path = self.owned_copy("copied-with-original-owner", bind_to_copy=False)
        before = snapshot(wrong_path)
        with self.assertRaisesRegex(ValueError, "modified"):
            self.run_gate(wrong_path)
        self.assertEqual(snapshot(wrong_path), before)

        unrelated = self.owned_copy("unrelated-file")
        (unrelated / "user-notes.txt").write_text("retain this")
        before = snapshot(unrelated)
        with self.assertRaisesRegex(ValueError, "unrelated"):
            self.run_gate(unrelated)
        self.assertEqual(snapshot(unrelated), before)

    def test_build_failure_preserves_prior_bytes_and_caller_report(self):
        output = self.owned_copy()
        caller_report = self.directory / "phase_2.json"
        caller_report.write_bytes(b"previous caller report")
        before = snapshot(output)
        def failing_build(scene, profile, staging, **kwargs):
            Path(staging).mkdir(parents=True)
            (Path(staging) / "partial-output").write_bytes(b"incomplete new run")
            raise RuntimeError("injected build failure")
        with patch("ares.phase2._build_phase2", side_effect=failing_build):
            with self.assertRaisesRegex(RuntimeError, "injected build failure"):
                self.run_gate(output)
        self.assertEqual(snapshot(output), before)
        self.assertEqual(caller_report.read_bytes(), b"previous caller report")
        self.assertFalse(list(self.directory.glob(".ares-phase2-*")))

    def test_extensionless_comparison_pairs_use_exact_path_and_refuse_overwrite(self):
        from ares.depth_comparison import compare_registered_depth
        output = self.directory / "comparison-pairs"
        stats = compare_registered_depth(self.comparison_recording, self.scene, self.profile,
                                         self.poses_path, pairs_output=output)
        self.assertGreater(stats["compared_pixels"], 0)
        self.assertTrue(output.is_file())
        self.assertFalse(output.with_suffix(".npz").exists())
        with np.load(output, allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), {"measured_m", "reference_m", "material_ids"})
        original = output.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            compare_registered_depth(self.comparison_recording, self.scene, self.profile,
                                     self.poses_path, pairs_output=output)
        self.assertEqual(output.read_bytes(), original)

    def test_dangling_comparison_output_symlink_is_not_followed(self):
        from ares.depth_comparison import compare_registered_depth
        target = self.directory / "absent-target.npz"
        output = self.directory / "pairs.npz"
        output.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            compare_registered_depth(self.comparison_recording, self.scene, self.profile,
                                     self.poses_path, pairs_output=output)
        self.assertTrue(output.is_symlink())
        self.assertFalse(target.exists())


class CUDAUniformRegressionTests(unittest.TestCase):
    def test_uniform_excludes_one_for_maximum_mixed_uint32(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                self.skipTest("CUDA uniform regression requires a GPU")
        except ImportError:
            self.skipTest("CUDA uniform regression requires optional CuPy")
        except (RuntimeError, OSError) as exc:
            self.skipTest(f"CUDA unavailable: {exc}")
        from ares.depth import _NOISE
        # Reverse the bijective uint32 mixer to force its exact maximum output.
        mask = 2**32 - 1
        def undo_right_xor(value, shift):
            output = value
            for _ in range(32 // shift):
                output = value ^ (output >> shift)
            return output & mask
        key = undo_right_xor(mask, 16)
        key = key * pow(0x846ca68b, -1, 2**32) & mask
        key = undo_right_xor(key, 15)
        key = key * pow(0x7feb352d, -1, 2**32) & mask
        key = undo_right_xor(key, 16)
        wrapper = '\nextern "C" __global__ void sample_uniform(unsigned int key, float* out) { out[0]=uniform01(key); }\n'
        kernel = cp.RawKernel(_NOISE + wrapper, "sample_uniform", options=("--std=c++11",))
        output = cp.empty(1, dtype=cp.float32)
        kernel((1,), (1,), (np.uint32(key), output))
        actual = float(output.get()[0])
        self.assertEqual(actual, 1. - 2.**-24)
        self.assertLess(actual, 1.)
