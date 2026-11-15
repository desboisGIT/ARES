"""Phase 2 recording validation; optional NumPy never blocks Phase 0/1."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Phase 2 recording tests require optional numpy")

from ares.contracts import ContractError
from ares.recording import CameraIntrinsics, from_profile, read_recording, write_recording


def example_recording():
    metadata = {"schema_version": "ares.sensor_recording.v1", "recording_id": "test", "source": "synthetic",
                "seed": 7, "profile_id": "test", "frames_file": "frames.npz",
                "camera": {"width": 3, "height": 2, "fx": 2., "fy": 2., "cx": 1., "cy": .5}}
    arrays = {"timestamps_s": np.array([0., .1]), "depth_m": np.ones((2, 2, 3), dtype=np.float32),
              "imu_timestamps_s": np.array([0., .05, .1]), "imu_accel_mps2": np.zeros((3, 3)),
              "imu_gyro_rps": np.zeros((3, 3))}
    arrays["depth_m"][0, 0, 0] = np.nan
    return metadata, arrays


class RecordingTests(unittest.TestCase):
    def test_round_trip_and_ground_truth_firewall(self):
        metadata, arrays = example_recording()
        with tempfile.TemporaryDirectory() as directory:
            # Native device files may already exist; never parse an oracle file.
            (Path(directory) / "capture.bag").write_bytes(b"native data")
            write_recording(directory, metadata, arrays)
            (Path(directory) / "ground_truth.npz").write_bytes(b"deliberately not a valid archive")
            original_load = np.load
            paths = []
            def tracked_load(path, **kwargs):
                paths.append(Path(path).name)
                self.assertFalse(kwargs["allow_pickle"])
                return original_load(path, **kwargs)
            with patch("ares.recording.np.load", side_effect=tracked_load):
                actual_metadata, actual = read_recording(directory)
            self.assertEqual(paths, ["frames.npz"])
            self.assertEqual(metadata, actual_metadata)
            for name, value in arrays.items():
                np.testing.assert_equal(value, actual[name])
            with self.assertRaisesRegex(ContractError, "existing output"):
                write_recording(directory, metadata, arrays)

    def test_metadata_rejects_unknown_and_path_traversal(self):
        for changes in ({"oracle_pose": []}, {"frames_file": "../frames.npz"},
                        {"seed": True}, {"source": "simulation"}):
            metadata, arrays = example_recording()
            metadata.update(changes)
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(ContractError):
                write_recording(directory, metadata, arrays)

    def test_symlink_frames_may_not_escape_directory(self):
        metadata, arrays = example_recording()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            outside, inside = base / "outside", base / "inside"
            write_recording(outside, metadata, arrays)
            inside.mkdir()
            (inside / "metadata.json").write_text(json.dumps(metadata))
            (inside / "frames.npz").symlink_to(outside / "frames.npz")
            with self.assertRaisesRegex(ContractError, "escape"):
                read_recording(inside)

    def test_bad_sensor_shapes_and_values_rejected(self):
        mutations = [lambda a: a.update(timestamps_s=np.array([.1, .1])),
                     lambda a: a.update(imu_timestamps_s=np.array([0., .1, .05])),
                     lambda a: a.update(depth_m=a["depth_m"].astype(np.float64)),
                     lambda a: a["depth_m"].__setitem__((0, 0, 1), 0),
                     lambda a: a["depth_m"].__setitem__((0, 0, 1), np.inf),
                     lambda a: a["imu_gyro_rps"].__setitem__((0, 0), np.nan),
                     lambda a: a.update(imu_accel_mps2=np.zeros((2, 3))),
                     lambda a: a.update(ground_truth_pose=np.eye(4)),
                     lambda a: a.update(flow_range_m=np.array([1.]))]
        for mutate in mutations:
            metadata, arrays = example_recording()
            mutate(arrays)
            with tempfile.TemporaryDirectory() as directory, self.assertRaises(ContractError):
                write_recording(directory, metadata, arrays)

    def test_flow_and_empty_imu_are_supported(self):
        metadata, arrays = example_recording()
        arrays.update(imu_timestamps_s=np.empty(0), imu_accel_mps2=np.empty((0, 3)), imu_gyro_rps=np.empty((0, 3)),
                      flow_timestamps_s=np.array([0., .1]), flow_velocity_body_mps=np.zeros((2, 2)),
                      flow_range_m=np.ones(2))
        with tempfile.TemporaryDirectory() as directory:
            write_recording(directory, metadata, arrays)
            self.assertEqual(read_recording(directory)[1]["imu_accel_mps2"].shape, (0, 3))

    def test_pickle_and_duplicate_json_are_rejected_on_read(self):
        metadata, arrays = example_recording()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_recording(path, metadata, arrays)
            np.savez(path / "frames.npz", **{**arrays, "depth_m": np.array([object()], dtype=object)})
            with self.assertRaises(ContractError):
                read_recording(path)
            (path / "metadata.json").write_text('{"schema_version":1,"schema_version":2}')
            with self.assertRaisesRegex(ContractError, "duplicate"):
                read_recording(path)

    def test_intrinsics_from_profile(self):
        profile_path = Path(__file__).resolve().parents[1] / "ares/data/configs/drone_nominal.json"
        profile = json.loads(profile_path.read_text())
        camera = from_profile(profile)
        self.assertIsInstance(camera, CameraIntrinsics)
        self.assertGreater(camera.fx, 0)
        self.assertEqual(camera.cx, (camera.width - 1) / 2)
        with self.assertRaises(ValueError):
            CameraIntrinsics(10, 10, -1., 1., 4.5, 4.5)
