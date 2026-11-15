"""Disk-backed capture validation must inspect the final partial chunk."""

from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("NumPy simulation extra is not installed")

from ares.contracts import ContractError
from ares.recording import CameraIntrinsics, validate_arrays


class ChunkValidationTests(unittest.TestCase):
    def test_memmap_final_chunk_is_validated_and_nan_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            depth = np.lib.format.open_memmap(Path(directory) / "depth.npy", mode="w+", dtype=np.float32, shape=(17, 3, 5))
            depth[:] = 1.
            depth[-1, 1, 1] = np.nan
            arrays = {"timestamps_s": np.arange(17, dtype=float), "depth_m": depth,
                      "imu_timestamps_s": np.zeros(0), "imu_accel_mps2": np.zeros((0, 3)),
                      "imu_gyro_rps": np.zeros((0, 3))}
            camera = CameraIntrinsics(5, 3, 4., 4., 2., 1.)
            validate_arrays(arrays, camera)
            for value in (0., -1., np.inf):
                depth[-1, 1, 1] = value
                with self.subTest(value=value), self.assertRaisesRegex(ContractError, "positive optical Z or NaN"):
                    validate_arrays(arrays, camera)
            depth._mmap.close()
