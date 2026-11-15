import unittest
try:
    import numpy as np
except ImportError:
    np = None
from ares.sensors import SensorSuite, DepthNoiseModel, apply_depth_noise


def profile():
    return {"imu_accel_noise_std": 0., "imu_gyro_noise_std": 0.,
            "odometry_drift_per_m": .03, "depth_max_range_m": 10.,
            "optical_flow_noise_std_mps": 0., "range_noise_std_m": 0.}


@unittest.skipIf(np is None, "NumPy optional")
class SensorTests(unittest.TestCase):
    def test_gravity_and_rotated_axes(self):
        suite = SensorSuite(profile(), 12)
        rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        result = suite.imu([1, 2, 0], [1, 2, 3], rotation, .01)
        np.testing.assert_allclose(result["accel_mps2"], [-9.80665, 2, 1])
        np.testing.assert_allclose(result["gyro_rps"], [1, 2, 3])
        with self.assertRaises(ValueError):
            suite.imu([0, 0, 0], [0, 0, 0], np.diag([1, 1, -1]), .01)
        with self.assertRaises(ValueError):
            suite.drift([0, 0, 0], 0)

    def test_flow(self):
        suite = SensorSuite(profile(), 3)
        result = suite.optical_flow([1, 2, 3], 2, .1)
        np.testing.assert_equal(result["velocity_body_mps2d"], [1, 2])
        self.assertEqual(result["range_m"], 2)
        for height in [0, -1, 11, np.nan, np.inf]:
            result = suite.optical_flow([1, 2, 3], height, .1)
            self.assertFalse(result["valid"])
            self.assertTrue(np.isnan(result["range_m"]))

    def test_drift_bound_and_seed(self):
        a, b = SensorSuite(profile(), 1), SensorSuite(profile(), 1)
        for distance in range(101):
            point = np.array([distance, 0., 0.])
            output = a.drift(point, .1)
            np.testing.assert_array_equal(output, b.drift(point, .1))
            self.assertLessEqual(np.linalg.norm(output - point), .03 * distance + 1e-12)
        self.assertGreater(np.linalg.norm(output - point), 1.)
        np.testing.assert_array_equal(a.drift(point, 1), output)

    def test_depth_variance_and_no_global_rng(self):
        model = DepthNoiseModel(base_std_m=.01, quadratic_std_per_m2=.01, max_range_m=10,
                                edge_width_px=0, max_range_dropout_probability=0,
                                material_hole_probability=(0, 0, 0, 0, 0))
        depth = np.stack([np.ones((100, 100)), np.full((100, 100), 4.)])
        mat = np.zeros(depth.shape, dtype=int)
        np.random.seed(1)
        output = apply_depth_noise(depth, mat, model, 42, 0)
        self.assertEqual(np.random.random(), np.random.RandomState(1).random())
        np.testing.assert_array_equal(output, apply_depth_noise(depth, mat, model, 42, 0))
        self.assertGreater(np.std(output[1]), 8 * np.std(output[0]))
        self.assertFalse(np.array_equal(output, apply_depth_noise(depth, mat, model, 42, 1)))

    def test_holes_edges_and_nan(self):
        depth = np.ones((10, 20))
        depth[:, 10:] = 3.
        depth[0, 0] = np.nan
        model = DepthNoiseModel(base_std_m=0, quadratic_std_per_m2=0, max_range_m=10,
                                edge_width_px=2, max_range_dropout_probability=0,
                                material_hole_probability=(0, 0, 0, 1, 1))
        mat = np.zeros(depth.shape, dtype=int)
        output = apply_depth_noise(depth, mat, model, 0, 0)
        self.assertTrue(np.isnan(output[:, 8:12]).all())
        self.assertTrue(np.isfinite(output[5, :8]).all())
        self.assertTrue(np.isnan(output[0, 0]))
        mat[:] = 3
        self.assertTrue(np.isnan(apply_depth_noise(depth, mat, model, 0, 0)).all())

    def test_small_images_wide_edge_band_and_range(self):
        model = DepthNoiseModel(edge_width_px=32, min_range_m=.2)
        depth = np.array([[.1, 1., 2.]])
        result = apply_depth_noise(depth, np.zeros_like(depth), model, 1, 0)
        self.assertTrue(np.isnan(result).all())

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            DepthNoiseModel(material_hole_probability=(0, 0))
        with self.assertRaises(ValueError):
            SensorSuite({**profile(), "imu_accel_noise_std": np.nan}, 0)
