import json
import unittest
try:
    import numpy as np
except ImportError:
    np = None
from ares.calibration import fit_depth_noise


@unittest.skipIf(np is None, "NumPy optional")
class CalibrationTests(unittest.TestCase):
    def test_recovers_coefficients_and_missing_rates(self):
        rng = np.random.default_rng(123)
        reference = rng.uniform(.5, 6, 120000)
        material = rng.integers(0, 5, len(reference))
        measured = reference + rng.normal(size=len(reference)) * np.sqrt(.02 ** 2 + (.004 * reference ** 2) ** 2)
        measured[rng.random(len(reference)) < material * .1] = np.nan
        fit = fit_depth_noise(measured, reference, material)
        self.assertAlmostEqual(fit["base_std_m"], .02, delta=.004)
        self.assertAlmostEqual(fit["quadratic_std_per_m2"], .004, delta=.0003)
        self.assertAlmostEqual(fit["material_hole_probability"][4], .4, delta=.02)
        self.assertEqual(fit["source"], "synthetic")
        self.assertIn("not real", fit["provenance"])
        json.dumps(fit, allow_nan=False)

    def test_bad_coverage_and_inputs(self):
        with self.assertRaises(ValueError):
            fit_depth_noise(np.ones(1000), np.ones(1000), np.zeros(1000))
        with self.assertRaises(ValueError):
            fit_depth_noise([1], [1, 2], [0])
        with self.assertRaises(ValueError):
            fit_depth_noise([1], [1], [0], source="unknown")

    def test_unobserved_material_null_and_real_provenance(self):
        rng = np.random.default_rng(1)
        reference = np.linspace(.5, 5, 1000)
        fit = fit_depth_noise(reference + rng.normal(0, .01, 1000), reference,
                              np.zeros(1000), source="real")
        self.assertIsNone(fit["material_hole_probability"][1])
        self.assertEqual(fit["source"], "real")
        json.dumps(fit, allow_nan=False)
