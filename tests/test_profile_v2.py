"""Profile v2 contract checks require only the Python standard library."""
import copy
import math
from pathlib import Path
import unittest

from ares.contracts import ContractError, load_json, validate_profile


ROOT = Path(__file__).resolve().parents[1]
NEW_PARAMETERS = {
    "gravity_mps2", "depth_min_range_m", "depth_noise_quadratic_per_m2",
    "depth_edge_width_px", "depth_edge_threshold_m", "max_range_dropout_probability",
    "optical_flow_noise_std_mps", "range_noise_std_m", "imu_rate_hz",
    "optical_flow_rate_hz", "material_hole_probability",
}


class ProfileV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1 = load_json(ROOT / "ares/data/configs/drone_nominal.json")
        cls.v2 = load_json(ROOT / "ares/data/configs/drone_sim_v2.json")

    def reject(self, profile):
        with self.assertRaises(ContractError):
            validate_profile(profile)

    def test_packaged_profiles_and_legacy_parameters(self):
        self.assertIsNone(validate_profile(self.v1))
        self.assertIsNone(validate_profile(self.v2))
        self.assertEqual(self.v1["schema_version"], "ares.profile.v1")
        self.assertEqual(set(self.v2["parameters"]) - set(self.v1["parameters"]), NEW_PARAMETERS)
        for name, value in self.v1["parameters"].items():
            self.assertEqual(self.v2["parameters"][name], value, name)

    def test_v1_rejects_each_v2_extra(self):
        for name in NEW_PARAMETERS:
            with self.subTest(name=name):
                profile = copy.deepcopy(self.v1)
                profile["parameters"][name] = copy.deepcopy(self.v2["parameters"][name])
                self.reject(profile)

    def test_v2_requires_every_new_parameter(self):
        for name in NEW_PARAMETERS:
            with self.subTest(name=name):
                profile = copy.deepcopy(self.v2)
                del profile["parameters"][name]
                self.reject(profile)

    def test_all_new_values_and_bounds_are_finite_and_not_bool(self):
        for name in NEW_PARAMETERS:
            paths = [("nominal",), ("range", 0), ("range", 1)]
            if name == "material_hole_probability":
                paths = [("nominal", i) for i in range(5)]
                paths += [("range", i, j) for i in range(5) for j in range(2)]
            for path in paths:
                for invalid in (True, False, math.nan, math.inf, -math.inf):
                    with self.subTest(name=name, path=path, invalid=invalid):
                        profile = copy.deepcopy(self.v2)
                        node = profile["parameters"][name]
                        for key in path[:-1]:
                            node = node[key]
                        node[path[-1]] = invalid
                        self.reject(profile)

    def test_probability_bounds_and_material_vector_shapes(self):
        for name in ("material_hole_probability", "max_range_dropout_probability"):
            for invalid in (-.1, 1.1):
                with self.subTest(name=name, invalid=invalid):
                    profile = copy.deepcopy(self.v2)
                    record = {"nominal": invalid, "range": [invalid, invalid]}
                    if name == "material_hole_probability":
                        record = {"nominal": [invalid] * 5, "range": [[invalid, invalid] for _ in range(5)]}
                    profile["parameters"][name] = record
                    self.reject(profile)
        for field in ("nominal", "range"):
            for length in (0, 4, 6):
                profile = copy.deepcopy(self.v2)
                profile["parameters"]["material_hole_probability"][field] = (
                    [0.] * length if field == "nominal" else [[0., 1.]] * length)
                self.reject(profile)
        profile = copy.deepcopy(self.v2)
        profile["parameters"]["material_hole_probability"] = {
            "nominal": [0, 1, 0, 1, 0], "range": [[0, 1]] * 5}
        self.assertIsNone(validate_profile(profile))

    def test_edge_width_requires_integer_inclusive_zero_to_32(self):
        for value in (0, 32):
            profile = copy.deepcopy(self.v2)
            profile["parameters"]["depth_edge_width_px"] = {"nominal": value, "range": [value, value]}
            self.assertIsNone(validate_profile(profile))
        for value in (-1, 33, 1.0, 1.5):
            profile = copy.deepcopy(self.v2)
            profile["parameters"]["depth_edge_width_px"] = {"nominal": value, "range": [value, value]}
            self.reject(profile)
        for bounds in ([0., 2], [0, 2.], [0, 33]):
            profile = copy.deepcopy(self.v2)
            profile["parameters"]["depth_edge_width_px"]["range"] = bounds
            self.reject(profile)

    def test_minimum_range_entire_interval_stays_below_maximum(self):
        for upper in (3., 3.1):
            profile = copy.deepcopy(self.v2)
            profile["parameters"]["depth_min_range_m"]["range"] = [.15, upper]
            self.reject(profile)
        profile = copy.deepcopy(self.v2)
        profile["parameters"]["depth_min_range_m"]["range"] = [.15, 2.999]
        self.assertIsNone(validate_profile(profile))

    def test_fov_contract_is_unchanged(self):
        for original in (self.v1, self.v2):
            for name in ("depth_horizontal_fov_rad", "depth_vertical_fov_rad"):
                for bounds in ([0, 1.6], [1.4, math.pi], [1.4, math.pi + .01]):
                    profile = copy.deepcopy(original)
                    profile["parameters"][name]["range"] = bounds
                    self.reject(profile)
