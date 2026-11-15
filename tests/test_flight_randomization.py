import copy
from pathlib import Path
import unittest

from ares.contracts import ContractError, load_json, validate_profile
from ares.flight_randomization import (randomization_manifest, sample_flight_profile,
                                       sampled_profile_values)


ROOT = Path(__file__).resolve().parents[1]


class FlightRandomizationTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_json(ROOT / "ares/data/configs/drone_sim_v2.json")

    def test_every_component_seeded_bounded_and_immutable(self):
        original = copy.deepcopy(self.profile)
        samples = [sample_flight_profile(self.profile, seed) for seed in range(40)]
        self.assertEqual(self.profile, original)
        self.assertEqual(samples[7], sample_flight_profile(self.profile, 7))
        reordered = copy.deepcopy(self.profile)
        reordered["parameters"] = dict(reversed(list(reordered["parameters"].items())))
        self.assertEqual(samples[7], sample_flight_profile(reordered, 7))
        self.assertNotEqual(samples[0], samples[1])
        manifest = randomization_manifest(samples)
        self.assertEqual(manifest["parameter_count"], 29)
        self.assertEqual(manifest["dimension_count"], 35)
        self.assertTrue(manifest["all_within_bounds"])
        for sample in samples:
            validate_profile(sample)
            self.assertEqual(sample["profile_id"], self.profile["profile_id"])
            self.assertEqual(sample["schema_version"], self.profile["schema_version"])
            for name, parameter in sample["parameters"].items():
                self.assertEqual(parameter["range"], original["parameters"][name]["range"])
                if name in ("depth_width", "depth_height", "depth_edge_width_px"):
                    self.assertIs(type(parameter["nominal"]), int)
        # Every nonconstant scalar/component actually varies across these seeds.
        for name, parameter in original["parameters"].items():
            vector = isinstance(parameter["nominal"], list)
            ranges = parameter["range"] if vector else [parameter["range"]]
            for i, (low, high) in enumerate(ranges):
                observed = {s["parameters"][name]["nominal"][i] if vector else
                            s["parameters"][name]["nominal"] for s in samples}
                self.assertTrue(all(low <= value <= high for value in observed))
                if low != high:
                    self.assertGreater(len(observed), 1, (name, i))
        self.assertEqual({s["parameters"]["depth_edge_width_px"]["nominal"] for s in samples}, {0, 1, 2})

    def test_all_endpoints_and_nominal(self):
        low = sample_flight_profile(self.profile, 123, "low")
        high = sample_flight_profile(self.profile, 123, "high")
        for mode, sample in ((0, low), (1, high)):
            for parameter in sample["parameters"].values():
                expected = ([r[mode] for r in parameter["range"]]
                            if isinstance(parameter["nominal"], list) else parameter["range"][mode])
                self.assertEqual(parameter["nominal"], expected)
        manifest = randomization_manifest([low, high])
        self.assertTrue(manifest["all_endpoints_seen"])
        self.assertEqual(sample_flight_profile(self.profile, 0, "nominal"), self.profile)
        self.assertFalse(randomization_manifest(sample_flight_profile(self.profile, 8))["all_endpoints_seen"])

    def test_deep_copy_and_values_record(self):
        sampled = sample_flight_profile(self.profile, 42)
        values = sampled_profile_values(sampled)
        self.assertEqual(set(values), set(self.profile["parameters"]))
        self.assertEqual(values["compute_latency_s"], sampled["parameters"]["compute_latency_s"]["nominal"])
        values["inertia_kg_m2"][0] = 999
        sampled["parameters"]["inertia_kg_m2"]["range"][0][0] = 999
        self.assertNotEqual(sampled["parameters"]["inertia_kg_m2"]["nominal"][0], 999)
        self.assertNotEqual(self.profile["parameters"]["inertia_kg_m2"]["range"][0][0], 999)

    def test_v1_and_bad_inputs(self):
        v1 = load_json(ROOT / "ares/data/configs/drone_nominal.json")
        validate_profile(sample_flight_profile(v1, -23))
        for seed in (True, 1.0, "1", None):
            with self.assertRaisesRegex(ValueError, "seed"):
                sample_flight_profile(self.profile, seed)
        with self.assertRaisesRegex(ValueError, "mode"):
            sample_flight_profile(self.profile, 0, "unknown")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            randomization_manifest([])
        with self.assertRaisesRegex(ValueError, "share"):
            randomization_manifest([self.profile, v1])
        changed = copy.deepcopy(self.profile)
        changed["parameters"]["mass_kg"]["range"][0] = .44
        with self.assertRaisesRegex(ValueError, "share"):
            randomization_manifest([self.profile, changed])
        self.profile["parameters"]["mass_kg"]["nominal"] = 999
        with self.assertRaises(ContractError):
            sample_flight_profile(self.profile, 1)


if __name__ == "__main__":
    unittest.main()
