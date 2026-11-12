import copy
import json
import unittest

from ares.contracts import ContractError
from ares.evaluator import GroundTruthEpisode, GroundTruthSample
from ares.flight_log import make_flight_log, validate_flight_log


class FlightLogTests(unittest.TestCase):
    def setUp(self):
        self.episode = GroundTruthEpisode(
            samples=(GroundTruthSample(0.0, (1., 2., 3.), False),
                     GroundTruthSample(0.5, (2., 2., 3.), True)),
            observations=(((1, 2, 3), 1),), collisions=0,
            left_bounds=False, minimum_clearance_m=1., landed_safely=True, seed=42, world_digest="test-fixture")
        self.log = make_flight_log(self.episode, scene_id="scene", profile_id="profile")

    def test_runtime_json_roundtrip(self):
        self.assertNotIn("ground_truth", self.log)
        self.assertEqual(validate_flight_log(json.loads(json.dumps(self.log))), self.log)
        self.assertIsNone(self.log["samples"][0]["imu"])
        self.assertEqual(self.log["samples"][0]["timestamp_s"], 0)

    def test_truth_requires_explicit_permission_and_is_isolated(self):
        log = make_flight_log(self.episode, scene_id="scene", profile_id="profile",
                              include_ground_truth=True)
        with self.assertRaisesRegex(ContractError, "evaluator-only"):
            validate_flight_log(log)
        validate_flight_log(log, allow_ground_truth=True)
        log["ground_truth"]["samples"][0]["position"][0] = 999
        del log["ground_truth"]
        self.assertEqual(log, self.log)

    def test_rejects_invalid_contracts(self):
        mutations = [
            lambda log: log.update(schema_version="ares.flight_log.v2"),
            lambda log: log.update(extra=True),
            lambda log: log.update(seed=True),
            lambda log: log.update(samples=[]),
            lambda log: log["samples"][0].update(imu={}),
            lambda log: log["samples"][0]["pose_estimate"].update(position=[1, 2]),
            lambda log: log["samples"][0]["pose_estimate"].update(quaternion_xyzw=[0]*4),
        ]
        for mutate in mutations:
            log = copy.deepcopy(self.log)
            mutate(log)
            with self.subTest(log=log), self.assertRaises(ContractError):
                validate_flight_log(log)

    def test_rejects_nonfinite_and_nonincreasing_timestamps(self):
        for timestamp in (float("nan"), float("inf"), -1, 0.1, True, "0"):
            log = copy.deepcopy(self.log)
            log["samples"][0]["timestamp_s"] = timestamp
            with self.subTest(timestamp=timestamp), self.assertRaises(ContractError):
                validate_flight_log(log)
        for timestamp in (0., -0.1):
            log = copy.deepcopy(self.log)
            log["samples"][1]["timestamp_s"] = timestamp
            with self.assertRaises(ContractError):
                validate_flight_log(log)
        log = copy.deepcopy(self.log)
        log["samples"][0]["pose_estimate"]["position"][0] = float("nan")
        with self.assertRaises(ContractError):
            validate_flight_log(log)


if __name__ == "__main__":
    unittest.main()
