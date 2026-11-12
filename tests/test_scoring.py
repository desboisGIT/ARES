import copy
from importlib.resources import files
import math
import unittest

from ares.attacks import run_attacks
from ares.contracts import load_json
from ares.evaluator import GroundTruthSample, GroundTruthWorld, segment_box_distance
from ares.reporting import make_report
from ares.scoring import score_episode


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = files("ares") / "data"
        cls.scene = load_json(data / "scenes/three_room.json")
        cls.profile = load_json(data / "configs/drone_nominal.json")
        cls.world = GroundTruthWorld(cls.scene, cls.profile)
        cls.episode = cls.world.witness(seed=23)
        cls.report = make_report(cls.world, dict(cls.episode.observations),
                                 cls.episode.samples[-1].timestamp_s, cls.scene["rooms"])

    def test_successful_witness(self):
        result = score_episode(self.world, self.episode, self.report)
        self.assertTrue(result["passed"], result["fail_reasons"])
        self.assertEqual(result["coverage"], 1)
        self.assertEqual(result["floor_plan_iou"], 1)
        self.assertEqual(len(result["rooms_found"]), 3)
        self.assertEqual(result["collisions"], 0)
        self.assertTrue(result["landed_safely"])

    def test_named_attacks(self):
        expected = {"report_instantly": "no_elapsed_time", "hover_forever": "timeout",
                    "leave_map": "left_map", "fake_map": "unsupported_map_claims"}
        attacks = run_attacks(self.world, 23)
        self.assertEqual(set(attacks), set(expected))
        for name, reason in expected.items():
            with self.subTest(name=name):
                self.assertFalse(attacks[name]["passed"])
                self.assertIn(reason, attacks[name]["fail_reasons"])

    def test_incomplete_claims_fail_despite_complete_observations(self):
        report = copy.deepcopy(self.report)
        cells = report["voxel_map"]["cells"]
        del cells[:math.ceil(len(cells) * 0.06)]
        result = score_episode(self.world, self.episode, report)
        self.assertIn("coverage", result["fail_reasons"])
        self.assertLess(result["coverage"], 0.95)
        self.assertEqual(result["unsupported_map_cells"], 0)

    def test_missing_occupied_floor_cells_fails_mean_iou(self):
        report = copy.deepcopy(self.report)
        report["floor_plan"]["cells"] = [0 if v == 2 else v for v in report["floor_plan"]["cells"]]
        result = score_episode(self.world, self.episode, report)
        self.assertEqual(result["floor_plan_iou"], 0.5)
        self.assertIn("floor_plan_iou", result["fail_reasons"])

    def test_report_cannot_shorten_elapsed_time(self):
        report = copy.deepcopy(self.report)
        report["end_time_s"] = 1
        result = score_episode(self.world, self.episode, report)
        self.assertIn("report_time_mismatch", result["fail_reasons"])
        self.assertEqual(result["time_s"], self.episode.samples[-1].timestamp_s)

    def test_returning_home_requires_land_event(self):
        episode = self.world.record(self.episode.samples, seed=23, land=False)
        result = score_episode(self.world, episode, self.report)
        self.assertIn("unsafe_landing", result["fail_reasons"])
        self.assertEqual(result["coverage"], 1)

    def test_continuous_collision_between_free_endpoints(self):
        a, b = (0.7, 0.5, 1.1), (2.7, 0.5, 1.1)
        self.assertNotIn(self.world.index(a), self.world.occupied)
        self.assertNotIn(self.world.index(b), self.world.occupied)
        samples = [GroundTruthSample(t, p, False) for t, p in
                   [(0, self.world.spawn), (1, a), (5, b),
                    (10, self.world.spawn), (11, self.world.spawn)]]
        episode = self.world.record(samples, seed=23, land=True)
        result = score_episode(self.world, episode, make_report(self.world, {}, 11))
        self.assertGreater(episode.collisions, 0)
        self.assertIn("collision", result["fail_reasons"])
        self.assertFalse(episode.landed_safely)

    def test_out_of_bounds(self):
        episode = self.world.record([GroundTruthSample(0, self.world.spawn, False),
                                    GroundTruthSample(1, (-1, -1, -1), False)], seed=23)
        result = score_episode(self.world, episode, make_report(self.world, {}, 1))
        self.assertIn("left_map", result["fail_reasons"])

    def test_invalid_traces(self):
        spawn = self.world.spawn
        invalid = [[GroundTruthSample(1, spawn, True)],
                   [GroundTruthSample(0, (0.7, 0.5, 1.1), True)],
                   [GroundTruthSample(0, spawn, True), GroundTruthSample(0, spawn, True)],
                   [GroundTruthSample(0, spawn, True), GroundTruthSample(-1, spawn, True)],
                   [GroundTruthSample(0, spawn, True), GroundTruthSample(float("nan"), spawn, True)],
                   [GroundTruthSample(0, spawn, True), GroundTruthSample(1, (float("inf"), 1, 1), True)]]
        for samples in invalid:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                self.world.record(samples, seed=23)

    def test_observable_target_is_trajectory_independent(self):
        before = dict(self.world.observable)
        run_attacks(self.world, 23)
        self.assertEqual(self.world.observable, before)
        self.assertEqual(dict(self.episode.observations), before)

    def test_closed_doors_exclude_unreachable_rooms(self):
        scene = copy.deepcopy(self.scene)
        added = []
        for door in scene["doors"]:
            door["state"] = "closed"
            added.extend(door["voxel_indices"])
        scene["grid"]["occupied"].extend(added)
        scene["surfaces"].append({"material": "normal", "voxel_indices": added})
        world = GroundTruthWorld(scene, self.profile)
        self.assertEqual(world.reachable_rooms, ("room_1",))
        self.assertLess(len(world.observable), len(self.world.observable))
        self.assertTrue(all(world.center(c)[0] < 2 for c in world.reachable))


class SegmentBoxDistanceTests(unittest.TestCase):
    def test_analytic_distances(self):
        box = ((0, 0, 0), (1, 1, 1))
        cases = [((-1, 0.5, 0.5), (2, 0.5, 0.5), 0),
                 ((-1, 1, 0.5), (2, 1, 0.5), 0),
                 ((-1, 2, 0.5), (2, 2, 0.5), 1),
                 ((-1, -1, -1), (-1, -1, -1), math.sqrt(3)),
                 ((-1, 1, 2), (1, -1, 2), 1),
                 ((-2, 0, 0), (0, -2, 0), math.sqrt(2))]
        for a, b, expected in cases:
            with self.subTest(a=a, b=b):
                self.assertAlmostEqual(segment_box_distance(a, b, *box), expected)
                self.assertAlmostEqual(segment_box_distance(b, a, *box), expected)


if __name__ == "__main__":
    unittest.main()
