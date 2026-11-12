"""Boundary and mutation regressions in the evaluator/report interface."""

import copy
from importlib.resources import files
import math
import unittest
from unittest.mock import patch

from ares.cli import check_invalid_scenes
from ares.contracts import ContractError, load_json
from ares.evaluator import GroundTruthSample, GroundTruthWorld, point_in_polygon
from ares.reporting import make_report, validate_report


class BoundaryRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = files("ares") / "data"
        cls.scene = load_json(data / "scenes/three_room.json")
        cls.profile = load_json(data / "configs/drone_nominal.json")
        cls.world = GroundTruthWorld(cls.scene, cls.profile)

    def test_polygon_membership_agrees_on_all_boundaries(self):
        polygon = [[0, 0], [2, 0], [2, 2], [0, 2]]
        for point in ((0, 1), (2, 1), (1, 0), (1, 2), (0, 0), (2, 2)):
            with self.subTest(point=point):
                self.assertTrue(point_in_polygon(point, polygon))

    def test_clearance_is_not_clipped_at_initial_broad_phase(self):
        # The center of room 2 is 1 m from its closest wall, beyond the .4m
        # initial search pad. It must report the actual surface clearance.
        position = (3.2, 1.2, 1.3)
        self.assertAlmostEqual(self.world.clearance(position, position), 0.85)

    def test_episode_positions_do_not_alias_mutable_input(self):
        position = list(self.world.spawn)
        episode = self.world.record([GroundTruthSample(0, position, True)], seed=1)
        position[0] = -999
        self.assertEqual(episode.samples[0].position, self.world.spawn)

    def test_world_does_not_alias_input_scene(self):
        scene = copy.deepcopy(self.scene)
        world = GroundTruthWorld(scene, self.profile)
        scene["rooms"][0]["polygon"][0][0] = -100
        self.assertNotEqual(scene["rooms"], world.scene["rooms"])

    def test_report_numbers_reject_infinite_huge_and_boolean_values(self):
        for value in (math.nan, math.inf, 10 ** 1000, True):
            report = make_report(self.world, {}, 0)
            report["end_time_s"] = value
            with self.subTest(value=str(value)[:20]), self.assertRaises(ContractError):
                validate_report(report)

    def test_invalid_scene_gate_checks_the_intended_reason(self):
        checks = check_invalid_scenes(self.scene)
        self.assertTrue(all(item["rejected"] and item["reason_matches"] for item in checks))
        occupied = next(item for item in checks if item["case"] == "occupied_spawn")
        self.assertEqual(occupied["reason"], "spawn: must lie in free space")
        with patch("ares.cli.validate_scene", side_effect=ContractError("unrelated failure")):
            self.assertFalse(any(item["reason_matches"] for item in check_invalid_scenes(self.scene)))
