"""Attacks against scorer authority and room geometry, using real trusted traces."""

import copy
from importlib.resources import files
import unittest

from ares.contracts import ContractError, load_json
from ares.evaluator import GroundTruthWorld
from ares.reporting import make_report
from ares.scoring import score_episode


class AdversarialScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = files("ares") / "data"
        cls.scene = load_json(data / "scenes/three_room.json")
        cls.profile = load_json(data / "configs/drone_nominal.json")
        cls.world = GroundTruthWorld(cls.scene, cls.profile)
        cls.episode = cls.world.witness(seed=17)
        cls.report = make_report(cls.world, dict(cls.episode.observations),
                                 cls.episode.samples[-1].timestamp_s, cls.scene["rooms"])

    def assert_rejected(self, world, episode, report):
        try:
            result = score_episode(world, episode, report)
        except (ContractError, ValueError):
            return
        self.assertFalse(result["passed"], result)

    def test_room_names_are_policy_owned(self):
        report = copy.deepcopy(self.report)
        for i, room in enumerate(report["rooms"]):
            room["room_id"] = f"discovered_{i}"
        result = score_episode(self.world, self.episode, report)
        self.assertTrue(result["passed"], result["fail_reasons"])
        self.assertEqual(result["rooms_found"], ["room_1", "room_2", "room_3"])

    def test_extra_duplicate_geometry_cannot_count_as_another_room(self):
        report = copy.deepcopy(self.report)
        room = copy.deepcopy(report["rooms"][0])
        room["room_id"] = "fourth_room"
        report["rooms"].append(room)
        self.assert_rejected(self.world, self.episode, report)

    def test_room_cannot_claim_exterior_area_hidden_between_raster_centers(self):
        report = copy.deepcopy(self.report)
        # Adds a simple, huge exterior wing via a narrow connector between
        # y=0.7 and y=0.9 cell centers. Its in-grid raster mask is unchanged.
        report["rooms"][0]["polygon"] = [
            [0.2,0.2],[2.0,0.2],[2.0,2.2],[0.2,2.2],
            [0.2,0.82],[-100.0,0.82],[-100.0,0.80],[0.2,0.80]]
        self.assert_rejected(self.world, self.episode, report)

    def test_self_intersecting_room_cannot_hide_a_zero_width_loop(self):
        report = copy.deepcopy(self.report)
        # A repeated point creates an invalid spike while preserving the mask.
        report["rooms"][0]["polygon"] = [
            [0.2,0.2],[2.0,0.2],[2.0,2.2],[0.2,2.2],
            [0.2,0.8],[0.4,0.8],[0.2,0.8]]
        self.assert_rejected(self.world, self.episode, report)

    def test_evidence_from_other_scene_identity_is_rejected(self):
        scene = copy.deepcopy(self.scene)
        scene["scene_id"] = "other_scene"
        world = GroundTruthWorld(scene, self.profile)
        self.assert_rejected(world, self.episode, self.report)

    def test_evidence_from_other_profile_identity_is_rejected(self):
        profile = copy.deepcopy(self.profile)
        profile["profile_id"] = "other_profile"
        world = GroundTruthWorld(self.scene, profile)
        self.assert_rejected(world, self.episode, self.report)


if __name__ == "__main__":
    unittest.main()
