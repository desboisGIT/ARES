"""Adversarial tests of Phase 3 observed evidence and scorer identity binding."""
import copy
from dataclasses import FrozenInstanceError
import unittest

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('Exploration scoring requires optional NumPy')

from ares.fast_world import FastWorldConfig, GroundTruthFastWorld
from ares.exploration_scoring import build_map_report, score_exploration
from tests.test_fast_world import analytic_scene, nominal_profile


class ExplorationScoringTests(unittest.TestCase):
    def setUp(self):
        self.world = GroundTruthFastWorld(analytic_scene(), nominal_profile())
        self.world.step(np.zeros((1, 5)))
        self.episode = self.world.snapshot_for_scoring()
        self.report = build_map_report(self.world.observation())

    def score(self, report=None):
        return score_exploration(self.world, self.episode, self.report if report is None else report)

    def test_versioned_result_preserves_old_scorer(self):
        from ares import SCORER_VERSION
        result = self.score()
        self.assertEqual(SCORER_VERSION, 'ares.scorer.v1')
        self.assertEqual(result['schema_version'], 'ares.exploration_result.v1')
        self.assertEqual(result['scorer_version'], 'ares.exploration_scorer.v1')
        self.assertFalse(result['passed'])
        self.assertIn('unsafe_landing', result['fail_reasons'])

    def test_report_builder_contains_only_public_observed_evidence(self):
        observed = self.world.observation()['known_grid']
        expected = {tuple(int(v) for v in cell): int(observed[tuple(cell)]) for cell in np.argwhere(observed)}
        self.assertEqual({tuple(c[:3]): c[3] for c in self.report['voxel_map']['cells']}, expected)
        self.assertEqual(self.report['rooms'], [])
        self.assertEqual(self.report['end_time_s'], self.episode.time_s)
        self.assertEqual(self.score()['unsupported_map_cells'], 0)

    def test_unseen_map_claim_is_rejected_even_when_ground_truth_correct(self):
        known = self.world.observation()['known_grid']
        unseen = next(tuple(int(v) for v in cell) for cell in np.argwhere(known == 0)
                      if tuple(cell) not in set(map(tuple, self.world.scene['grid']['occupied'])))
        report = copy.deepcopy(self.report)
        report['voxel_map']['cells'].append([*unseen, 1])
        result = self.score(report)
        self.assertEqual(result['unsupported_map_cells'], 1)
        self.assertIn('unsupported_map_claims', result['fail_reasons'])
        self.assertFalse(result['passed'])

    def test_unseen_floor_claim_cannot_borrow_ground_truth(self):
        report = copy.deepcopy(self.report)
        floor = report['floor_plan']['cells']
        index = floor.index(0)
        floor[index] = 1
        self.assertIn('unsupported_floor_plan_claims', self.score(report)['fail_reasons'])

    def test_time_lie_is_rejected(self):
        report = copy.deepcopy(self.report)
        report['end_time_s'] += 1
        self.assertIn('report_time_mismatch', self.score(report)['fail_reasons'])

    def test_duplicate_and_invalid_map_labels_fail_contract(self):
        for change in ('duplicate', 'label'):
            report = copy.deepcopy(self.report)
            if change == 'duplicate':
                report['voxel_map']['cells'].append(report['voxel_map']['cells'][0].copy())
            else:
                report['voxel_map']['cells'][0][3] = 3
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.score(report)

    def test_snapshot_is_immutable_and_detached_from_future_world(self):
        original = self.episode.observations
        with self.assertRaises(FrozenInstanceError):
            self.episode.time_s = 99
        with self.assertRaises(TypeError):
            self.episode.observations[0][0][0] = 99
        self.world.step(np.array([[0, 0, 0, 1.5, 0]]))
        self.assertEqual(self.episode.observations, original)
        self.assertAlmostEqual(self.episode.time_s, .05)

    def test_identity_binding_includes_config_camera_and_profile(self):
        scene, profile = analytic_scene(), nominal_profile()
        changed = copy.deepcopy(profile)
        changed['parameters']['depth_horizontal_fov_rad']['nominal'] *= .99
        worlds = [GroundTruthFastWorld(scene, profile, config=FastWorldConfig(dt=.1)),
                  GroundTruthFastWorld(scene, profile, config=FastWorldConfig(camera_width=16)),
                  GroundTruthFastWorld(scene, changed)]
        for other in worlds:
            with self.subTest(digest=other.digest), self.assertRaises(ValueError):
                score_exploration(other, self.episode, self.report)

    def test_room_matching_uses_geometry_and_evidence_not_policy_ids(self):
        report = copy.deepcopy(self.report)
        report['rooms'] = [{'room_id': 'policy-room-847', 'polygon': [[.2, .2], [5.8, .2], [5.8, 5.8], [.2, 5.8]]}]
        result = self.score(report)
        self.assertEqual(result['rooms_found'], ['r'])
        self.assertEqual(result['rooms_required'], ['r'])
        self.assertNotIn('rooms_missing', result['fail_reasons'])
        self.assertNotIn('unmatched_reported_rooms', result['fail_reasons'])

    def test_target_denominator_is_independent_of_policy_path_and_claims(self):
        from ares.exploration_scoring import GroundTruthExplorationReference
        original = GroundTruthExplorationReference(self.world).observable
        self.world.step(np.array([[.2, .1, 0, 1.5, 0]]))
        moved = GroundTruthExplorationReference(self.world).observable
        self.assertEqual(moved, original)
        report = copy.deepcopy(self.report)
        report['voxel_map']['cells'] = []
        self.score(report)
        self.assertEqual(self.world._scoring_reference.observable, original)
