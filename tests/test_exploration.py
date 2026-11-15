"""Observed-map policy geometry, information boundary and landing tests."""
import math
import unittest

try:
    import numpy as np
    import scipy
except ImportError:
    np = None


@unittest.skipIf(np is None, "planning extra unavailable")
class ClassicalExplorerTests(unittest.TestCase):
    def observation(self, grid=None):
        if grid is None:
            grid = np.ones((12, 12, 12), dtype=np.uint8)
            grid[[0, -1], :, :] = 2
            grid[:, [0, -1], :] = 2
            grid[:, :, [0, -1]] = 2
        return {"known_grid": grid, "position": np.array([1.1, 1.1, 1.1]),
                "velocity": np.zeros(3), "yaw": 0., "time_s": 0.,
                "home": np.array([1.1, 1.1, 1.1]), "resolution": .2,
                "origin": np.zeros(3), "body_radius_m": .15,
                "max_speed": 2., "max_yaw_rate": 1.5, "dt": .05}

    def test_inflation_uses_voxel_boxes_including_edge_neighbors(self):
        from ares.exploration import body_clear_mask
        grid = np.ones((9, 9, 9), dtype=np.uint8)
        grid[4, 4, 4] = 0
        safe = body_clear_mask(grid, .2, .15)
        self.assertEqual(np.count_nonzero(~safe[2:7, 2:7, 2:7]), 19)
        self.assertFalse(safe[5, 5, 4])
        self.assertTrue(safe[5, 5, 5])
        grid[4, 4, 4] = 2
        self.assertTrue(np.array_equal(safe, body_clear_mask(grid, .2, .15)))

    def test_exact_swept_hull_rejects_corner_cut_and_unknown(self):
        from ares.exploration import segment_is_known_free
        grid = np.ones((12, 12, 12), dtype=np.uint8)
        grid[5, 5, 5] = 2
        # Endpoints clear, but the diagonal's interior clips the obstacle.
        start, end = [1.1, .7, 1.1], [.7, 1.1, 1.1]
        self.assertTrue(segment_is_known_free(grid, start, start, .2, [0, 0, 0], .15))
        self.assertTrue(segment_is_known_free(grid, end, end, .2, [0, 0, 0], .15))
        self.assertFalse(segment_is_known_free(grid, start, end, .2, [0, 0, 0], .15))
        grid[5, 5, 5] = 0
        self.assertFalse(segment_is_known_free(grid, start, end, .2, [0, 0, 0], .15))
        self.assertFalse(segment_is_known_free(grid, [.05, 1., 1.], [.1, 1., 1.], .2, [0, 0, 0], .15))

    def test_no_scene_or_ground_truth_access_and_stationary_landing(self):
        from ares.exploration import ClassicalExplorer
        class PublicOnly(dict):
            def __getitem__(self, key):
                if key not in {"known_grid", "position", "velocity", "yaw", "time_s", "home",
                               "resolution", "origin", "body_radius_m", "max_speed", "max_yaw_rate", "dt"}:
                    raise AssertionError(f"forbidden access: {key}")
                return super().__getitem__(key)
            def get(self, key, default=None):
                if key != "camera":
                    raise AssertionError(f"forbidden access: {key}")
                return default
        observation = PublicOnly(self.observation())
        explorer = ClassicalExplorer()
        first_land = None
        for step in range(180):
            observation["time_s"] = step * observation["dt"]
            action = explorer.act(observation)
            self.assertEqual(action.shape, (5,))
            self.assertTrue(np.isfinite(action).all())
            self.assertLessEqual(abs(action[3]), observation["max_yaw_rate"])
            self.assertEqual(np.linalg.norm(action[:3]), 0.)
            observation["yaw"] += action[3] * observation["dt"]
            if action[4]:
                first_land = observation["time_s"]
                break
        self.assertIsNotNone(first_land)
        self.assertGreater(first_land, 2 * math.pi / 1.4 + 1.)
        self.assertEqual(explorer.diagnostics["return_reason"], "observed_viewpoints_exhausted")
        with self.assertRaises(TypeError):
            ClassicalExplorer({"scene": "forbidden constructor input"})

    def test_observed_map_stagnation_returns_without_true_coverage(self):
        from ares.exploration import ClassicalExplorer
        observation = self.observation()
        explorer = ClassicalExplorer()
        explorer.state = "scan"
        explorer._scan_angle = 2 * math.pi
        explorer._last_observed_count = int(np.count_nonzero(observation["known_grid"]))
        explorer._stagnant_scans = explorer.config.stagnation_scans - 1
        action = explorer.act(observation)
        self.assertEqual(explorer.diagnostics["return_reason"], "observed_map_stagnation")
        self.assertEqual(explorer.state, "landing")
        self.assertTrue(np.array_equal(action, np.zeros(5)))

    def test_reached_view_visit_uses_planned_key_despite_float_pose_drift(self):
        from ares.exploration import ClassicalExplorer
        explorer = ClassicalExplorer()
        observation = self.observation()
        planned = np.array([1.3, .7, 1.5])
        observation["position"] = planned - [0., 0., .00004]
        key = explorer._view_key(planned)
        self.assertNotEqual(key, explorer._view_key(observation["position"]))
        explorer._active_view_key = key
        explorer._scan_angle = 2 * math.pi
        self.assertIsNone(explorer._scan(observation))
        self.assertEqual(explorer._visits[key], 1)

    def test_expected_gain_stops_at_observed_wall(self):
        from ares.exploration import ClassicalExplorer
        grid = np.zeros((18, 18, 12), dtype=np.uint8)
        grid[3:9, 3:9, 3:9] = 1
        # Closed observed shell means all remaining unknown is occluded.
        grid[2:10, 2:10, 2:10] = 2
        grid[3:9, 3:9, 3:9] = 1
        gain, _ = ClassicalExplorer()._gain(grid, np.array([5, 5, 5]), .2,
                                            (math.radians(87), math.radians(58), 4.))
        self.assertEqual(gain, 0.)
        grid[9, 4:8, 4:8] = 1
        gain, _ = ClassicalExplorer()._gain(grid, np.array([5, 5, 5]), .2,
                                            (math.radians(87), math.radians(58), 4.))
        self.assertGreater(gain, 0.)

    def test_planned_motion_is_known_safe_and_speed_limited(self):
        from ares.exploration import ClassicalExplorer, ExplorationConfig, segment_is_known_free
        observation = self.observation()
        observation["known_grid"][8:, 2:10, 2:10] = 0
        explorer = ClassicalExplorer(ExplorationConfig(maximum_candidates=12))
        explorer.state = "explore"
        moved = False
        for step in range(100):
            observation["time_s"] = step * observation["dt"]
            action = explorer.act(observation)
            moved |= bool(np.linalg.norm(action[:3]) > 0)
            self.assertLessEqual(np.linalg.norm(action[:3]), observation["max_speed"] + 1e-12)
            destination = observation["position"] + action[:3] * observation["dt"]
            self.assertTrue(segment_is_known_free(observation["known_grid"], observation["position"],
                                                 destination, .2, observation["origin"], .15))
            observation["position"] = destination
            observation["velocity"] = action[:3]
            observation["yaw"] += action[3] * observation["dt"]
        self.assertTrue(moved)

    def test_room_polygons_are_observed_components_without_map_mutation(self):
        from ares.exploration import extract_rooms
        grid = np.zeros((24, 20, 15), dtype=np.uint8)
        grid[2:10, 3:15, 11] = 1
        grid[12:22, 3:15, 11] = 1
        # Door joins the lower layer but must not merge the upper footprints.
        grid[2:22, 3:15, 6] = 1
        original = grid.copy()
        rooms = extract_rooms(grid, .2, [1., -2., 0.])
        self.assertEqual(len(rooms), 2)
        self.assertTrue(np.array_equal(original, grid))
        self.assertEqual(rooms[0]["polygon"], [[1.4, -1.4], [3., -1.4], [3., 1.], [1.4, 1.]])
        for room in rooms:
            polygon = np.asarray(room["polygon"])
            signed = np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) -
                            polygon[:, 1] * np.roll(polygon[:, 0], -1)) / 2
            self.assertGreater(signed, 0.)
        self.assertEqual(extract_rooms(np.zeros_like(grid), .2, [0, 0, 0]), [])


if __name__ == "__main__":
    unittest.main()
