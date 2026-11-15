"""Independent analytic geometry tests for the ideal Phase 3 environment.

These fixtures are hand-built and never consume validation/test worldgen splits.
"""
import math
from pathlib import Path
import unittest

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Fast world requires optional NumPy")

from ares.contracts import load_json, validate_scene
from ares.geometry import voxel_mesh
from ares.fast_world import FastWorldConfig, GroundTruthFastWorld


def analytic_scene(extra=(), spawn=(3.1, 3.1, 1.5)):
    shape = (30, 30, 15)
    occupied = {(x, y, z) for x in range(shape[0]) for y in range(shape[1])
                for z in range(shape[2])
                if x in (0, shape[0]-1) or y in (0, shape[1]-1) or z in (0, shape[2]-1)}
    occupied.update(extra)
    scene = {"schema_version": "ares.scene.v1", "scene_id": "analytic-phase3", "seed": 0,
             "units": "m", "frame": "right_handed_z_up",
             "license": {"spdx": "NOASSERTION", "commercial_ok": False},
             "grid": {"resolution": .2, "origin": [0, 0, 0], "shape": list(shape),
                      "occupied": [list(c) for c in sorted(occupied)]},
             "collision_mesh": voxel_mesh(sorted(occupied)),
             "rooms": [{"room_id": "r", "label": "analytic room",
                        "polygon": [[.2, .2], [5.8, .2], [5.8, 5.8], [.2, 5.8]]}],
             "doors": [], "surfaces": [{"material": "normal", "voxel_indices": [list(c) for c in sorted(occupied)]}],
             "lighting": [{"room_id": "r", "state": "lit"}], "spawn": list(spawn)}
    validate_scene(scene)
    return scene


def nominal_profile():
    return load_json(Path(__file__).resolve().parents[1] / "ares/data/configs/drone_sim_v2.json")


class FastWorldTests(unittest.TestCase):
    def world(self, extra=(), **kwargs):
        return GroundTruthFastWorld(analytic_scene(extra), nominal_profile(), **kwargs)

    def test_speed_yaw_caps_batch_independence_and_reset(self):
        world = self.world(num_envs=2)
        initial = world.observation()
        world.step(np.array([[100., 100., 0., 100., 0.], [0., 0., 0., 0., 0.]]))
        moved, still = world.observation(0), world.observation(1)
        self.assertAlmostEqual(float(np.linalg.norm(moved['velocity'])), moved['max_speed'], places=6)
        self.assertAlmostEqual(np.linalg.norm(moved['position'] - initial['position']), moved['max_speed'] * moved['dt'])
        self.assertAlmostEqual(moved['yaw'], moved['max_yaw_rate'] * moved['dt'])
        np.testing.assert_array_equal(still['position'], initial['position'])
        self.assertEqual(still['yaw'], 0.)
        world.reset()
        for index in range(2):
            reset = world.observation(index)
            np.testing.assert_array_equal(reset['position'], initial['position'])
            np.testing.assert_array_equal(reset['known_grid'], initial['known_grid'])
            self.assertEqual(reset['time_s'], 0.)

    def test_observation_is_owned_and_has_no_ground_truth_fields(self):
        world = self.world()
        before = world.observation()
        allowed = {'known_grid', 'position', 'velocity', 'yaw', 'time_s', 'home', 'resolution',
                   'origin', 'body_radius_m', 'max_speed', 'max_yaw_rate', 'dt', 'camera', 'depth_m'}
        self.assertEqual(set(before), allowed)
        self.assertTrue(set(np.unique(before['known_grid'])) <= {0, 1, 2})
        original_position = before['position'].copy()
        original_grid = before['known_grid'].copy()
        before['position'][:] = 99
        before['home'][:] = 99
        before['known_grid'][:] = 2
        before['depth_m'][:] = -99
        before['camera']['max_range_m'] = -99
        after = world.observation()
        np.testing.assert_array_equal(after['position'], original_position)
        np.testing.assert_array_equal(after['home'], original_position)
        np.testing.assert_array_equal(after['known_grid'], original_grid)
        self.assertGreater(after['camera']['max_range_m'], 0)

    def test_forward_cone_yaw_and_occluder(self):
        wall = [(20, y, z) for y in range(1, 29) for z in range(1, 14)]
        world = self.world(wall)
        first = world.observation()['known_grid']
        self.assertEqual(first[20, 15, 7], 2)
        self.assertEqual(first[21, 15, 7], 0, 'solid wall must not clear voxels behind it')
        self.assertEqual(first[10, 15, 7], 0, 'rear cell starts outside the forward cone')
        for _ in range(43):
            world.step(np.array([[0., 0., 0., 1.5, 0.]]))
        self.assertEqual(world.observation()['known_grid'][10, 15, 7], 1)

    def test_swept_sphere_hits_with_free_endpoints_and_centerline(self):
        # The point centerline y=3.1 misses this box (y>=3.2), but a .15 m
        # sphere crosses its lower edge between two free endpoints.
        world = self.world([(18, 16, 7)], config=FastWorldConfig(dt=.5))
        start = world.observation()['position'].copy()
        world.step(np.array([[2., 0., 0., 0., 0.]]))
        stopped = world.observation()
        self.assertEqual(world.ground_truth_state(0)['collisions'], 1)
        self.assertTrue(world.ground_truth_state(0)['done'])
        self.assertGreaterEqual(stopped['position'][0], start[0])
        world.step(np.array([[-2., 0., 0., 0., 0.]]))
        frozen = world.observation()
        np.testing.assert_array_equal(frozen['position'], stopped['position'])
        self.assertEqual(frozen['time_s'], stopped['time_s'])

    def test_invalid_actions_are_atomic(self):
        world = self.world(num_envs=2)
        before = world.observation()
        for actions in (np.zeros((1, 5)), np.zeros((2, 4)), np.full((2, 5), np.nan),
                        np.full((2, 5), np.inf)):
            with self.subTest(shape=actions.shape), self.assertRaises((ValueError, TypeError)):
                world.step(actions)
            self.assertEqual(world.observation()['time_s'], before['time_s'])
            np.testing.assert_array_equal(world.observation()['position'], before['position'])

    def test_seeded_worlds_are_deterministic(self):
        a, b = self.world(seed=47), self.world(seed=47)
        for action in ([.2, .1, 0, .2, 0], [0, 0, .1, -.1, 0]):
            a.step(np.array([action]))
            b.step(np.array([action]))
            for key in ('known_grid', 'position', 'velocity', 'depth_m'):
                np.testing.assert_array_equal(a.observation()[key], b.observation()[key])

    def test_timeout_freezes_completed_environment(self):
        world = self.world(config=FastWorldConfig(max_time_s=.1))
        for _ in range(2):
            world.step(np.array([[.1, 0, 0, 0, 0]]))
        before = world.observation()
        world.step(np.array([[1., 0, 0, 1., 0]]))
        after = world.observation()
        for key in ('position', 'velocity', 'known_grid', 'depth_m'):
            np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual(after['time_s'], before['time_s'])

    def test_cpu_cuda_state_and_observation_parity_on_real_gpu(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                self.skipTest('CUDA device unavailable')
            # Force a genuine allocation/kernel before calling the CUDA world.
            self.assertEqual(int(cp.sum(cp.arange(4)).get()), 6)
        except (ImportError, RuntimeError) as exc:
            self.skipTest(f'CuPy/CUDA unavailable: {exc}')
        except Exception as exc:
            self.skipTest(f'CUDA runtime unavailable: {exc}')
        cpu, cuda = self.world(num_envs=2), self.world(num_envs=2, backend='cuda')
        for action in (np.zeros((2, 5)), np.array([[.2, .1, 0, .7, 0], [0, -.1, .1, -.6, 0]])):
            cpu.step(action)
            cuda.step(action)
            for index in range(2):
                left, right = cpu.observation(index), cuda.observation(index)
                for key in ('position', 'velocity', 'depth_m'):
                    np.testing.assert_allclose(left[key], right[key], atol=2e-5, rtol=2e-5, equal_nan=True)
                np.testing.assert_array_equal(left['known_grid'], right['known_grid'])
                self.assertAlmostEqual(left['time_s'], right['time_s'])
                self.assertAlmostEqual(left['yaw'], right['yaw'], places=6)

    def test_virtual_home_landing_requires_one_second_dwell(self):
        world = self.world(config=FastWorldConfig(dt=.25))
        for _ in range(3):
            world.step(np.array([[0., 0., 0., 0., 1.]]))
            self.assertFalse(world.ground_truth_state(0)['done'])
            self.assertFalse(world.ground_truth_state(0)['landed_safely'])
        world.step(np.array([[0., 0., 0., 0., 1.]]))
        terminal = world.ground_truth_state(0)
        self.assertTrue(terminal['done'])
        self.assertTrue(terminal['landed_safely'])
        self.assertEqual(terminal['collisions'], 0)

    def test_landing_away_from_home_cannot_pass(self):
        world = self.world(config=FastWorldConfig(dt=.25))
        world.step(np.array([[2., 0., 0., 0., 0.]]))
        world.step(np.array([[0., 0., 0., 0., 1.]]))
        terminal = world.ground_truth_state(0)
        self.assertFalse(terminal['done'])
        self.assertFalse(terminal['landed_safely'])
        for _ in range(4):
            world.step(np.array([[0., 0., 0., 0., 1.]]))
        self.assertFalse(world.ground_truth_state(0)['landed_safely'])

    def test_collision_is_counted_once_and_completed_batch_member_freezes(self):
        world = self.world([(18, 16, 7)], num_envs=2, config=FastWorldConfig(dt=.5))
        world.step(np.array([[2., 0., 0., 0., 0.], [0., -.2, 0., 0., 0.]]))
        terminal = world.ground_truth_state(0)
        self.assertEqual(terminal['collisions'], 1)
        self.assertTrue(terminal['done'])
        self.assertFalse(world.ground_truth_state(1)['done'])
        for _ in range(2):
            world.step(np.array([[-2., 0., 0., 0., 0.], [0., -.2, 0., 0., 0.]]))
        current = world.ground_truth_state(0)
        self.assertEqual(current['collisions'], 1)
        self.assertEqual(current['time_s'], terminal['time_s'])
        self.assertGreater(world.ground_truth_state(1)['time_s'], current['time_s'])

    def test_ideal_sensor_ignores_phase2_material_and_lighting_dropout(self):
        scene = analytic_scene()
        other = analytic_scene()
        other['surfaces'][0]['material'] = 'glass'
        other['lighting'][0]['state'] = 'dark'
        worlds = [GroundTruthFastWorld(s, nominal_profile(), seed=3) for s in (scene, other)]
        for key in ('known_grid', 'depth_m'):
            np.testing.assert_array_equal(worlds[0].observation()[key], worlds[1].observation()[key])
        self.assertTrue(np.isfinite(worlds[0].observation()['depth_m']).any())

    def test_sphere_corner_clearance_does_not_use_inflated_box_false_positive(self):
        # Both transverse offsets are .14 m, so an inflated AABB would collide,
        # but the sphere stays sqrt(2)*.14 > .15 m away from the box corner.
        world = GroundTruthFastWorld(analytic_scene([(18, 16, 7)], spawn=(3.1, 3.06, 1.26)),
                                     nominal_profile(), config=FastWorldConfig(dt=.5))
        world.step(np.array([[2., 0., 0., 0., 0.]]))
        state = world.ground_truth_state(0)
        self.assertEqual(state['collisions'], 0)
        self.assertFalse(state['done'])
        np.testing.assert_allclose(world.observation()['position'], [4.1, 3.06, 1.26], atol=1e-9)

    def test_huge_finite_velocity_commands_preserve_direction_when_capped(self):
        world = self.world()
        world.step(np.array([[1e20, 1e20, 0., 0., 0.]]))
        velocity = world.observation()['velocity']
        np.testing.assert_allclose(velocity, [math.sqrt(2), math.sqrt(2), 0.], rtol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(velocity)), 2., places=6)

    def test_offcenter_launch_hull_exact_evidence_cpu_and_cuda(self):
        backends = ['cpu']
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                cp.zeros(1).sum().get()
                backends.append('cuda')
        except Exception:
            pass
        for backend in backends:
            with self.subTest(backend=backend):
                hidden = GroundTruthFastWorld(
                    analytic_scene([(14, 14, 7)], spawn=(3.19, 3.19, 1.5)),
                    nominal_profile(), backend=backend)
                # This voxel is behind the cone and .269 m from the actual
                # launch point, despite intersecting a sphere at the cell center.
                self.assertEqual(hidden.observation()['known_grid'][14, 14, 7], 0)
                world = GroundTruthFastWorld(analytic_scene(spawn=(3.01, 3.01, 1.41)),
                                             nominal_profile(), backend=backend)
                observed = world.observation()['known_grid']
                self.assertEqual(observed[14, 14, 6], 1, 'off-center triple diagonal belongs to launch hull')
                home = world.observation()['home']
                radius = world.observation()['body_radius_m']
                for cell in np.ndindex(observed.shape):
                    distance2 = math.fsum(max(cell[k]*.2-home[k], home[k]-(cell[k]+1)*.2, 0.)**2
                                          for k in range(3))
                    if distance2 <= radius**2+1e-12:
                        self.assertEqual(observed[cell], 1, f'physical launch hull cell {cell} must be known free')
