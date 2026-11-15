"""Independent analytic sensor frames test pose estimation and map evidence."""

import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Phase 2 odometry/mapping tests require optional numpy")

from ares.mapping import OccupancyMapper
from ares.odometry import CAMERA_TO_BODY, DepthInertialOdometry, _rotation
from ares.recording import CameraIntrinsics


def render_room(camera, pose):
    """Camera inside an axis-aligned room; independently raycast each moved frame."""
    y, x = np.indices((camera.height, camera.width))
    rays = np.stack(((x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy,
                     np.ones_like(x)), axis=-1) @ pose[:3, :3].T
    limits = np.array([[-1.6, 1.6], [-1., 1.], [-1., 3.8]])
    intersections = []
    for axis in range(3):
        bound = np.where(rays[..., axis] > 0, limits[axis, 1], limits[axis, 0])
        with np.errstate(divide="ignore"):
            intersections.append((bound - pose[axis, 3]) / rays[..., axis])
    return np.min(intersections, axis=0).astype(np.float32)


class OdometryTests(unittest.TestCase):
    def setUp(self):
        self.camera = CameraIntrinsics(96, 72, 60., 60., 47.5, 35.5)

    def test_estimates_translation_and_rotation_from_new_depth(self):
        tracker = DepthInertialOdometry(self.camera)
        first = tracker.update(render_room(self.camera, np.eye(4)), 0.)
        self.assertEqual(first["status"], "initialized")
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = _rotation(np.array([.01, -.015, .02])), [.04, -.02, .03]
        result = tracker.update(render_room(self.camera, pose), .1)
        self.assertEqual(result["status"], "tracking")
        self.assertEqual(result["rank"], 6)
        self.assertLess(result["rmse_m"], .005)
        np.testing.assert_allclose(result["pose"], pose, atol=.002)
        self.assertTrue(np.isfinite(result["covariance"]).all())

    def test_low_resolution_sloped_walls_keep_observable_normals(self):
        camera = CameraIntrinsics(32, 24, 20., 20., 15.5, 11.5)
        tracker = DepthInertialOdometry(camera)
        self.assertEqual(tracker.update(render_room(camera, np.eye(4)), 0.)["status"], "initialized")
        pose = np.eye(4)
        pose[:3, 3] = [.025, -.015, .02]
        result = tracker.update(render_room(camera, pose), .1)
        self.assertEqual(result["status"], "tracking")
        np.testing.assert_allclose(result["pose"][:3, 3], pose[:3, 3], atol=.012)

    def test_single_plane_cannot_claim_stationary_tracking(self):
        tracker = DepthInertialOdometry(self.camera)
        plane = np.full((72, 96), 2., dtype=np.float32)
        for stamp in (0., .1):
            result = tracker.update(plane, stamp)
            self.assertEqual(result["status"], "lost")
            self.assertFalse(result["can_integrate"])
            self.assertEqual(result["reason"], "unobservable_geometry")
            self.assertLess(result["rank"], 6)

    def test_noisy_depth_does_not_manufacture_lateral_observability(self):
        from ares.contracts import load_json
        from ares.depth import GroundTruthDepthRenderer, OPTICAL_TO_BODY
        data = Path(__file__).resolve().parents[1] / "ares/data"
        scene = load_json(data / "scenes/three_room.json")
        profile = load_json(data / "configs/drone_sim_v2.json")
        # This actual smoke view sees front wall and horizontal planes, with no
        # independently observed lateral-facing wall. Noise must not add a DoF.
        for seed in (1, 41, 1741):
            renderer = GroundTruthDepthRenderer(scene, profile, seed=seed)
            tracker = DepthInertialOdometry(renderer.camera)
            for index in range(2):
                depth = renderer.render(scene["spawn"], OPTICAL_TO_BODY, frame_index=index).depth_m[0]
                result = tracker.update(depth, index * .1)
                self.assertEqual(result["status"], "lost")
                self.assertEqual(result["reason"], "unobservable_geometry")
                self.assertFalse(result["can_integrate"])
                self.assertLess(result["normal_span_rank"], 3)
                self.assertLess(result["normal_span_ratio"], .01)

    def test_noisy_low_resolution_motion_and_keyframe_transition(self):
        camera = CameraIntrinsics(32, 24, 20., 20., 15.5, 11.5)
        for seed in (0, 1, 2):
            tracker = DepthInertialOdometry(camera)
            rng = np.random.default_rng(seed)
            for index in range(9):
                pose = np.eye(4)
                pose[:3, 3] = [index * .02, -index * .012, index * .008]
                pose[:3, :3] = _rotation(np.array([0., 0., index * .005]))
                depth = render_room(camera, pose) + rng.normal(0., .01, (24, 32))
                estimate = tracker.update(depth.astype(np.float32), index * .1)
                self.assertTrue(estimate["can_integrate"])
                self.assertEqual(estimate["normal_span_rank"], 3)
                self.assertLess(np.linalg.norm(estimate["pose"][:3, 3] - pose[:3, 3]), .015)
            # Total motion exceeds the default 10cm keyframe promotion distance.
            self.assertGreater(np.linalg.norm(pose[:3, 3]), .15)

    def test_gyro_noise_configuration_is_explicit_and_validated(self):
        for name in ("gyro_noise_std_rps", "gyro_min_angle_std_rad"):
            for invalid in (0., -1., np.nan, np.inf, True):
                with self.assertRaises(ValueError):
                    DepthInertialOdometry(self.camera, **{name: invalid})
        tracker = DepthInertialOdometry(self.camera, gyro_noise_std_rps=.002,
                                       gyro_min_angle_std_rad=.001)
        self.assertEqual(tracker.gyro_noise_std_rps, .002)

    def test_dropout_keeps_pose_and_recovers_against_last_good_frame(self):
        tracker = DepthInertialOdometry(self.camera)
        frame = render_room(self.camera, np.eye(4))
        tracker.update(frame, 0.)
        result = tracker.update(np.full_like(frame, np.nan), .1)
        self.assertEqual(result["status"], "lost")
        self.assertFalse(result["can_integrate"])
        np.testing.assert_array_equal(result["pose"], np.eye(4))
        pose = np.eye(4)
        pose[:3, 3] = [.03, .01, -.02]
        recovered = tracker.update(render_room(self.camera, pose), .2)
        self.assertEqual(recovered["status"], "tracking")
        np.testing.assert_allclose(recovered["pose"], pose, atol=.002)

    def test_body_gyro_rotation_prior_uses_optical_extrinsics(self):
        # A body Z yaw is a camera negative Y rotation with the default mount.
        initial = np.eye(4)
        initial[:3, :3] = CAMERA_TO_BODY
        tracker = DepthInertialOdometry(self.camera, initial_pose=initial)
        tracker.update(render_room(self.camera, np.eye(4)), 0.)
        gyro, dt = np.array([0., 0., .3]), .1
        relative = np.eye(4)
        relative[:3, :3] = CAMERA_TO_BODY.T @ _rotation(gyro * dt) @ CAMERA_TO_BODY
        result = tracker.update(render_room(self.camera, relative), dt,
                                imu_accel_mps2=[0., 0., 9.81], imu_gyro_rps=gyro)
        self.assertEqual(result["status"], "tracking")
        self.assertTrue(result["gyro_prior_used"])
        self.assertFalse(result["accelerometer_translation_used"])
        np.testing.assert_allclose(result["pose"], initial @ relative, atol=.002)

    def test_bad_inputs_and_nonmonotonic_time_rejected(self):
        tracker = DepthInertialOdometry(self.camera)
        frame = render_room(self.camera, np.eye(4))
        tracker.update(frame, 1.)
        for timestamp in (1., .5, np.nan, np.inf):
            with self.assertRaises(ValueError):
                tracker.update(frame, timestamp)
        with self.assertRaises(ValueError):
            tracker.update(frame[:, :-1], 2.)
        with self.assertRaises(ValueError):
            tracker.update(frame, 2., imu_gyro_rps=[0, 0, np.nan])
        bad_pose = np.eye(4)
        bad_pose[0, 0] = 2.
        with self.assertRaises(ValueError):
            DepthInertialOdometry(self.camera, initial_pose=bad_pose)


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.camera = CameraIntrinsics(1, 1, 1., 1., 0., 0.)

    def cells(self, mapper):
        return {tuple(cell[:3]): cell[3] for cell in mapper.export()["cells"]}

    def test_ray_frees_until_hit_and_never_behind_occlusion(self):
        mapper = OccupancyMapper(resolution=1., shape=(3, 3, 8))
        pose = np.eye(4)
        pose[:3, 3] = [.5, .5, .5]
        stats = mapper.integrate(np.array([[2.2]], dtype=np.float32), self.camera, pose)
        self.assertEqual(self.cells(mapper), {(0, 0, 0): 1, (0, 0, 1): 1, (0, 0, 2): 2})
        self.assertEqual(stats["occupied_updates"], 1)
        self.assertEqual(stats["unknown_cells"], 69)

    def test_empty_map_and_nan_depth_remain_unknown(self):
        mapper = OccupancyMapper(shape=(2, 2, 2))
        self.assertEqual(mapper.export()["cells"], [])
        mapper.integrate(np.array([[np.nan]], dtype=np.float32), self.camera, np.eye(4))
        self.assertEqual(mapper.export()["cells"], [])
        self.assertEqual(mapper.counts()["unknown_cells"], 8)

    def test_log_odds_accumulates_and_can_clear_old_obstacle(self):
        mapper = OccupancyMapper(resolution=1., shape=(1, 1, 8))
        pose = np.eye(4)
        pose[:3, 3] = [.5, .5, .5]
        near, far = np.array([[2.]], dtype=np.float32), np.array([[4.]], dtype=np.float32)
        mapper.integrate(near, self.camera, pose)
        self.assertEqual(self.cells(mapper)[(0, 0, 2)], 2)
        for _ in range(3):
            mapper.integrate(far, self.camera, pose)
        self.assertEqual(self.cells(mapper)[(0, 0, 2)], 1)
        self.assertEqual(self.cells(mapper)[(0, 0, 4)], 2)

    def test_camera_outside_bounded_grid_clips_ray_correctly(self):
        mapper = OccupancyMapper(resolution=1., shape=(1, 1, 3))
        pose = np.eye(4)
        pose[:3, 3] = [.5, .5, -2.]
        mapper.integrate(np.array([[3.5]], dtype=np.float32), self.camera, pose)
        self.assertEqual(self.cells(mapper), {(0, 0, 0): 1, (0, 0, 1): 2})
        # A hit beyond the grid clears the traversed cells, but creates no clipped hit.
        other = OccupancyMapper(resolution=1., shape=(1, 1, 3))
        other.integrate(np.array([[7.]], dtype=np.float32), self.camera, pose)
        self.assertEqual(self.cells(other), {(0, 0, 0): 1, (0, 0, 1): 1, (0, 0, 2): 1})

    def test_sparse_negative_indices_export_shifted_origin(self):
        mapper = OccupancyMapper(resolution=1.)
        pose = np.eye(4)
        pose[:3, 3] = [-2.5, -.5, -.5]
        mapper.integrate(np.array([[2.]], dtype=np.float32), self.camera, pose)
        result = mapper.export()
        self.assertEqual(result["origin"], [-3., -1., -1.])
        self.assertEqual(result["shape"], [1, 1, 3])
        self.assertEqual(result["cells"], [[0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 2, 2]])

    def test_diagonal_dda_does_not_free_cells_touched_only_at_corner(self):
        mapper = OccupancyMapper(resolution=1., shape=(4, 4, 4))
        pose = np.eye(4)
        pose[:3, :3] = np.array([[1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [0, 1, 0],
                               [-1 / np.sqrt(2), 0, 1 / np.sqrt(2)]])
        pose[:3, 3] = [.5, .5, .5]
        mapper.integrate(np.array([[2.8]], dtype=np.float32), self.camera, pose)
        self.assertEqual(set(self.cells(mapper)), {(0, 0, 0), (1, 0, 1), (2, 0, 2)})
