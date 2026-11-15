"""Analytic CPU raycasts and optional real-device CUDA reference parity."""

import builtins
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Phase 2 depth tests require optional numpy")

from ares.contracts import load_json, validate_profile
from ares.depth import FrameClock, GroundTruthDepthRenderer, OPTICAL_TO_BODY
from ares.importer import import_recorded_map
from ares.odometry import _rotation

DATA = Path(__file__).resolve().parents[1] / "ares/data"


def test_profile(**parameters):
    profile = load_json(DATA / "configs/drone_sim_v2.json")
    values = {"depth_width": 7, "depth_height": 5, "depth_min_range_m": .1,
              "depth_max_range_m": 5., "depth_edge_width_px": 0,
              "max_range_dropout_probability": .05, "depth_noise_std_m": .01}
    values.update(parameters)
    for name, value in values.items():
        profile["parameters"][name] = {"nominal": value, "range": [value, value]}
    validate_profile(profile)
    return profile


def room_scene(*, obstacle=False):
    recording = load_json(DATA / "recorded_maps/synthetic_enclosed_room.json")
    recording["grid"] = {"resolution": .2, "origin": [0., 0., 0.], "shape": [11, 11, 11],
                         "cells": [[x, y, z, 2 if (0 in (x, y, z) or 10 in (x, y, z) or
                                                   (obstacle and (x, y, z) == (7, 5, 5))) else 1]
                                   for x in range(11) for y in range(11) for z in range(11)]}
    recording["rooms"] = [{"room_id": "room", "label": "Analytic room",
                           "polygon": [[.2, .2], [2., .2], [2., 2.], [.2, 2.]]}]
    recording["doors"] = []
    recording["lighting"] = [{"room_id": "room", "state": "lit"}]
    recording["spawn"] = [1.1, 1.1, 1.1]
    scene = import_recorded_map(recording)
    if obstacle:
        scene["surfaces"][0]["voxel_indices"].remove([7, 5, 5])
        scene["surfaces"].append({"material": "glass", "voxel_indices": [[7, 5, 5]]})
    return scene


class DepthCPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = room_scene()
        cls.profile = test_profile()
        cls.rotation = np.asarray(OPTICAL_TO_BODY)

    def test_fronto_parallel_wall_is_optical_z_not_radial_range(self):
        renderer = GroundTruthDepthRenderer(self.scene, self.profile)
        frame = renderer.render([1.1, 1.1, 1.1], self.rotation, noise=False)
        self.assertEqual(frame.depth_m.dtype, np.float32)
        self.assertEqual(frame.depth_m.shape, (1, 5, 7))
        # Every pixel reaches the x=2 wall at optical Z=.9, even corner rays.
        np.testing.assert_allclose(frame.depth_m, .9, atol=2e-6)
        np.testing.assert_array_equal(frame.ground_truth["material_ids"], 0)
        self.assertEqual(frame.backend, "cpu")

    def test_near_obstacle_occludes_far_wall_and_reports_material(self):
        renderer = GroundTruthDepthRenderer(room_scene(obstacle=True), self.profile)
        frame = renderer.render([1.1, 1.1, 1.1], self.rotation, noise=False)
        self.assertAlmostEqual(float(frame.depth_m[0, 2, 3]), .3, places=6)
        self.assertEqual(frame.ground_truth["material_ids"][0, 2, 3], 3)
        self.assertAlmostEqual(float(frame.depth_m[0, 0, 0]), .9, places=6)

    def test_out_of_range_hits_remain_unknown_and_near_hit_still_occludes(self):
        too_far = GroundTruthDepthRenderer(self.scene, test_profile(depth_max_range_m=.5))
        result = too_far.render([1.1, 1.1, 1.1], self.rotation, noise=False)
        self.assertTrue(np.isnan(result.depth_m).all())
        self.assertTrue((result.ground_truth["material_ids"] == -1).all())
        too_near = GroundTruthDepthRenderer(room_scene(obstacle=True), test_profile(depth_min_range_m=.35))
        result = too_near.render([1.1, 1.1, 1.1], self.rotation, noise=False)
        self.assertTrue(np.isnan(result.depth_m[0, 2, 3]))
        self.assertEqual(result.ground_truth["material_ids"][0, 2, 3], -1)
        self.assertTrue(np.isfinite(result.depth_m[0, 0, 0]))

    def test_outside_grid_ray_hits_entry_wall_or_misses_away_from_grid(self):
        renderer = GroundTruthDepthRenderer(self.scene, self.profile)
        entry = renderer.render([-1., 1.1, 1.1], self.rotation, noise=False)
        self.assertAlmostEqual(float(entry.depth_m[0, 2, 3]), 1., places=6)
        away = self.rotation @ _rotation(np.array([0., np.pi, 0.]))
        missed = renderer.render([-1., 1.1, 1.1], away, noise=False)
        self.assertTrue(np.isnan(missed.depth_m).all())

    def test_frame_clock_uses_capture_fps_and_sensor_latency(self):
        renderer = GroundTruthDepthRenderer(self.scene, self.profile)
        frame = renderer.render([1.1, 1.1, 1.1], self.rotation, frame_index=9, noise=False)
        self.assertAlmostEqual(frame.timestamp_s, 9 / self.profile["parameters"]["depth_fps"]["nominal"])
        self.assertAlmostEqual(frame.available_at_s - frame.timestamp_s,
                               self.profile["parameters"]["compute_latency_s"]["nominal"])
        for args in ((0, 0), (90, -1), (np.inf, 0), (True, 0)):
            with self.assertRaises(ValueError):
                FrameClock(*args)
        for index in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                renderer.render([1.1, 1.1, 1.1], self.rotation, frame_index=index)

    def test_noise_replay_environment_identity_and_batch_splitting(self):
        renderer = GroundTruthDepthRenderer(self.scene, self.profile, seed=19)
        positions = np.array([[1.1, 1.1, 1.1], [1., 1.1, 1.1]])
        first = renderer.render(positions, self.rotation, frame_index=4, environment_ids=[41, 73]).depth_m
        replay = renderer.render(positions, self.rotation, frame_index=4, environment_ids=[41, 73]).depth_m
        np.testing.assert_array_equal(first, replay)
        split = np.concatenate([renderer.render(position, self.rotation, frame_index=4,
                                                environment_ids=[identity]).depth_m
                                for position, identity in zip(positions, (41, 73))])
        np.testing.assert_array_equal(first, split)
        another_frame = renderer.render(positions, self.rotation, frame_index=5, environment_ids=[41, 73]).depth_m
        self.assertFalse(np.array_equal(first, another_frame, equal_nan=True))
        another_identity = renderer.render(positions, self.rotation, frame_index=4, environment_ids=[42, 74]).depth_m
        self.assertFalse(np.array_equal(first, another_identity, equal_nan=True))

    def test_invalid_poses_environment_ids_and_missing_cuda_fail_explicitly(self):
        renderer = GroundTruthDepthRenderer(self.scene, self.profile)
        for rotation in (np.eye(2), np.diag([1., 1., -1.]), np.eye(3) * 2, np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                renderer.render([1., 1., 1.], rotation)
        for positions in ([], [1., np.nan, 1.], [[1., 1.]]):
            with self.assertRaises(ValueError):
                renderer.render(positions, self.rotation)
        for identities in ([1, 1], [-1, 2], [1.5, 2.], [1, 2**32], [True, False]):
            with self.assertRaises(ValueError):
                renderer.render([[1., 1., 1.], [1., 1., 1.]], self.rotation, environment_ids=identities)
        original_import = builtins.__import__
        def without_cupy(name, *args, **kwargs):
            if name == "cupy":
                raise ImportError("simulated missing CUDA dependency")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=without_cupy):
            with self.assertRaisesRegex(RuntimeError, "no silent CPU fallback"):
                GroundTruthDepthRenderer(self.scene, self.profile, backend="cuda")


class DepthCUDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise unittest.SkipTest("CUDA ray parity requires a GPU")
            cp.zeros(1).sum().item()
        except ImportError:
            raise unittest.SkipTest("CUDA ray parity requires optional CuPy")
        except (RuntimeError, OSError) as exc:
            raise unittest.SkipTest(f"CUDA ray parity unavailable: {exc}")
        cls.scene = room_scene(obstacle=True)
        cls.profile = test_profile()

    def test_seeded_ideal_depth_and_material_parity_on_real_cuda(self):
        cpu = GroundTruthDepthRenderer(self.scene, self.profile, backend="cpu")
        cuda = GroundTruthDepthRenderer(self.scene, self.profile, backend="cuda")
        rng = np.random.default_rng(991)
        positions = rng.uniform([.3, .3, .3], [1.8, 1.8, 1.8], (12, 3))
        positions = np.vstack((positions, [-1., 1.1, 1.1], [3., 1.1, 1.1]))
        rotations = np.stack([np.asarray(OPTICAL_TO_BODY) @ _rotation(rng.uniform(-.5, .5, 3))
                              for _ in positions])
        expected = cpu.render(positions, rotations, noise=False)
        actual = cuda.render(positions, rotations, noise=False)
        cuda.synchronize()
        values = cuda.to_host(actual.depth_m)
        np.testing.assert_array_equal(np.isnan(values), np.isnan(expected.depth_m))
        np.testing.assert_allclose(values, expected.depth_m, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(cuda.to_host(actual.ground_truth["material_ids"]),
                                      expected.ground_truth["material_ids"])

    def test_cuda_noise_replay_and_stable_ids_across_split_batches(self):
        renderer = GroundTruthDepthRenderer(room_scene(), self.profile, backend="cuda", seed=415)
        rotation = np.asarray(OPTICAL_TO_BODY)
        positions = np.array([[1.1, 1.1, 1.1], [1., 1.1, 1.1]])
        def render(points, frame, ids):
            return renderer.to_host(renderer.render(points, rotation, frame_index=frame,
                                                     environment_ids=ids).depth_m)
        first = render(positions, 7, [0, 1])
        np.testing.assert_array_equal(first, render(positions, 7, [0, 1]))
        split = np.concatenate((render(positions[0], 7, [0]), render(positions[1], 7, [1])))
        np.testing.assert_array_equal(first, split)
        self.assertFalse(np.array_equal(first, render(positions, 8, [0, 1]), equal_nan=True))
        self.assertFalse(np.array_equal(first, render(positions, 7, [8, 9]), equal_nan=True))

    def test_cuda_noise_does_not_alias_swapped_environment_pixel_keys(self):
        profile = test_profile(max_range_dropout_probability=0., depth_noise_quadratic_per_m2=0.)
        profile["parameters"]["material_hole_probability"] = {
            "nominal": [0.] * 5, "range": [[0., 0.]] * 5}
        renderer = GroundTruthDepthRenderer(room_scene(), profile, backend="cuda", seed=515)
        result = renderer.render([[1.1, 1.1, 1.1], [1.1, 1.1, 1.1]], OPTICAL_TO_BODY,
                                 environment_ids=[0, 1])
        pixels = renderer.to_host(result.depth_m).reshape(2, -1)
        # XOR-ing identical env/pixel hash functions formerly swapped these keys.
        self.assertNotEqual(float(pixels[0, 1]), float(pixels[1, 0]))
