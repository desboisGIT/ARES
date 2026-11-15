"""Forward sensor geometry, profile randomization, noise and GPU interop."""
import unittest
import numpy as np
try:
    import torch
except ImportError:
    raise unittest.SkipTest('optional torch unavailable')
from ares.flight_sensors import FlightSensors
from tests.test_depth import room_scene, test_profile


def sampled(**overrides):
    profile = test_profile(depth_width=4, depth_height=3)
    values = {key: value['nominal'] for key, value in profile['parameters'].items()}
    values.update(overrides)
    return values


class FlightSensorTests(unittest.TestCase):
    def pose(self, batch=2, device='cpu'):
        return torch.tensor([[1.1, 1.1, 1.1]]*batch, device=device), torch.tensor([[0., 0., 0., 1.]]*batch, device=device)

    def test_forward_optical_depth_features_and_per_episode_range(self):
        sensor = FlightSensors(room_scene(), [sampled(depth_max_range_m=2.), sampled(depth_max_range_m=.5)])
        result = sensor.observe(*self.pose(), 0, noise=False)
        torch.testing.assert_close(result['raw_depth'][0], torch.full((3, 4), .9))
        self.assertTrue(bool(result['raw_depth'][1].isnan().all()))
        torch.testing.assert_close(result['depth'][0, 0], torch.full((3, 4), .45))
        self.assertEqual(result['depth'].shape, (2, 2, 3, 4))
        self.assertTrue(bool((result['depth'][1] == 0).all()))
        self.assertTrue(result['material_ids_ground_truth_only'])
        self.assertIs(sensor.renderers[0].volume, sensor.renderers[1].volume)

    def test_noise_repeatability_and_per_episode_material_holes(self):
        sensor = FlightSensors(room_scene(), [sampled(material_hole_probability=[0.]*5),
                                               sampled(material_hole_probability=[1.]*5)], seed=19)
        first = sensor.observe(*self.pose(), 3)
        second = sensor.observe(*self.pose(), 3)
        torch.testing.assert_close(first['raw_depth'], second['raw_depth'], equal_nan=True)
        self.assertTrue(bool(first['raw_depth'][1].isnan().all()))
        self.assertTrue(bool(first['raw_depth'][0].isfinite().any()))
        changed = sensor.observe(*self.pose(), 4)
        self.assertFalse(torch.equal(first['depth'][0], changed['depth'][0]))

    def test_min_range_occlusion_and_material_ids(self):
        sensor = FlightSensors(room_scene(obstacle=True), [sampled(depth_min_range_m=.35)])
        result = sensor.observe(*self.pose(1), 0, noise=False)
        # Four-pixel width has no optical-center column; use wider obstacle check
        # by comparing directly with the exact reference under the same optics.
        ref = sensor.renderers[0].render([1.1]*3, np.asarray(((0.,0.,1.),(-1.,0.,0.),(0.,-1.,0.))), noise=False)
        np.testing.assert_allclose(result['raw_depth'][0].numpy(), ref.depth_m[0], equal_nan=True)
        np.testing.assert_array_equal(result['material_ids'][0].numpy(), ref.ground_truth['material_ids'][0])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_ideal_parity_with_distinct_fovs_and_nondefault_stream(self):
        profiles = [sampled(depth_horizontal_fov_rad=.9, depth_vertical_fov_rad=.7),
                    sampled(depth_horizontal_fov_rad=1.8, depth_vertical_fov_rad=1.3)]
        cpu = FlightSensors(room_scene(obstacle=True), profiles)
        gpu = FlightSensors(room_scene(obstacle=True), profiles, 'cuda')
        quaternion = torch.tensor([[0.,0.,.258819,.9659258], [0.,.1736482,0.,.9848078]])
        expected = cpu.observe(self.pose()[0], quaternion, 4, noise=False)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            observed = gpu.observe(self.pose(device='cuda')[0], quaternion.cuda(), 4, noise=False)
            copied = observed['raw_depth'].clone()
        stream.synchronize()
        torch.testing.assert_close(expected['raw_depth'], copied.cpu(), equal_nan=True, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(expected['material_ids'], observed['material_ids'].cpu())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_randomized_noise_matches_existing_reference(self):
        from ares.depth import GroundTruthDepthRenderer, OPTICAL_TO_BODY
        profiles = [sampled(depth_noise_std_m=.02, depth_noise_quadratic_per_m2=.003,
                            depth_edge_width_px=1, depth_edge_threshold_m=.1,
                            max_range_dropout_probability=.2),
                    sampled(depth_noise_std_m=.08, depth_noise_quadratic_per_m2=.02,
                            depth_edge_width_px=2, depth_edge_threshold_m=.4,
                            max_range_dropout_probability=.8)]
        scene = room_scene(obstacle=True)
        sensor = FlightSensors(scene, profiles, 'cuda', seed=101)
        actual = sensor.observe(*self.pose(device='cuda'), 8)
        for index, profile in enumerate(sensor.profiles):
            reference = GroundTruthDepthRenderer(scene, profile, backend='cuda', seed=101)
            frame = reference.render([1.1]*3, np.asarray(OPTICAL_TO_BODY), frame_index=8,
                                     environment_ids=np.array([index], dtype=np.uint32))
            np.testing.assert_allclose(actual['raw_depth'][index].cpu().numpy(),
                                       frame.depth_m.get()[0], equal_nan=True, atol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_noise_profile_holes_and_reproducibility(self):
        profiles = [sampled(material_hole_probability=[0.]*5), sampled(material_hole_probability=[1.]*5)]
        sensor = FlightSensors(room_scene(), profiles, 'cuda', seed=15)
        result = sensor.observe(*self.pose(device='cuda'), 6)
        again = sensor.observe(*self.pose(device='cuda'), 6)
        torch.testing.assert_close(result['raw_depth'], again['raw_depth'], equal_nan=True)
        self.assertTrue(bool(result['raw_depth'][1].isnan().all()))
        self.assertTrue(bool(result['raw_depth'][0].isfinite().any()))
        self.assertFalse(result['depth'].requires_grad)


if __name__ == '__main__':
    unittest.main()
