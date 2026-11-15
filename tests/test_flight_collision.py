"""Continuous swept sphere tests on CPU and real CUDA."""
import unittest
try:
    import torch
except ImportError:
    raise unittest.SkipTest('optional Torch unavailable')
from ares.flight_collision import FlightCollisionChecker
from tests.test_depth import room_scene


class FlightCollisionTests(unittest.TestCase):
    def checks(self, device='cpu'):
        checker = FlightCollisionChecker(room_scene(obstacle=True),device)
        # Central obstacle is x=[1.4,1.6], y,z=[1,1.2]. Both endpoints
        # of the first path are collision free, but its middle hits the box.
        start = [[1.1,1.1,1.1], [1.1,.85,1.1], [1.1,.84,1.1], [1.1,.5,.5], [.1,1.1,1.1]]
        end = [[1.9,1.1,1.1], [1.8,.85,1.1], [1.8,.84,1.1], [1.1,.5,.5], [1.1,1.1,1.1]]
        return checker.check(torch.tensor(start,device=device),torch.tensor(end,device=device),
                             torch.tensor([.15,.15,.15,.1,.15],device=device))

    def test_midpath_contact_grazing_and_bounds(self):
        result = self.checks()
        self.assertEqual(result['collision'].tolist(), [True,True,False,False,True])
        torch.testing.assert_close(result['clearance'][:3],torch.tensor([-.15,0.,.01]),atol=2e-6,rtol=0)
        self.assertTrue(bool(result['left_bounds'][-1]))
        self.assertFalse(bool(result['left_bounds'][0]))

    def test_clearance_cap_explicit_and_large_radius_no_false_contact(self):
        scene = room_scene()
        # Large room center is farther than the .4 search cap from every wall.
        checker = FlightCollisionChecker(scene)
        result = checker.check([[1.1,1.1,1.1]],[[1.1,1.1,1.1]],[.15])
        self.assertAlmostEqual(float(result['clearance'][0]),.25,places=6)
        self.assertTrue(bool(result['clearance_is_lower_bound'][0]))
        self.assertFalse(bool(result['collision'][0]))
        result = checker.check([[1.1,1.1,1.1]],[[1.1,1.1,1.1]],[.5])
        self.assertFalse(bool(result['collision'][0]))
        self.assertEqual(float(result['clearance'][0]),0.)
        self.assertTrue(bool(result['clearance_is_lower_bound'][0]))

    def test_rejects_invalid_pose_and_radius(self):
        checker = FlightCollisionChecker(room_scene())
        for args in (([[1,1,1]],[[1,1,1]],[0.]), ([[1,1,1]],[[1,1,1]],[float('nan')]),
                     ([[1,1,1]],[[1,1]],[.1]), ([[float('inf'),1,1]],[[1,1,1]],[.1])):
            with self.assertRaises(ValueError):
                checker.check(*args)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_reference_parity_on_nondefault_stream(self):
        expected = self.checks()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            actual = self.checks('cuda')
        stream.synchronize()
        for key in expected:
            torch.testing.assert_close(expected[key],actual[key].cpu(),atol=2e-6,rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
