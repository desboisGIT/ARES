"""Analytic turn budgets and differentiability of public waypoint guidance."""
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest('optional Phase 4 torch unavailable')

from ares.flight_guidance import GuidanceConfig, guidance_velocity


class FlightGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.config = GuidanceConfig()

    def reference(self, distance, outgoing, speed=1.5):
        p = torch.tensor([[-distance, 0., 0.]], dtype=torch.float64)
        waypoint = torch.zeros_like(p)
        next_waypoint = torch.tensor([outgoing], dtype=torch.float64)
        return guidance_velocity(p, waypoint, next_waypoint, speed, self.config)[0]

    def test_straight_continuation_preserves_cruise_before_handoff(self):
        for distance in (.01, .35, .45, 1., 5.):
            torch.testing.assert_close(self.reference(distance, [2., 0., 0.]),
                                       torch.tensor([1.5, 0., 0.], dtype=torch.float64))

    def test_speed_fraction_reserves_tracking_margin_without_changing_limit(self):
        p=torch.tensor([[-4.,0.,0.]])
        waypoint=torch.zeros_like(p)
        next_waypoint=torch.tensor([[3.,0.,0.]])
        limit=torch.tensor([[1.5]])
        reduced=guidance_velocity(p,waypoint,next_waypoint,limit,
                                  GuidanceConfig(speed_fraction=.95))
        torch.testing.assert_close(reduced,torch.tensor([[1.425,0.,0.]]))
        torch.testing.assert_close(limit,torch.tensor([[1.5]]))
        for fraction in (0.,-1.,1.01,float('nan'),float('inf')):
            with self.subTest(fraction=fraction),self.assertRaises(ValueError):
                GuidanceConfig(speed_fraction=fraction)

    def test_right_angle_and_reversal_slow_before_waypoint_switch(self):
        for outgoing, handoff in (([0., 2., 0.], .20), ([-2., 0., 0.], .10)):
            speeds = [float(self.reference(distance, outgoing).norm())
                      for distance in (.10, .30, .60, 1., 4.)]
            self.assertLessEqual(speeds[0], handoff+1e-6)
            self.assertEqual(speeds, sorted(speeds))
            self.assertLess(speeds[2], 1.5)
            self.assertAlmostEqual(speeds[-1], 1.5)
            # Positive handoff speed crosses the mission acceptance boundary.
            self.assertGreater(speeds[0], 0.)
            stopping = speeds[0]*self.config.reaction_time+speeds[0]**2/(2*self.config.minimum_deceleration)
            self.assertLess(stopping, .13)

    def test_braking_cap_satisfies_reaction_and_deceleration_budget(self):
        c = self.config
        for outgoing, handoff in (([0., 2., 0.], .20), ([-2., 0., 0.], .10)):
            for distance in (.55, .7, 1., 1.5, 2., 4.):
                velocity = self.reference(distance, outgoing)
                speed = float(velocity.norm())
                available = max(distance-c.waypoint_radius-c.lookahead_distance, 0.)
                if speed > handoff+1e-6:
                    consumed = speed*c.reaction_time+(speed**2-handoff**2)/(2*c.minimum_deceleration)
                    self.assertLessEqual(consumed, available+1e-6)
                torch.testing.assert_close(velocity[1:], torch.zeros(2, dtype=torch.float64))

    def test_endpoint_tends_to_zero_and_obeys_speed_limit(self):
        speeds = [float(self.reference(distance, [0., 0., 0.]).norm())
                  for distance in (0., .001, .01, .1, .35, 1., 5.)]
        self.assertEqual(speeds, sorted(speeds))
        self.assertEqual(speeds[0], 0.)
        self.assertLess(speeds[1], .003)
        self.assertLess(speeds[4], .8)
        self.assertEqual(speeds[-1], 1.5)
        self.assertAlmostEqual(float(self.reference(5., [0., 2., 0.], speed=.1).norm()), .1)

    def test_gradients_finite_at_straight_reversal_vertical_and_coincident_targets(self):
        cases = (([-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]),
                 ([-.7, 0., 0.], [0., 0., 0.], [-1., 0., 0.]),
                 ([0., 0., -1.], [0., 0., 0.], [0., 0., 1.]),
                 ([0., 0., 1.], [0., 0., 0.], [0., 0., -1.]),
                 ([0., 0., 0.], [0., 0., 0.], [0., 0., 0.]),
                 ([-.7, 0., 0.], [0., 0., 0.], [0., 0., 0.]))
        for points in cases:
            tensors = [torch.tensor([point], dtype=torch.float64, requires_grad=True) for point in points]
            limit = torch.tensor([[1.5]], dtype=torch.float64, requires_grad=True)
            velocity = guidance_velocity(*tensors, limit)
            (velocity.square().sum()+velocity.sum()).backward()
            self.assertTrue(bool(torch.isfinite(velocity).all()))
            for tensor in (*tensors, limit):
                self.assertIsNotNone(tensor.grad)
                self.assertTrue(bool(torch.isfinite(tensor.grad).all()))

    def test_tensor_inputs_repeat_exactly_and_speed_limits_broadcast(self):
        p = torch.tensor([[-1., 0., 0.], [-1., 0., 0.]])
        wp1 = torch.zeros_like(p)
        wp2 = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
        expected = torch.tensor([[.5, 0., 0.], [1., 0., 0.]])
        for limit in (torch.tensor([.5, 1.]), torch.tensor([[.5], [1.]])):
            before = [value.clone() for value in (p, wp1, wp2, limit)]
            actual = guidance_velocity(p, wp1, wp2, limit)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual, guidance_velocity(p, wp1, wp2, limit), atol=0, rtol=0)
            for value, old in zip((p, wp1, wp2, limit), before):
                torch.testing.assert_close(value, old, atol=0, rtol=0)

    def test_invalid_developer_config_is_rejected(self):
        for config in ({'minimum_deceleration': 0.}, {'reaction_time': float('nan')},
                       {'reversal_speed': .3, 'corner_speed': .2}):
            with self.assertRaises(ValueError):
                GuidanceConfig(**config)


if __name__ == '__main__':
    unittest.main()
