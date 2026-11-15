"""Exact input witnesses are conditional declarations, never flight certificates."""
from dataclasses import asdict, replace
from fractions import Fraction as F
import itertools
import math
import random
import sys
import unittest

from ares.safety_contract import (MeasurementContract, validate_contract,
                                  state_boxes, robust_free_witness)


class SafetyContractTests(unittest.TestCase):
    def contract(self, **changes):
        values = dict(position_error_m=.005, velocity_error_mps=.01,
                      world_ray_error=.003, depth_error_m=.02,
                      capture_time_s=1., expires_s=1.1,
                      supplier='external bounded capture rig',
                      provenance='fixture deterministic set guarantee; not measured RMS',
                      source_kind='hard_external', bound_basis='deterministic_guarantee',
                      voxel_model='homogeneous_static_until_expiry', depth_model='first_hit_range')
        values.update(changes)
        return MeasurementContract(**values)

    def witness(self, contract=None, **changes):
        values = dict(position=[.5, .5, .5], world_direction=[1., .2, -.1],
                      measured_depth=2., voxel_lower=[1.4, .6, .3],
                      voxel_upper=[1.6, .8, .5], optical_range=1.,
                      contract=contract or self.contract(), time_s=1.05)
        values.update(changes)
        return robust_free_witness(**values)

    def test_positive_3d_witness_encloses_all_component_corners(self):
        c = self.contract()
        result = self.witness(c)
        self.assertTrue(result.conditional_input_valid)
        self.assertTrue(result.free_witness)
        self.assertFalse(result.flight_certificate)
        self.assertFalse(asdict(result)['flight_certificate'])
        p, d, lo, hi = map(lambda v: list(map(F.from_float, v)),
                           ([.5, .5, .5], [1., .2, -.1], [1.4, .6, .3], [1.6, .8, .5]))
        ep, er = F.from_float(c.position_error_m), F.from_float(c.world_ray_error)
        # The componentwise box even includes these cube corners beyond the
        # supplier's Euclidean balls, so the universal voxel test is conservative.
        for a, b in itertools.product(itertools.product((-1, 1), repeat=3), repeat=2):
            point = [p[i]+a[i]*ep+d[i]+b[i]*er for i in range(3)]
            for i in range(3):
                self.assertLess(lo[i], point[i]); self.assertLess(point[i], hi[i])
                self.assertLessEqual(F.from_float(result.ray_point_lower[i]), point[i])
                self.assertLessEqual(point[i], F.from_float(result.ray_point_upper[i]))

    def test_first_hit_and_voxel_endpoint_contacts_are_rejected_exactly(self):
        c = self.contract(position_error_m=0., world_ray_error=0., depth_error_m=.125)
        result = self.witness(c, position=[0., 0., 0.], world_direction=[1., 0., 0.],
                              measured_depth=1., optical_range=.875,
                              voxel_lower=[.5, -.5, -.5], voxel_upper=[1., .5, .5])
        self.assertTrue(result.conditional_input_valid)
        self.assertFalse(result.free_witness)
        self.assertEqual(result.reason, 'not_strictly_before_first_hit')
        before = self.witness(c, position=[0., 0., 0.], world_direction=[1., 0., 0.],
                              measured_depth=1., optical_range=math.nextafter(.875, -math.inf),
                              voxel_lower=[.5, -.5, -.5], voxel_upper=[1., .5, .5])
        self.assertTrue(before.free_witness)
        c = replace(c, position_error_m=.125)
        for lower, upper in (([.875, -.5, -.5], [1.5, .5, .5]),
                             ([.5, -.5, -.5], [1.125, .5, .5])):
            result = self.witness(c, position=[0., 0., 0.], world_direction=[1., 0., 0.],
                                  optical_range=1., voxel_lower=lower, voxel_upper=upper)
            self.assertFalse(result.free_witness)
            self.assertEqual(result.minimum_interior_slack_lower_m, 0.)

    def test_exact_freshness_boundaries_and_capture_epoch_boxes(self):
        c = self.contract()
        first = state_boxes([1., 2., 3.], [.1, -.1, .05], c, time_s=c.capture_time_s)
        later = state_boxes([1., 2., 3.], [.1, -.1, .05], c, time_s=c.expires_s)
        self.assertTrue(later.conditional_input_valid)
        self.assertEqual(first, later)  # Freshness never propagates physical state.
        self.assertEqual(later.capture_time_s, 1.)
        self.assertFalse(later.free_witness)
        self.assertFalse(later.flight_certificate)
        for now, reason in ((math.nextafter(c.expires_s, math.inf), 'expired'),
                            (math.nextafter(c.capture_time_s, -math.inf), 'capture_in_future')):
            witness = self.witness(c, time_s=now)
            self.assertFalse(witness.conditional_input_valid)
            self.assertFalse(witness.free_witness)
            self.assertEqual(witness.reason, reason)
            box = state_boxes([1, 2, 3], [0, 0, 0], c, time_s=now)
            self.assertIsNone(box.position_lower)

    def test_probabilistic_rms_unknown_and_missing_map_models_fail_closed(self):
        for kind in ('Gaussian', 'gaussian', 'probabilistic', 'rms', 'three_sigma', 'empirical_maximum'):
            c = self.contract(source_kind=kind)
            self.assertFalse(validate_contract(c, time_s=1.05).conditional_input_valid)
            self.assertFalse(self.witness(c).free_witness)
        for basis in ('rms', 'gaussian_3sigma', 'confidence_0.999', 'finite_observed_maximum'):
            self.assertFalse(self.witness(self.contract(bound_basis=basis)).conditional_input_valid)
        for name, value in (('voxel_model', 'unspecified'), ('depth_model', 'depth_mean'),
                            ('coordinate_frame', 'body'), ('units', 'pixels')):
            result = self.witness(self.contract(**{name: value}))
            self.assertFalse(result.conditional_input_valid)
            self.assertFalse(result.free_witness)
        # State boxes do not manufacture a free-space claim without map semantics.
        c = self.contract(voxel_model='unspecified', depth_model='unspecified')
        self.assertTrue(state_boxes([0, 0, 0], [0, 0, 0], c, time_s=1.).conditional_input_valid)

    def test_roundoff_adversary_rejects_a_naively_accepted_boundary(self):
        big = float(2**53)
        c = self.contract(position_error_m=1., world_ray_error=0., depth_error_m=0.)
        lower, upper = [big+2., -2., -2.], [big+8., 2., 2.]
        # Binary64 center rounds upward, and subtracting the radius rounds up
        # again: the naive lower endpoint lies strictly inside the voxel.
        naive_center = big+3.*1.
        self.assertGreater(naive_center-1., lower[0])
        self.assertLess(naive_center+1., upper[0])
        result = self.witness(c, position=[big, 0., 0.], world_direction=[1., 0., 0.],
                              measured_depth=10., optical_range=3., voxel_lower=lower, voxel_upper=upper)
        self.assertTrue(result.conditional_input_valid)
        self.assertFalse(result.free_witness)
        self.assertEqual(result.minimum_interior_slack_lower_m, 0.)
        self.assertEqual(result.ray_point_lower[0], big+2.)

    def test_state_and_ray_float_endpoints_round_outward_against_exact_reference(self):
        c = self.contract(position_error_m=1., velocity_error_mps=.1)
        big = float(2**53)
        box = state_boxes([big, 0., -.1], [.1, -.1, 0.], c, time_s=1.)
        self.assertEqual(box.position_upper[0], math.nextafter(big, math.inf))
        for values, error, lower, upper in (([big, 0., -.1], c.position_error_m, box.position_lower, box.position_upper),
                                            ([.1, -.1, 0.], c.velocity_error_mps, box.velocity_lower, box.velocity_upper)):
            for p, lo, hi in zip(values, lower, upper):
                self.assertLessEqual(F.from_float(lo), F.from_float(p)-F.from_float(error))
                self.assertGreaterEqual(F.from_float(hi), F.from_float(p)+F.from_float(error))
        rng = random.Random(712)
        for _ in range(150):
            p = [rng.uniform(-4, 4) for _ in range(3)]
            d = [rng.uniform(-2, 2) for _ in range(3)]
            s = rng.uniform(.1, 2.)
            ep, er = rng.uniform(0, .1), rng.uniform(0, .05)
            c = self.contract(position_error_m=ep, world_ray_error=er, depth_error_m=0.)
            radius = F.from_float(ep)+F.from_float(s)*F.from_float(er)
            center = [F.from_float(a)+F.from_float(s)*F.from_float(b) for a, b in zip(p, d)]
            # Mix interior, exact-near-boundary and excluded queries, then compare
            # every decision with an independent direct rational inequality.
            lower = [float(x)-rng.uniform(0, .3) for x in center]
            upper = [float(x)+rng.uniform(0, .3) for x in center]
            result = self.witness(c, position=p, world_direction=d, measured_depth=3.,
                                  optical_range=s, voxel_lower=lower, voxel_upper=upper)
            expected = all(F.from_float(lo)<x-radius and x+radius<F.from_float(hi)
                           for x, lo, hi in zip(center, lower, upper))
            self.assertEqual(result.free_witness, expected)
            for x, lo, hi in zip(center, result.ray_point_lower, result.ray_point_upper):
                self.assertLessEqual(F.from_float(lo), x-radius)
                self.assertGreaterEqual(F.from_float(hi), x+radius)

    def test_nonfinite_invalid_and_unrepresentable_inputs_never_certify(self):
        for field in ('position_error_m', 'velocity_error_mps', 'world_ray_error', 'depth_error_m',
                      'capture_time_s', 'expires_s'):
            for invalid in (math.inf, math.nan):
                with self.assertRaises(ValueError):
                    self.contract(**{field: invalid})
        for field in ('position_error_m', 'velocity_error_mps', 'world_ray_error', 'depth_error_m'):
            with self.assertRaises(ValueError):self.contract(**{field: -.1})
        for field in ('supplier', 'provenance'):
            with self.assertRaises(ValueError):self.contract(**{field: ''})
        with self.assertRaises(ValueError):self.contract(expires_s=.9)
        with self.assertRaises(ValueError):self.witness(position=[math.nan, 0, 0])
        with self.assertRaises(ValueError):self.witness(voxel_lower=[1, 1, 1], voxel_upper=[1, 2, 2])
        with self.assertRaises(ValueError):validate_contract(self.contract(), time_s=math.nan)
        for depth in (math.inf, math.nan, -1.):
            result = self.witness(measured_depth=depth)
            self.assertFalse(result.conditional_input_valid)
            self.assertFalse(result.free_witness)
        c = self.contract(position_error_m=sys.float_info.max)
        result = state_boxes([sys.float_info.max, 0, 0], [0, 0, 0], c, time_s=1.)
        self.assertFalse(result.conditional_input_valid)
        self.assertEqual(result.reason, 'nonfinite_outward_bounds')
        self.assertFalse(result.flight_certificate)


if __name__ == '__main__':
    unittest.main()
