"""Analytic adversarial fixtures; only observed maps enter the shield."""
from dataclasses import replace
import math
import unittest

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Safety shield requires optional NumPy")

from ares.safety import SafetyShield, ShieldConfig


class SafetyShieldTests(unittest.TestCase):
    origin = np.array([-5., -3., -2.])
    resolution = .1

    def setUp(self):
        self.grid = np.ones((100, 60, 40), dtype=np.uint8)
        self.config = ShieldConfig(body_radius_m=.1, position_uncertainty_m=0,
                                   velocity_uncertainty_mps=0, latency_s=.1,
                                   control_interval_s=.1, max_acceleration_mps2=2,
                                   min_braking_acceleration_mps2=1, max_speed_mps=5)

    def run_shield(self, position=(0, 0, 0), velocity=(1, 0, 0), acceleration=(2, 0, 0), **kwargs):
        config = kwargs.pop('config', self.config)
        return SafetyShield(config).filter(self.grid, position, velocity, acceleration,
                                          origin=self.origin, voxel_size=self.resolution, **kwargs)

    def cell(self, xyz):
        return tuple(np.floor((np.array(xyz)-self.origin)/self.resolution).astype(int))

    def wall(self, x, value=2):
        self.grid[self.cell((x, 0, 0))[0]:, :, :] = value

    def test_open_space_accepts_acceleration_and_returns_owned_command(self):
        command = np.array([2., 0, 0])
        result = self.run_shield(acceleration=command)
        self.assertTrue(result.accepted)
        self.assertTrue(result.certified)
        self.assertTrue(result.state_viable)
        np.testing.assert_array_equal(result.safe_acceleration, command)
        command[:] = 99
        np.testing.assert_array_equal(result.safe_acceleration, [2, 0, 0])

    def test_unknown_slab_blocks_exactly_like_occupied_wall(self):
        for value in (0, 2):
            self.grid[:] = 1
            # Only one 10 cm slab, with known free cells on BOTH sides.
            self.grid[self.cell((1., 0, 0))[0], :, :] = value
            result = self.run_shield()
            self.assertFalse(result.accepted)
            self.assertTrue(result.certified)
            self.assertTrue(result.emergency)
            self.assertLess(result.safe_acceleration[0], 0)

    def test_discrete_candidate_reduces_acceleration_before_braking(self):
        self.wall(1.5)
        result = self.run_shield()
        self.assertEqual(result.reason, 'limited')
        self.assertGreater(result.candidate_fraction, 0)
        self.assertLess(result.candidate_fraction, 1)
        self.assertTrue(result.certified)

    def test_already_unavoidable_wall_does_not_claim_hover_is_safe(self):
        self.wall(.2)
        result = self.run_shield()
        # Even immediate perfect -1 m/s² braking from 1 m/s needs .5 m,
        # whereas the physical body has only .1 m clearance to this wall.
        self.assertFalse(result.certified)
        self.assertFalse(result.state_viable)
        self.assertTrue(result.already_unsafe)
        self.assertEqual(result.reason, 'no_braking_certificate')
        np.testing.assert_array_equal(result.safe_acceleration, [-1., 0, 0])

    def test_delay_and_motor_lag_grow_stopping_distance_analytically(self):
        fast = replace(self.config, latency_s=0)
        slow = replace(self.config, latency_s=.2)
        fast_result = self.run_shield(config=fast)
        slow_result = self.run_shield(config=slow)
        self.assertAlmostEqual(fast_result.stopping_distance_m, .5)
        expected = 1*.2 + .5*2*.2**2 + (1+2*.2)**2/(2*1)
        self.assertAlmostEqual(slow_result.stopping_distance_m, expected)
        self.wall(.9)
        self.assertTrue(self.run_shield(config=fast).state_viable)
        self.assertFalse(self.run_shield(config=slow).state_viable)

    def test_side_obstacle_inside_worst_case_latency_tube_is_blocked(self):
        self.config = replace(self.config, latency_s=.3, max_acceleration_mps2=4)
        self.grid[self.cell((.4, .2, 0))] = 2
        result = self.run_shield()
        self.assertFalse(result.state_viable)
        self.assertFalse(result.certified)

    def test_physical_radius_detects_diagonal_grazing(self):
        self.config = replace(self.config, body_radius_m=.16, latency_s=0)
        # The diagonal centerline misses this voxel; its sphere grazes the corner.
        self.grid[self.cell((.4, .6, 0))] = 2
        result = self.run_shield(velocity=(1, 1, 0), acceleration=(1, 1, 0))
        self.assertFalse(result.state_viable)

    def test_vertical_wall_and_out_of_map_are_not_projected_to_2d(self):
        self.grid[:, :, self.cell((0, 0, .6))[2]] = 0
        result = self.run_shield(velocity=(0, 0, 1), acceleration=(0, 0, 1))
        self.assertFalse(result.certified)
        self.grid[:] = 1
        self.assertFalse(self.run_shield(position=(4.8, 0, 0)).certified)

    def test_uncertainty_inflation_blocks_side_cell(self):
        self.grid[self.cell((.2, .3, 0))] = 2
        exact = self.run_shield(config=replace(self.config, latency_s=0))
        uncertain = self.run_shield(config=replace(self.config, latency_s=0, position_uncertainty_m=.25))
        self.assertTrue(exact.state_viable)
        self.assertFalse(uncertain.state_viable)

    def test_speed_limit_and_acceleration_limit(self):
        result = self.run_shield(acceleration=(100, 0, 0))
        self.assertFalse(result.accepted)
        self.assertLessEqual(np.linalg.norm(result.safe_acceleration), self.config.max_acceleration_mps2)
        result = self.run_shield(velocity=(5, 0, 0))
        self.assertFalse(result.certified)
        self.assertLess(result.safe_acceleration[0], 0)

    def test_invalid_controller_assumptions_preclude_certificate(self):
        result = self.run_shield(assumptions_valid=False)
        self.assertTrue(result.state_viable)
        self.assertFalse(result.certified)
        self.assertTrue(result.emergency)
        self.assertEqual(result.reason, 'assumptions_invalid')

    def test_stationary_body_in_unknown_cell_is_uncertified(self):
        self.grid[self.cell((0, 0, 0))] = 0
        result = self.run_shield(velocity=(0, 0, 0), acceleration=(0, 0, 0))
        self.assertFalse(result.certified)

    def test_contact_with_lower_voxel_face_is_blocked(self):
        # Explicitly exercise a blocked cell just BELOW the broadphase lower
        # boundary, where floor-only enumeration would omit exact contact.
        self.grid[:] = 1
        self.grid[49, 29, 19] = 2
        result = self.run_shield(position=(.1, .1, .1), velocity=(0, 0, 0),
                                 acceleration=(0, 0, 0), config=replace(self.config, latency_s=0))
        self.assertFalse(result.certified)

    def test_independent_worst_case_delay_braking_trajectories(self):
        # Continuous analytic paths with arbitrary constant acceleration during
        # delay, followed by a physical minimum-rate straight stop. A certified
        # envelope must contain every sampled trajectory, including sideways lag.
        self.config = replace(self.config, latency_s=.2)
        result = self.run_shield()
        self.assertTrue(result.certified)
        shield = SafetyShield(self.config)
        segments, _, _ = shield._envelope(np.zeros(3), np.array([1., 0, 0]))
        rng = np.random.default_rng(71)
        for _ in range(80):
            acceleration = rng.normal(size=3)
            acceleration *= rng.uniform(0, 2)/np.linalg.norm(acceleration)
            delay = self.config.latency_s
            q = np.array([delay, 0, 0]) + .5*acceleration*delay**2
            u = np.array([1., 0, 0]) + acceleration*delay
            speed = np.linalg.norm(u)
            for t in np.linspace(0, speed, 31):
                point = q + u*(t-.5*t*t/speed)
                # Distance to the geometric center segments, independent of
                # voxel slab code. Spheres around these segments must enclose it.
                covered = False
                for segment in segments:
                    a, b = np.array(segment.start), np.array(segment.end)
                    delta = b-a
                    f = np.clip(np.dot(point-a, delta)/max(np.dot(delta, delta), 1e-30), 0, 1)
                    if np.linalg.norm(point-(a+f*delta)) <= segment.radius_m-self.config.body_radius_m+1e-12:
                        covered = True
                self.assertTrue(covered)

    def test_terminal_defaults_preserve_legacy_distance_segments_and_brake(self):
        from ares.safety import SAFETY_VERSION
        self.assertEqual(SAFETY_VERSION, 'ares.safety.v2')
        explicit = replace(self.config, terminal_speed_mps=0., terminal_stop_radius_m=0.)
        default = self.run_shield()
        actual = self.run_shield(config=explicit)
        self.assertEqual(default.checked_segments, actual.checked_segments)
        self.assertEqual(default.stopping_distance_m, actual.stopping_distance_m)
        np.testing.assert_array_equal(default.safe_acceleration, actual.safe_acceleration)
        np.testing.assert_array_equal(SafetyShield(explicit)._emergency_acceleration(
            np.array([.001, 0., 0.])), [-1., 0., 0.])

    def test_terminal_tail_in_unknown_voxel_invalidates_otherwise_free_stop(self):
        base = replace(self.config, latency_s=0.)
        terminal = replace(base, terminal_speed_mps=.2, terminal_stop_radius_m=.3)
        # The active straight stop ends before x=.5. A permitted terminal
        # excursion from the .48 entry can reach this unknown x=.7 slab.
        self.grid[self.cell((.7, 0., 0.))[0], :, :] = 0
        legacy = self.run_shield(config=base)
        result = self.run_shield(config=terminal)
        self.assertTrue(legacy.state_viable)
        self.assertFalse(result.state_viable)
        self.assertFalse(result.certified)
        self.assertTrue(result.emergency)
        self.assertAlmostEqual(result.stopping_distance_m, legacy.stopping_distance_m + .3)

    def test_terminal_expansion_covers_active_prefix_and_reversing_hold(self):
        c = replace(self.config, latency_s=0., terminal_speed_mps=.2,
                    terminal_stop_radius_m=.3)
        p, v = np.zeros(3), np.array([1., .6, .2])
        shield = SafetyShield(c)
        segments, _, distance = shield._envelope(p, v, np.array([.1, -.1, 0.]))
        old_segments, _, old_distance = SafetyShield(replace(c, terminal_speed_mps=0.,
            terminal_stop_radius_m=0.))._envelope(p, v, np.array([.1, -.1, 0.]))
        for actual, old in zip(segments, old_segments):
            self.assertEqual(actual.start, old.start)
            self.assertEqual(actual.end, old.end)
            self.assertAlmostEqual(actual.radius_m, old.radius_m + .3)
        self.assertAlmostEqual(distance, old_distance + .3)
        # Independently form a straight active prefix and a bounded terminal
        # curve. Terminal velocities reverse and need not remain on the ray.
        segments, _, _ = shield._envelope(p, v)
        speed = np.linalg.norm(v)
        active_duration = (speed-c.terminal_speed_mps)/c.min_braking_acceleration_mps2
        points = [v*(t-.5*c.min_braking_acceleration_mps2*t*t/speed)
                  for t in np.linspace(0, active_duration, 51)]
        entry = points[-1].copy()
        tail = np.array([entry + c.terminal_stop_radius_m / math.sqrt(3) * np.array(
            [math.sin(t), math.sin(2*t), math.sin(3*t)])
            for t in np.linspace(0, 4*math.pi, 201)])
        self.assertTrue(np.all(np.linalg.norm(tail-entry, axis=1) <= c.terminal_stop_radius_m))
        self.assertTrue(np.any(np.diff(tail[:, 0]) > 0))
        self.assertTrue(np.any(np.diff(tail[:, 0]) < 0))
        for point in [*points, *tail]:
            covered = False
            for segment in segments:
                a, b = np.array(segment.start), np.array(segment.end)
                delta = b-a
                f = np.clip(np.dot(point-a, delta)/max(np.dot(delta, delta), 1e-30), 0, 1)
                covered |= np.linalg.norm(point-a-f*delta) <= segment.radius_m-c.body_radius_m+1e-12
            self.assertTrue(covered)

    def test_terminal_emergency_command_is_smooth_and_cannot_discretely_reverse(self):
        c = replace(self.config, terminal_speed_mps=.2, terminal_stop_radius_m=.3)
        shield = SafetyShield(c)
        for speed in (0., 1e-6, .01, .1, .2, 1., 3.):
            v = np.array([speed, 0., 0.])
            a = shield._emergency_acceleration(v)
            self.assertLessEqual(np.linalg.norm(a), c.min_braking_acceleration_mps2 + 1e-12)
            self.assertGreaterEqual((v+a*c.control_interval_s)[0], -1e-12)
        np.testing.assert_allclose(shield._emergency_acceleration(np.array([.01, 0., 0.])),
                                   [-.05, 0., 0.])
        # The 1/H cap also handles a control period larger than v_terminal/b.
        slow = SafetyShield(replace(c, control_interval_s=.5))
        np.testing.assert_allclose(slow._emergency_acceleration(np.array([.01, 0., 0.])),
                                   [-.02, 0., 0.])

    def test_terminal_contract_parameter_validation(self):
        for speed, radius in ((.2, 0.), (-.1, .2), (.2, -.1), (math.inf, .2), (.2, math.nan)):
            with self.assertRaises(ValueError):
                replace(self.config, terminal_speed_mps=speed, terminal_stop_radius_m=radius)
        replace(self.config, terminal_speed_mps=.2, terminal_stop_radius_m=.3)
        # An extra geometric reserve alone remains a conservative legacy stop.
        replace(self.config, terminal_speed_mps=0., terminal_stop_radius_m=.3)

    def test_validation_rejects_invalid_grids_vectors_and_bounds(self):
        with self.assertRaises(ValueError):
            SafetyShield(ShieldConfig(min_braking_acceleration_mps2=6))
        with self.assertRaises(ValueError):
            self.run_shield(velocity=(math.nan, 0, 0))
        self.grid[0, 0, 0] = 3
        with self.assertRaises(ValueError):
            self.run_shield()


if __name__ == '__main__':
    unittest.main()
