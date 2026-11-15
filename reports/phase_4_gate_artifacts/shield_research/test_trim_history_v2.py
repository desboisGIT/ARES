"""Conditional lemma tests only; supplied hypotheses are not certified here."""
from fractions import Fraction as F
from pathlib import Path
import json
import unittest
from trim_history_v2 import rk4_trim_bound


def arguments(**changes):
    args = dict(times=[F(i, 100) for i in range(101)],
                measured_vertical_velocity=[F(0)] * 101,
                velocity_error=F(0), control_period=F(1, 100),
                physics_period=F(1, 500), hold_thrust=5,
                public_force_upper=6, mass_lower=F(1, 2), drag_upper=F(1, 5),
                authority_drop_per_period_upper=F(0), motor_lag_upper=F(1, 25),
                transport_upper=F(0), constant_command_settle=1,
                stage_up_axis_error=F(0), stage_angular_rate_upper=F(0),
                velocity_roundoff_per_period=F(0), motor_roundoff_per_period=F(0))
    args.update(changes)
    return args


def scalar_step(v, h, mismatch, drag):
    fun = lambda x: mismatch - drag * x
    k1 = fun(v)
    v2 = v + h * k1 / 2
    k2 = fun(v2)
    v3 = v + h * k2 / 2
    k3 = fun(v3)
    v4 = v + h * k3
    k4 = fun(v4)
    return v + h * (k1 + 2*k2 + 2*k3 + k4) / 6, (v, v2, v3, v4), (k1, k2, k3, k4)


class TrimHistoryTests(unittest.TestCase):
    def test_exact_scalar_rk4_telescoping_and_stage_enclosure(self):
        h = F(1, 100)
        for drag in (F(0), F(1, 5), F(2, 5)):
            for mismatch in (F(-1, 10), F(0), F(1, 10)):
                with self.subTest(drag=drag, mismatch=mismatch):
                    values, stages, accelerations, weighted = [F(1, 50)], [], [], F(0)
                    for _ in range(10):
                        end, sv, sa = scalar_step(values[-1], h, mismatch, drag)
                        weighted += h * (sa[0] + 2*sa[1] + 2*sa[2] + sa[3]) / 6
                        values.append(end); stages.extend(sv); accelerations.extend(sa)
                    self.assertEqual(values[-1] - values[0], weighted)
                    result = rk4_trim_bound(**arguments(times=[i*h for i in range(11)],
                        measured_vertical_velocity=values, physics_period=h,
                        drag_upper=drag/2))
                    self.assertLessEqual(abs(mismatch), result['conditional_trim_acceleration_bound'])
                    self.assertLessEqual(max(map(abs, stages)), result['conditional_stage_vertical_speed_bound'])
                    self.assertLessEqual(max(map(abs, accelerations)), result['conditional_stage_acceleration_bound'])
                    self.assertFalse(result['uses_continuous_ODE_solution'])

    def test_full_step_fourth_quaternion_stage_needs_full_h_bound(self):
        # qdot=Aq, ||A||=omega/2, exact rational RK4 stage quaternions.
        h, omega = F(1, 10), F(2)
        a = h * omega / 2
        stages = ((1, 0), (1, a/2), (1-a*a/4, a/2),
                  (1-a*a/2, a-a*a*a/4))
        full = 1 / (1 - h*omega)
        for q in stages:
            self.assertLessEqual(sum(x*x for x in q), full*full)
        # A generic recurrence bound for fourth stage contains h, not h/2.
        fourth_norm_bound = 1 + a*(1 + a/2*(1 + a/2))
        self.assertGreater(fourth_norm_bound, 1 + a/2*(1 + a/2*(1 + a/2)))
        result = rk4_trim_bound(**arguments(physics_period=F(1, 500),stage_angular_rate_upper=2))
        self.assertEqual(result['stage_drag_multiplier_upper'], 1/(1-F(1, 250))**4)

    def test_sampling_denominator_and_changed_packet_gaps(self):
        tight = rk4_trim_bound(**arguments())
        sparse = rk4_trim_bound(**arguments(times=[F(0), F(1, 2), F(1)],
                                           measured_vertical_velocity=[0, 0, 0]))
        self.assertGreater(sparse['conditional_trim_acceleration_bound'],tight['conditional_trim_acceleration_bound'])
        with self.assertRaisesRegex(ValueError, 'sampling gap'):
            rk4_trim_bound(**arguments(times=[0, 1],measured_vertical_velocity=[0, 0],drag_upper=F(1, 4)))
        near = rk4_trim_bound(**arguments(times=[0, 1],measured_vertical_velocity=[0, 0],drag_upper=F(249, 1000)))
        self.assertGreater(near['conditional_stage_acceleration_bound'], tight['conditional_stage_acceleration_bound'])

    def test_exact_timestamp_grid_and_rational_inputs(self):
        rk4_trim_bound(**arguments(times=[F(0),F(1,100)],measured_vertical_velocity=[0,0]))
        with self.assertRaisesRegex(ValueError, 'exact macro grid'):
            rk4_trim_bound(**arguments(times=[0.,.01],measured_vertical_velocity=[0,0]))
        for times in ([0,0], [1,0]):
            with self.assertRaises(ValueError):
                rk4_trim_bound(**arguments(times=times,measured_vertical_velocity=[0,0]))

    def test_numerical_jump_and_motor_error_accumulation(self):
        base = rk4_trim_bound(**arguments())
        rounded = rk4_trim_bound(**arguments(velocity_roundoff_per_period=F(1,10000)))
        self.assertGreater(rounded['conditional_stage_vertical_speed_bound'], base['conditional_stage_vertical_speed_bound'])
        motor = rk4_trim_bound(**arguments(motor_roundoff_per_period=F(1,1000)))
        self.assertGreater(motor['motor_tracking_force_bound'], F(1,1000))
        self.assertGreater(motor['conditional_trim_acceleration_bound'],base['conditional_trim_acceleration_bound'])
        tilted = rk4_trim_bound(**arguments(motor_roundoff_per_period=100,stage_up_axis_error=F(1,10),drag_upper=0))
        expected_fixed = (tilted['motor_tracking_force_bound']+(5+tilted['motor_tracking_force_bound'])*F(1,10))/F(1,2)
        self.assertEqual(tilted['conditional_trim_acceleration_bound'], expected_fixed)

    def test_invalid_domains_and_hypotheses_never_certified(self):
        for changes in ({'mass_lower':0},{'velocity_error':-1},{'hold_thrust':7},
                        {'battery_lower':2},{'constant_command_settle':0},
                        {'stage_angular_rate_upper':500},{'motor_roundoff_per_period':-1},
                        {'physics_period':F(3,1000)}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                rk4_trim_bound(**arguments(**changes))
        result = rk4_trim_bound(**arguments())
        self.assertFalse(result['hypotheses_independently_established'])
        self.assertFalse(result['flight_certificate'])


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TrimHistoryTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    audit={'schema_version':'ares.trim_history_v2_test_audit.v1','tests_run':result.testsRun,
           'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),
           'hypotheses_supplied_by_this_test':False,'flight_certificate':False,
           'review_correction':'Root included motor-roundoff overshoot times stage-up-axis error in fixed_error.',
           'scope':'Exact finite scalar RK4 cases and conditional algebra/domain checks only',
           'example':rk4_trim_bound(**arguments())}
    Path(__file__).with_name('trim_history_v2_test_audit.json').write_text(json.dumps(audit,indent=2,default=str)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
