import math
import unittest
from fractions import Fraction as F
from trim_history import sqrt_upper, exp_negative_upper, trim_bound, demo


class TrimHistoryTests(unittest.TestCase):
    def test_sqrt_enclosure_uses_exact_square(self):
        for value in (F(0),F(2),F(49,4),F(13,2),F(245,10)):
            upper=sqrt_upper(value)
            self.assertGreaterEqual(upper*upper,value)
            self.assertLess((upper-F(1,10**12))**2,value if value else F(1))

    def test_decay_bound_matches_positive_series_argument(self):
        for value in (F(0),F(1),F(12),F(17)):
            self.assertGreaterEqual(float(exp_negative_upper(value)),math.exp(-float(value))*(1-1e-14))
        self.assertEqual(exp_negative_upper(F(0)),1)

    def test_finite_3d_supplier_scenario_reduces_trim_uncertainty(self):
        result=demo()
        self.assertLess(result['conditional_trim_acceleration_bound'],F(1,20))
        # m_upper=.55 is a declared physical bound, not the episode's actual mass.
        displacement=F(55,100)*result['position_offset_coefficient_per_mass_upper']*result['conditional_trim_acceleration_bound']
        self.assertLess(displacement,F(1,25))
        self.assertFalse(result['flight_certificate'])
        self.assertFalse(result['supplier_contract_established'])

    def test_unstable_or_fast_history_cannot_claim_the_small_trim_bound(self):
        kwargs=dict(times=[0.,.5,1.],measured_vertical_velocity=[0.,.5,1.],
            velocity_error=.01,hold_thrust=5.,public_force_upper=6.5,
            mass_lower=.45,drag_upper=.2,battery_energy_lower=43200.,
            motor_lag_upper=.04,transport_upper=.02,constant_command_settle=.7,
            up_axis_error=.0002)
        result=trim_bound(**kwargs)
        self.assertGreater(result['conditional_trim_acceleration_bound'],1)
        with self.assertRaises(ValueError):trim_bound(**(kwargs|{'times':[0.,0.,1.]}))
        with self.assertRaises(ValueError):trim_bound(**(kwargs|{'velocity_error':math.nan}))


if __name__=='__main__':unittest.main()
