"""Targeted exact rounding/updater cases; not a native pow or flight proof."""
from fractions import Fraction as F
from pathlib import Path
import json
import unittest
from verified_battery_v1 import (bits, word, frac, positive_bin, rn32_positive,
                                 power32, authority32, energy_step32, updater_bound)


class BatteryBoundaryTests(unittest.TestCase):
    def test_ties_even_at_zero_subnormals_and_normal_boundary(self):
        self.assertEqual(bits(rn32_positive(F(1, 2**150))), 0)
        self.assertEqual(bits(rn32_positive(F(3, 2**150))), 2)
        self.assertEqual(bits(rn32_positive(F(5, 2**150))), 2)
        boundary = (frac(word(0x007fffff)) + frac(word(0x00800000))) / 2
        self.assertEqual(bits(rn32_positive(boundary)), 0x00800000)
        for index in (0x3f800000, 0x3f800001):
            lo, hi = positive_bin(index)
            self.assertEqual(bits(rn32_positive(hi)), index if index % 2 == 0 else index+1)

    def test_power_exact_tie_and_underflow_bins(self):
        # T=r² is exactly representable; T^(3/2)=r³ is a Float32 midpoint.
        r = F(257, 256)
        thrust, exact_power = r*r, r*r*r
        rounded = power32(thrust)
        self.assertEqual(rounded, rn32_positive(exact_power))
        index = bits(rounded)
        lo, hi = positive_bin(index)
        self.assertTrue(exact_power == lo or exact_power == hi)
        self.assertEqual(index % 2, 0)
        self.assertEqual(bits(power32(word(1))), 0)
        self.assertEqual(bits(power32(word(0x007fffff))), 0)
        for thrust in (F(0), F(1), F(13,2)):
            result = power32(thrust)
            lo, hi = positive_bin(bits(result))
            self.assertLessEqual(thrust**3, hi*hi)
            self.assertTrue(lo <= 0 or lo*lo <= thrust**3)

    def test_updater_preserves_declared_order_and_monotonicity(self):
        bound = updater_bound()
        for energy in (0., 8000., 10000.):
            for thrust in (0., word(1), .5, 6.5):
                step = rn32_positive(frac(power32(thrust))*frac(word(bits(.01))))
                increment = rn32_positive(frac(step)*8)
                expected = rn32_positive(frac(energy)+frac(increment))
                updated = energy_step32(energy,thrust)
                self.assertEqual(bits(updated), bits(expected))
                self.assertGreaterEqual(updated,energy)
                self.assertLessEqual(frac(updated)-frac(energy), bound['energy_jump_upper_J'])
                for capacity in (43200.,100000.):
                    drop = frac(authority32(energy,capacity))-frac(authority32(updated,capacity))
                    self.assertGreaterEqual(drop,0)
                    self.assertLessEqual(drop,bound['declared_authority_decrease_per_period'])
        self.assertEqual(energy_step32(8000.,word(1)),8000.)
        self.assertFalse(bound['flight_certificate'])
        self.assertFalse(bound['native_pow_audited'])

    def test_domain_failures_are_explicit(self):
        for call in (lambda: power32(-1), lambda: power32(7),
                     lambda: rn32_positive(F(20001)), lambda: energy_step32(10001,1),
                     lambda: authority32(1,43199), lambda: power32(float('nan'))):
            with self.assertRaises(ValueError):call()


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(BatteryBoundaryTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    audit={'schema_version':'ares.verified_battery_boundary_audit.v1','tests_run':result.testsRun,
           'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),
           'flight_certificate':False,'native_pow_audited':False,
           'scope':'Targeted rational tie/subnormal/update tests of verified research backend; supplied arithmetic hypotheses remain conditional'}
    Path(__file__).with_name('verified_battery_v1_boundary_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
