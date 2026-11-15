import unittest
from ares.rate_probe import measure_callable


class RateProbeTests(unittest.TestCase):
    def test_statistics_and_completed_calls(self):
        ticks = iter([0, 1, 2, 4, 5, 8, 9, 13])
        calls = []
        result = measure_callable(lambda: calls.append(1), warmup=2, iterations=4,
                                  clock=lambda: next(ticks), target_hz=.25)
        self.assertEqual(len(calls), 6)
        self.assertEqual(result['median_s'], 2.5)
        self.assertAlmostEqual(result['p90_s'], 3.7)
        self.assertAlmostEqual(result['p99_s'], 3.97)
        self.assertEqual(result['hz'], .4)
        self.assertTrue(result['meets_target'])

    def test_target_failure(self):
        ticks = iter([0, 2])
        self.assertFalse(measure_callable(lambda: None, warmup=0, iterations=1,
                         target_hz=1, clock=lambda: next(ticks))['meets_target'])

    def test_bad_arguments_and_clock(self):
        for kwargs in ({'warmup': -1}, {'iterations': 0}, {'iterations': True},
                       {'target_hz': 0}, {'target_hz': float('nan')}):
            with self.assertRaises(ValueError):
                measure_callable(lambda: None, **kwargs)
        with self.assertRaises(ValueError):
            measure_callable(lambda: None, warmup=0, iterations=1, clock=lambda: 0)
