"""Exact-reference regressions for the independent outward matrix checker."""
from fractions import Fraction as F
import itertools
import json
from pathlib import Path
import random
import unittest

import numpy as np

import interval_lmi_audit_v1 as audit


def scalar(value):
    return F.from_float(float(value))


def enclosed(interval, value):
    return scalar(interval[0]) <= value <= scalar(interval[1])


def exact_spd(matrix):
    n = len(matrix)
    lower = [[F(i == j) for j in range(n)] for i in range(n)]
    diagonal = []
    for j in range(n):
        pivot = matrix[j][j]-sum(lower[j][k]**2*diagonal[k] for k in range(j))
        if pivot <= 0:
            return False
        diagonal.append(pivot)
        for i in range(j+1, n):
            lower[i][j] = (matrix[i][j]-sum(lower[i][k]*lower[j][k]*diagonal[k]
                                           for k in range(j)))/pivot
    return True


def data(vertices=5):
    n = 3
    result = {}
    arrays = {'A': np.tile(.5*np.eye(n), (vertices, 1, 1)),
              'Q': np.eye(n), 'outer_Q': 2*np.eye(n),
              'command_w': np.tile([.01, -.02, .03], (vertices, 1)),
              'additive_offsets': np.array(list(itertools.product((-.001, .001), repeat=n)))}
    for name, array in arrays.items():
        result[name+'_lo'] = array.copy()
        result[name+'_hi'] = array.copy()
    return result


class IntervalAuditTests(unittest.TestCase):
    def test_arithmetic_against_exact_rational_endpoints(self):
        rng = random.Random(772341)
        for _ in range(1000):
            a = sorted(rng.uniform(-10., 10.) for _ in range(2))
            b = sorted(rng.uniform(.01, 10.) for _ in range(2))
            x, y = audit.iv(*a), audit.iv(*b)
            for aa, bb in itertools.product(map(scalar, a), map(scalar, b)):
                self.assertTrue(enclosed(audit.add(x, y), aa+bb))
                self.assertTrue(enclosed(audit.sub(x, y), aa-bb))
                self.assertTrue(enclosed(audit.mul(x, y), aa*bb))
                self.assertTrue(enclosed(audit.divide(x, y), aa/bb))
            for aa in (*map(scalar, a), F(0)):
                if scalar(a[0]) <= aa <= scalar(a[1]):
                    self.assertTrue(enclosed(audit.square(x), aa*aa))
        for a, b in ((1., 2.**-53), (2.**-1022, 2.**-53), (2.**-1074, .5)):
            with np.errstate(under='ignore'):
                self.assertTrue(enclosed(audit.mul(audit.iv(a), audit.iv(b)), scalar(a)*scalar(b)))
        self.assertTrue(enclosed(audit.exact_rational_interval('99/100'), F(99, 100)))
        with self.assertRaises(ArithmeticError):
            audit.divide(audit.iv(1.), audit.iv(-1., 1.))

    def test_interval_ldl_against_fraction_ldl(self):
        rng = random.Random(913)
        for n in range(1, 7):
            for _ in range(10):
                b = [[F(rng.randint(-8, 8), 16) for _ in range(n)] for _ in range(n)]
                matrix = [[sum(b[i][k]*b[j][k] for k in range(n))+F(i == j, 16)
                           for j in range(n)] for i in range(n)]
                represented = np.array([[float(v) for v in row] for row in matrix])
                self.assertTrue(exact_spd(matrix))
                self.assertTrue(audit.interval_ldl(audit.iv(represented))['passed'])
        self.assertFalse(audit.interval_ldl(audit.iv([[1., 1.], [1., 1.]]))['passed'])
        self.assertFalse(audit.interval_ldl(audit.iv([[1., 1.], [1., 1.-2.**-50]]))['passed'])

    def test_all_vertices_signs_and_global_command_signs(self):
        result = audit.verify_arrays(data(), alpha='9/10', chunk_size=2, command_signs=(-1, 1))
        self.assertTrue(result['passed'])
        self.assertEqual(result['matrix_inequalities_checked'], 5*8*2)
        self.assertTrue(result['every_supplied_vertex_and_sign_checked'])
        self.assertFalse(result['flight_certificate'])

    def test_bad_last_vertex_is_not_hidden_by_chunking(self):
        candidate = data()
        candidate['A_lo'][-1] = candidate['A_hi'][-1] = np.eye(3)*1.1
        result = audit.verify_arrays(candidate, alpha='9/10', chunk_size=2)
        self.assertFalse(result['passed'])
        self.assertEqual(result['vertex_begin'], 4)
        self.assertEqual(result['failure']['bad_batch_indices'], [[0]])

    def test_shape_margin_and_bad_inputs_fail(self):
        candidate = data()
        candidate['outer_Q_lo'] = candidate['outer_Q_hi'] = np.eye(3)
        self.assertFalse(audit.verify_arrays(candidate, alpha='9/10')['passed'])
        candidate = data()
        candidate['Q_lo'][0, 1] = 1.
        with self.assertRaises(ValueError):
            audit.verify_arrays(candidate, alpha='9/10')
        with self.assertRaises(ValueError):
            audit.verify_arrays(data(), alpha='1')
        with self.assertRaises(ValueError):
            audit.iv(float('nan'))

    def test_parameter_dependent_offsets(self):
        candidate = data()
        for name in ('additive_offsets_lo', 'additive_offsets_hi'):
            candidate[name] = np.tile(candidate[name], (5, 1, 1))
        self.assertTrue(audit.verify_arrays(candidate, alpha='9/10')['passed'])
        candidate['additive_offsets_lo'][-1, -1] = 1.
        candidate['additive_offsets_hi'][-1, -1] = 1.
        self.assertFalse(audit.verify_arrays(candidate, alpha='9/10')['passed'])


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IntervalAuditTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    Path(__file__).with_suffix('.json').write_text(json.dumps({
        'tests_run': result.testsRun, 'passed': result.wasSuccessful(),
        'arithmetic_rational_reference_intervals': 1000,
        'SPD_fraction_reference_matrices': 60,
        'flight_certificate': False}, indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
