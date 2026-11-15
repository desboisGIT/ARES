import json
from pathlib import Path
import unittest
from reservation import F, ReservationCache, box_in_ellipsoid, box_in_metric_ellipsoid, free_grid_reserve, rational, spd_ldl


class ExactReservationTests(unittest.TestCase):
    def grid(self, lo=(1.25, 1.25, 1.25), hi=(1.5, 1.5, 1.5), **changes):
        args = dict(origin=(0, 0, 0), shape=(4, 4, 4), resolution=1,
                    free_cells={(1, 1, 1)}, body_radius='.1', interstep_reserve='.1')
        args.update(changes)
        return free_grid_reserve(lo, hi, **args)

    def test_positive_full_3d_and_exact_inflation(self):
        result = self.grid()
        self.assertTrue(result['reserved'])
        self.assertEqual(result['intersecting_cells'], [(1, 1, 1)])
        self.assertEqual(result['inflated_lower'], (F(21, 20),) * 3)
        self.assertFalse(result['flight_certificate'])

    def test_closed_unknown_faces_edges_and_corners(self):
        for hi in ((2, 1.5, 1.5), (2, 2, 1.5), (2, 2, 2)):
            with self.subTest(hi=hi):
                result = self.grid(hi=hi, body_radius=0, interstep_reserve=0)
                self.assertFalse(result['reserved'])
                self.assertIn((2, 1, 1), result['unavailable_cells'])
        result = self.grid(lo=(1, 1, 1), body_radius=0, interstep_reserve=0)
        self.assertIn((0, 0, 0), result['unavailable_cells'])

    def test_support_boundary_and_nonfree_interior(self):
        self.assertEqual(self.grid(lo=(0, 1, 1), body_radius=0,
                                   interstep_reserve=0)['reason'], 'support_boundary')
        all_free = {(i, j, k) for i in range(4) for j in range(4) for k in range(4)}
        self.assertFalse(self.grid(hi=(4, 2, 2), body_radius=0, interstep_reserve=0,
                                  free_cells=all_free)['reserved'])
        self.assertFalse(self.grid(free_cells=set())['reserved'])

    def test_binary64_cancellation_is_not_rounded_geometry(self):
        # Binary64 subtract rounds this back to 1.0; exact represented subtraction
        # is strictly below 1, so the neighboring unknown cell must be included.
        tiny = 2. ** -54
        self.assertEqual(1.0 - tiny, 1.0)
        result = self.grid(lo=(1., 1.25, 1.25), body_radius=tiny, interstep_reserve=0)
        self.assertLess(result['inflated_lower'][0], rational(1.))
        self.assertFalse(result['reserved'])

    def test_ellipsoid_all_vertices_boundary_and_indefinite(self):
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = box_in_ellipsoid(identity, ['-1/3'] * 3, ['1/3'] * 3)
        self.assertTrue(result['included'])
        self.assertEqual(result['corners_checked'], 8)
        self.assertTrue(box_in_ellipsoid(identity, [1, 0, 0], [1, 0, 0])['included'])
        self.assertFalse(box_in_ellipsoid(identity, [1, 0, 0], [1, '.01', 0])['included'])
        self.assertFalse(box_in_ellipsoid([[1, 2], [2, 1]], [0, 0], [0, 0])['included'])
        self.assertFalse(spd_ldl([[1, 0], [0, 0]])['spd'])
        q = [[2., '1/2'], ['1/2', 1.]]
        ldl = spd_ldl(q)
        self.assertTrue(ldl['spd'])
        for i in range(2):
            for j in range(2):
                self.assertEqual(sum(ldl['lower'][i][k] * ldl['diagonal'][k] * ldl['lower'][j][k]
                                     for k in range(2)), rational(q[i][j]))

    def test_shape_covariance_inverse_is_not_direct_metric(self):
        shape = [[4, 0], [0, '1/4']]
        self.assertTrue(box_in_ellipsoid(shape, [2, 0], [2, 0])['included'])
        self.assertFalse(box_in_metric_ellipsoid(shape, [2, 0], [2, 0])['included'])
        self.assertFalse(box_in_ellipsoid(shape, [0, 1], [0, 1])['included'])
        self.assertTrue(box_in_metric_ellipsoid(shape, [0, 1], [0, 1])['included'])
        correlated = [[2, 1], [1, 2]]
        result = box_in_ellipsoid(correlated, [1, 1], [1, 1])
        self.assertTrue(result['included'])
        self.assertEqual(result['maximum'], F(2, 3))
        self.assertEqual(result['shape_matrix_inverse'], [[F(2, 3), F(-1, 3)],
                                                        [F(-1, 3), F(2, 3)]])
        self.assertFalse(box_in_ellipsoid(correlated, [1, -1], [1, -1])['included'])

    def test_cache_rejection_preserves_record_but_not_expiry(self):
        cache = ReservationCache()
        obligations = {n: {'valid': True, 'artifact': n + '.json'} for n in cache.REQUIRED}
        known = [v['artifact'] for v in obligations.values()]
        propose = lambda t, d, e, o=obligations, k=known: cache.propose(
            time=t, backup_deadline=d, supplier_expiry=e, obligations=o, known_artifacts=k)
        self.assertTrue(propose(1, 3, 4)['committed'])
        original = cache.cached.copy()
        bad = dict(obligations, terminal_invariance={'valid': False, 'artifact': 'terminal_invariance.json'})
        self.assertFalse(propose(2, 3, 4, bad)['committed'])
        self.assertEqual(cache.cached, original)
        self.assertIsNotNone(cache.current('2.5'))
        self.assertFalse(propose(2, 3, 4)['committed'])
        self.assertFalse(propose('1.5', 3, 4)['committed'])
        self.assertIsNone(cache.current('1.5'))
        self.assertFalse(propose('2.6', 5, 4)['committed'])
        self.assertFalse(propose('2.7', 2, 4)['committed'])
        self.assertFalse(propose('2.8', 3, 4, k=[])['committed'])
        self.assertEqual(cache.cached, original)
        self.assertIsNone(cache.current('3.1'))
        self.assertIsNone(cache.current(5))
        self.assertIsNone(cache.current('2.9'))
        self.assertFalse(cache.cached['flight_certificate'])

    def test_geometry_alone_and_invalid_dimensions_fail_closed(self):
        cache = ReservationCache()
        self.assertFalse(cache.propose(time=1, backup_deadline=2, supplier_expiry=3,
                         obligations={'geometry': {'valid': True, 'artifact': 'g'}},
                         known_artifacts=['g'])['committed'])
        with self.assertRaises(ValueError): spd_ldl([[1] * 7] * 7)
        with self.assertRaises(ValueError): spd_ldl([[1, 0], [1, 1]])
        with self.assertRaises(ValueError): self.grid(body_radius=-1)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExactReservationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    audit = {'schema_version': 'ares.research_terminal_reservation.v1',
             'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
             'passed': result.wasSuccessful(), 'flight_certificate': False,
             'rpi_proof_complete': False, 'scope': 'Exact represented-input geometry and caller-declared reservation metadata only',
             'positive_3d': ExactReservationTests().grid(),
             'ellipsoid_boundary': box_in_ellipsoid([[1, 0], [0, 1]], [1, 0], [1, 0])}
    Path(__file__).with_name('audit.json').write_text(json.dumps(audit, indent=2, default=str) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
