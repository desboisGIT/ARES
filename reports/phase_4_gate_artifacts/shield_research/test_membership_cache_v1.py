"""Exact equivalence and nonbinding desktop timing for cached membership."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import random
import statistics
import time
import unittest
from current_terminal_membership_v1 import current_terminal_membership
from reservation import prepare_shape_metric, box_in_ellipsoid

ROOT=Path(__file__).resolve().parents[2]
PROOF=ROOT/'phase_4_shield_artifacts/cpu_rpi_certificate_v1/certificate.json'
ADMISSION=Path(__file__).with_name('initial_contract_v1.json')


def real_arguments():
    proof=json.loads(PROOF.read_text());case=json.loads(ADMISSION.read_text())['cases'][0]
    shapes={}
    for name,group in proof['groups'].items():
        scale=list(map(F,group['state_scales_exact_rational']))
        shapes[name]=[[F(value)*scale[i]*scale[j] for j,value in enumerate(row)]
                      for i,row in enumerate(group['Q_scaled_exact_rational'])]
    def outward(values):
        lo,hi=map(F,values);scale=2**40
        return (F(lo.numerator*scale//lo.denominator,scale),
                F(-((-hi.numerator*scale)//hi.denominator),scale))
    return dict(**case['public_final_packet'],anchor=case['anchor'],hold_thrust=case['hold_thrust'],
                issued_queue=case['issued_queue'],trim_force_interval=outward(case['trim_force_interval']),
                motor_residual_interval=outward(case['motor_residual_interval']),shapes=shapes)


def full_quadratic(matrix,x):
    return sum(x[i]*matrix[i][j]*x[j] for i in range(len(x)) for j in range(len(x)))


class CacheEquivalenceTests(unittest.TestCase):
    def test_random_correlated_spd_cached_against_full_exact_vertices(self):
        rng=random.Random(482)
        for n in range(1,7):
            for _ in range(8):
                a=[[F(rng.randrange(-3,4),3) for _ in range(n)] for _ in range(n)]
                shape=[[sum(a[i][k]*a[j][k] for k in range(n))+F(i==j) for j in range(n)] for i in range(n)]
                metric=prepare_shape_metric(shape)
                self.assertIs(metric,prepare_shape_metric(tuple(tuple(row) for row in shape)))
                lo=[F(rng.randrange(-4,2),5) for _ in range(n)]
                hi=[x+(F(0) if i%2 else F(1,5)) for i,x in enumerate(lo)]
                expected=max(full_quadratic(metric.inverse,x) for x in product(*zip(lo,hi)))
                cached=box_in_ellipsoid(shape,lo,hi,metric=metric)
                self.assertEqual(cached['maximum'],expected)
                self.assertEqual(cached['included'],expected<=1)
                self.assertEqual(cached['corners_checked'],2**sum(a!=b for a,b in zip(lo,hi)))
        with self.assertRaises(ValueError):box_in_ellipsoid([[2]],[0],[0],metric=prepare_shape_metric([[1]]))

    def test_real_case_optional_metrics_and_full_vertex_equivalence(self):
        args=real_arguments();metrics={name:prepare_shape_metric(q) for name,q in args['shapes'].items()}
        first=current_terminal_membership(**args)
        cached=current_terminal_membership(**args,metrics=metrics)
        self.assertEqual(first,cached)
        for name,channel in cached['channels'].items():
            shape='planar_x' if name=='planar_y' else name
            vertices=channel.get('vertices')
            if vertices is None:vertices=product(*channel['state_bounds'])
            maximum=max(full_quadratic(metrics[shape].inverse,x) for x in vertices)
            self.assertEqual(channel['maximum'],maximum)
        self.assertLessEqual(cached['channels']['planar_x']['corners_checked'],16)
        self.assertLessEqual(cached['channels']['yaw']['corners_checked'],4)


def benchmark():
    args=real_arguments();metrics={name:prepare_shape_metric(q) for name,q in args['shapes'].items()}
    # Warm caches first. Timing is evidence, never a platform acceptance assert.
    result=current_terminal_membership(**args,metrics=metrics)
    samples=[]
    for _ in range(100):
        start=time.perf_counter_ns();current_terminal_membership(**args,metrics=metrics)
        samples.append((time.perf_counter_ns()-start)/1e6)
    return {'samples':len(samples),'median_ms':statistics.median(samples),
            'maximum_ms':max(samples),'p95_ms':sorted(samples)[94],
            'desktop_only':True,'timing_threshold_asserted':False,
            'intervals_outward_quantized_grid':'2^-40',
            'real_initial_set_included':result['current_set_included'],
            'corners':{name:value.get('corners_checked',value.get('vertices_checked'))
                       for name,value in result['channels'].items()}}


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CacheEquivalenceTests))
    audit={'schema_version':'ares.membership_cache_equivalence.v1','tests_run':result.testsRun,
           'passed':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),
           'benchmark':benchmark(),'flight_certificate':False,'scope':'Exact cache/dedup equivalence and desktop timing only'}
    Path(__file__).with_name('membership_cache_v1_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
