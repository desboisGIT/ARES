"""Exact old64-corner equivalence; no uncertainty or physics changes."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import random
import statistics
import time
import unittest
from current_terminal_membership_v1 import propagate_public_residuals
from reservation import ExactShapeMetric,box_in_ellipsoid,prepare_shape_metric
from test_current_terminal_membership_v1 import propagation

HERE=Path(__file__).resolve().parent


def old_bounds(args):
    y=args['motor_residual_interval'];e=args['decay_interval'];b=args['battery_interval']
    thrusts=[F(c[0]) for c in args['issued_commands']];hold=F(args['hold_thrust'])
    values=[ee*yy+(1-ee)*bb*(u-hold)+hold*db+error
            for yy,ee,bb,u,db,error in product(y,e,b,(min(thrusts),max(thrusts)),
                (F(0),args['authority_drop_upper']),(-args['numerical_error'],args['numerical_error']))]
    return min(values),max(values)


def real_history():
    case=json.loads((HERE/'initial_contract_v1.json').read_text())['cases'][0]
    records=json.loads((HERE/'coupled_reanchoring_v1/retry_002/case_0.json').read_text())['cases'][0]['trace']
    certificate=json.loads((HERE.parents[1]/'phase_4_shield_artifacts/cpu_rpi_certificate_v1/certificate.json').read_text())
    physical=certificate['physical_parameter_intervals_outward']
    mg=F(physical['mass'][1])*F(physical['gravity'][1])
    args=propagation(motor_residual_interval=list(map(F,case['motor_residual_interval'])),
                     trim_force_interval=list(map(F,case['trim_force_interval'])),
                     hold_thrust=F(case['hold_thrust']),mass_gravity_upper=mg)
    queue=[tuple(map(F,c)) for c in case['issued_queue']]
    # Match caller's outward dyadic endpoint representation.
    def outward(bounds,bits):
        lo,hi=bounds;s=2**bits
        return F(lo.numerator*s//lo.denominator,s),F(-((-hi.numerator*s)//hi.denominator),s)
    args['motor_residual_interval']=outward(args['motor_residual_interval'],40)
    args['trim_force_interval']=outward(args['trim_force_interval'],40)
    for record in records:
        word=tuple(map(F,record['command']))
        args['issued_commands']=[word,*queue]
        yield dict(args)
        result=propagate_public_residuals(**args)
        args['motor_residual_interval']=outward(result['motor_residual_interval'],48)
        args['trim_force_interval']=result['trim_force_interval']
        queue=[word,queue[0]]


class ExtremaTests(unittest.TestCase):
    def test_3000_random_exact_intervals_match_old64(self):
        rng=random.Random(7839)
        for _ in range(3000):
            def interval(low,high,den):return sorted(F(rng.randrange(low,high),den) for _ in range(2))
            args=propagation(motor_residual_interval=interval(-30,31,17),
                 decay_interval=interval(0,101,100),battery_interval=interval(950,1001,1000),
                 hold_thrust=F(rng.randrange(1,100),11),
                 issued_commands=[[F(rng.randrange(0,100),13),0,0,0] for _ in range(3)],
                 authority_drop_upper=F(rng.randrange(100),10000000),
                 numerical_error=F(rng.randrange(100),10000000))
            result=propagate_public_residuals(**args)
            self.assertEqual(result['motor_residual_interval'],old_bounds(args))
            self.assertEqual(result['vertices_checked'],64)
            self.assertEqual(result['actual_mixture_evaluations'],4)

    def test_real_public_residual_history_exact_equivalence(self):
        count=0
        for args in real_history():
            self.assertEqual(propagate_public_residuals(**args)['motor_residual_interval'],old_bounds(args))
            count+=1
        self.assertGreaterEqual(count,1000)

    def test_forged_same_shape_inverse_rejected(self):
        trusted=prepare_shape_metric([[1,0],[0,1]])
        forged=ExactShapeMetric(trusted.shape,((F(0),F(0)),(F(0),F(0))))
        with self.assertRaisesRegex(ValueError,'verified shape inverse'):
            box_in_ellipsoid([[1,0],[0,1]],[10,10],[10,10],metric=forged)
        self.assertFalse(box_in_ellipsoid([[1,0],[0,1]],[10,10],[10,10],metric=trusted)['included'])


def benchmark():
    args=next(real_history());samples={'old64':[],'optimized':[]}
    for _ in range(100):
        for name,call in (('old64',lambda:old_bounds(args)),
                          ('optimized',lambda:propagate_public_residuals(**args))):
            start=time.perf_counter_ns();call();samples[name].append((time.perf_counter_ns()-start)/1e6)
    return {name:{'median_ms':statistics.median(values),'max_ms':max(values),'samples':len(values)}
            for name,values in samples.items()}


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExtremaTests))
    audit={'schema_version':'ares.residual_extrema_equivalence.v1','tests_run':result.testsRun,
           'passed':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),
           'random_rational_cases':3000,'real_history_source':'coupled_reanchoring_v1/retry_002/case_0.json',
           'benchmark':benchmark(),'flight_certificate':False,'timing_gate_claim':False}
    (HERE/'residual_extrema_v1_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
