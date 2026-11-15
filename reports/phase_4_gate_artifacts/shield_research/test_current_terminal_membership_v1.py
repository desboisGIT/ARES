from fractions import Fraction as F
from pathlib import Path
from itertools import product
import json
import unittest
from current_terminal_membership_v1 import current_terminal_membership, propagate_public_residuals
from reservation import ReservationCache


def diagonal(n):return [[F(i==j) for j in range(n)] for i in range(n)]

def arguments(**changes):
    args=dict(p=[0,0,0],v=[0,0,0],q=[0,0,0,1],w=[0,0,0],anchor=[0,0,0],
              hold_thrust=5,issued_queue=[[F(51,10),F(1,100),F(2,100),F(3,100)],
                                        [F(49,10),F(-1,100),F(-2,100),F(-3,100)]],
              trim_force_interval=[F(-1,100),F(1,100)],
              motor_residual_interval=[F(1,100),F(2,100)],
              shapes={'planar_x':diagonal(6),'vertical':diagonal(5),'yaw':diagonal(4)})
    args.update(changes);return args


def propagation(**changes):
    args=dict(motor_residual_interval=[F(-1,10),F(2,10)],trim_force_interval=[F(-1,100),F(1,100)],
              decay_interval=[F(51,100),F(79,100)],battery_interval=[F(19,20),1],hold_thrust=5,
              issued_commands=[[4,0,0,0],[5,0,0,0],[6,0,0,0]],
              authority_drop_upper=F(64,10000000),numerical_error=F(7,1000000),mass_gravity_upper=6)
    args.update(changes);return args


class CurrentMembershipTests(unittest.TestCase):
    def test_nonzero_queue_and_motor_center_preserved(self):
        result=current_terminal_membership(**arguments())
        self.assertTrue(result['current_set_included'])
        self.assertEqual(result['channels']['planar_x']['state_bounds'][4],(F(2,100),)*2)
        self.assertEqual(result['channels']['planar_y']['state_bounds'][4],(F(-1,100),)*2)
        self.assertEqual(result['channels']['yaw']['state_bounds'][2],(F(3,100),)*2)
        vertices=result['channels']['vertical']['vertices']
        self.assertEqual(len(vertices),32)
        self.assertEqual(vertices[0][2],F(-19,2000)+F(1,100))
        self.assertEqual(vertices[0][3],F(9,100))
        self.assertEqual(vertices[0][4],F(-11,100))
        self.assertFalse(result['flight_certificate'])

    def test_reanchor_candidate_rejection_leaves_cache_to_caller(self):
        candidate=current_terminal_membership(**arguments(p=[5,0,0],anchor=[5,0,0]))
        self.assertTrue(candidate['current_set_included'])
        rejected=current_terminal_membership(**arguments(p=[5,0,0],anchor=[0,0,0]))
        self.assertFalse(rejected['current_set_included'])
        cache=ReservationCache()
        obligations={n:{'valid':True,'artifact':n} for n in cache.REQUIRED}
        self.assertTrue(cache.propose(time=1,backup_deadline=3,supplier_expiry=4,
                         obligations=obligations,known_artifacts=cache.REQUIRED)['committed'])
        old=cache.cached.copy()
        obligations['geometry']['valid']=rejected['current_set_included']
        self.assertFalse(cache.propose(time=2,backup_deadline=3,supplier_expiry=4,
                          obligations=obligations,known_artifacts=cache.REQUIRED)['committed'])
        self.assertEqual(cache.cached,old)

    def test_exact_transient_vertices_and_interior_enclosure(self):
        args=propagation();result=propagate_public_residuals(**args)
        lo,hi=result['motor_residual_interval']
        self.assertEqual(result['vertices_checked'],64)
        for y,e,b,u,db,error in product([F(-1,10),F(0),F(2,10)],
                [F(51,100),F(13,20),F(79,100)], [F(19,20),F(39,40),F(1)],
                [F(4),F(9,2),F(6)], [F(0),F(32,10000000),F(64,10000000)],
                [F(-7,1000000),F(0),F(7,1000000)]):
            actual=e*y+(1-e)*b*(u-5)+5*db+error
            self.assertLessEqual(lo,actual);self.assertLessEqual(actual,hi)
        self.assertEqual(result['delayed_command_thrust_hull'],(F(4),F(6)))
        # Holding the same U and e shows e*y and (1-e) are not varied independently.
        fixed=propagate_public_residuals(**propagation(motor_residual_interval=[1,1],
             issued_commands=[[6,0,0,0]]*3, battery_interval=[1,1],
             authority_drop_upper=0,numerical_error=0))
        self.assertEqual(fixed['motor_residual_interval'],(F(1),F(1)))

    def test_trim_drift_exact_inequality_for_public_profile_bound(self):
        args=propagation();result=propagate_public_residuals(**args)
        drop=args['authority_drop_upper'];mg=args['mass_gravity_upper']
        for bnext in (F(19,20),F(97,100),1-drop):
            b=bnext+drop
            actual=-mg*drop/(b*bnext)
            self.assertLessEqual(result['trim_drift_lower'],actual)
            self.assertLessEqual(actual,0)
        self.assertEqual(result['trim_force_interval'][1],args['trim_force_interval'][1])
        self.assertLess(result['trim_force_interval'][0],args['trim_force_interval'][0])

    def test_invalid_domains_and_missing_history_reject(self):
        for changes in ({'decay_interval':[-1,1]},{'battery_interval':[F(9,10),1]},
                        {'numerical_error':-1},{'authority_drop_upper':-1},
                        {'issued_commands':[[5,0,0,0]]*2},{'mass_gravity_upper':float('nan')}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                propagate_public_residuals(**propagation(**changes))
        for changes in ({'trim_force_interval':[1,0]},{'issued_queue':[[5,0,0,0]]},
                        {'hold_thrust':True},{'motor_residual_interval':[float('inf'),1]}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                current_terminal_membership(**arguments(**changes))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CurrentMembershipTests))
    audit={'schema_version':'ares.current_terminal_membership_audit.v1','tests_run':result.testsRun,
           'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),
           'flight_certificate':False,'hypotheses_independently_established':False,
           'scope':'Conditional exact public-packet reanchoring and residual interval propagation',
           'membership_example':current_terminal_membership(**arguments()),
           'propagation_example':propagate_public_residuals(**propagation())}
    Path(__file__).with_name('current_terminal_membership_v1_audit.json').write_text(json.dumps(audit,indent=2,default=str)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
