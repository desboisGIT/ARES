from fractions import Fraction as F
from pathlib import Path
import json
import unittest
from initial_terminal_membership_v1 import initial_terminal_membership, projective_xyz, sqrt_bounds
from reservation import box_in_ellipsoid


def diagonal(n, value=1):
    return [[F(value) if i==j else F(0) for j in range(n)] for i in range(n)]


def arguments(**changes):
    args=dict(p=[0,0,0],v=[0,0,0],q=[0,0,0,1],w=[0,0,0],anchor=[0,0,0],
              correlated_trim_force_upper=F(1,100),motor_tracking_error=F(1,1000),
              shapes={'planar_x':diagonal(6),'vertical':diagonal(5),'yaw':diagonal(4)})
    args.update(changes)
    return args


def correlated_shape(direction, along=4, other=F(1,100000)):
    return [[along*a*b+(other if i==j else 0) for j,b in enumerate(direction)]
            for i,a in enumerate(direction)]


class MembershipTests(unittest.TestCase):
    def test_diagonal_positive_3d_and_32_correlated_vertices(self):
        result=initial_terminal_membership(**arguments(p=[F(1,10)]*3,v=[F(1,100)]*3))
        self.assertTrue(result['initial_set_included'])
        self.assertEqual(result['channels']['vertical']['vertices_checked'],32)
        self.assertTrue(result['hypotheses_external'])
        self.assertFalse(result['hypotheses_independently_established'])
        self.assertFalse(result['flight_certificate'])

    def test_strong_vertical_correlation_not_cartesian_box(self):
        direction=[F(-4,3),0,1,1,1]
        shape=correlated_shape(direction)
        shapes={'planar_x':diagonal(6),'vertical':shape,'yaw':diagonal(4)}
        result=initial_terminal_membership(**arguments(shapes=shapes,
            correlated_trim_force_upper=1,motor_tracking_error=0,battery_interval=[1,1],
            ep=0,ev=0,ew=0,q_state_overapprox_error=0))
        self.assertTrue(result['channels']['vertical']['included'])
        lo=[-abs(x) for x in direction];hi=[abs(x) for x in direction]
        self.assertFalse(box_in_ellipsoid(shape,lo,hi)['included'])
        # Battery variation remains correlated through b*d, not replaced by d.
        varied=initial_terminal_membership(**arguments(correlated_trim_force_upper=F(1,10)))
        vertices=varied['channels']['vertical']['vertices']
        self.assertEqual(vertices[0][2],F(-19,200)-F(1,1000))

    def test_planar_y_required_sign_transform(self):
        direction=[1,0,F(-3,5),F(-1,5),0,0]
        shape=correlated_shape(direction)
        result=initial_terminal_membership(**arguments(p=[0,1,0],q=[F(3,5),0,0,F(4,5)],
            w=[F(1,5),0,0],ep=0,ev=0,ew=0,q_state_overapprox_error=0,
            shapes={'planar_x':shape,'vertical':diagonal(5),'yaw':diagonal(4)}))
        self.assertTrue(result['channels']['planar_y']['included'])
        wrong=[1,0,F(3,5),F(1,5),0,0]
        self.assertFalse(box_in_ellipsoid(shape,wrong,wrong)['included'])

    def test_projective_sqrt_enclosure_shell_and_state_uncertainty(self):
        q=[F(3,5),0,0,F(4,5)]
        self.assertEqual(projective_xyz(q,0)[0],(F(3,5),F(3,5)))
        q=[0,F(1,1000),0,1]
        with self.assertRaisesRegex(ValueError,'norm shell'): projective_xyz(q,0)
        q=[F(1,10000),0,0,1]
        norm2=sum(x*x for x in q)
        lo,hi=sqrt_bounds(norm2)
        self.assertLessEqual(lo*lo,norm2);self.assertGreaterEqual(hi*hi,norm2)
        bounds=projective_xyz(q,F(1,1000000))[0]
        self.assertLessEqual(bounds[0],q[0]/hi)
        self.assertGreaterEqual(bounds[1],q[0]/lo)
        for bad in ([0,0,0,0],[0,0,0,-1],[0,0,0,2]):
            with self.assertRaises(ValueError): projective_xyz(bad,0)

    def test_ellipsoid_boundary_and_invalid_shape(self):
        result=initial_terminal_membership(**arguments(p=[1,0,0],ep=0,ev=0,ew=0,
            q_state_overapprox_error=0,correlated_trim_force_upper=0,motor_tracking_error=0))
        self.assertTrue(result['initial_set_included'])
        result=initial_terminal_membership(**arguments(p=[1,0,0],ep=F(1,1000000)))
        self.assertFalse(result['initial_set_included'])
        shapes=arguments()['shapes'];shapes['vertical'][0][0]=-1
        self.assertFalse(initial_terminal_membership(**arguments(shapes=shapes))['initial_set_included'])

    def test_missing_domains_and_nonfinite_values_rejected(self):
        for changes in ({'correlated_trim_force_upper':-1},{'motor_tracking_error':-1},
                        {'battery_interval':[F(9,10),1]},{'battery_interval':[1,F(19,20)]},
                        {'ep':True},{'ev':float('nan')},{'q_state_overapprox_error':-1},
                        {'p':[0,float('inf'),0]},{'shapes':{}},{'w':[0,0]}):
            with self.subTest(changes=changes),self.assertRaises((ValueError,TypeError)):
                initial_terminal_membership(**arguments(**changes))
        missing=arguments();del missing['correlated_trim_force_upper']
        with self.assertRaises(TypeError):initial_terminal_membership(**missing)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MembershipTests))
    audit={'schema_version':'ares.initial_terminal_membership_audit.v1','tests_run':result.testsRun,
           'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),
           'flight_certificate':False,'hypotheses_independently_established':False,
           'scope':'Exact initial-set membership under externally supplied domains; no final RPI proof',
           'positive_example':initial_terminal_membership(**arguments())}
    Path(__file__).with_name('initial_terminal_membership_v1_audit.json').write_text(json.dumps(audit,indent=2,default=str)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
