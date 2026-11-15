from pathlib import Path
from types import SimpleNamespace
import sys,unittest,math,struct
import numpy as np
HERE=Path(__file__).resolve().parents[1];BASE=HERE.parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(BASE))
from adapter_v3 import approximate_backup_word,calibration_hold_xy_candidate,propose_anchor
from verified_backup_point_v2 import backup_point
class AdapterV3Tests(unittest.TestCase):
    def packet(self):return SimpleNamespace(p=(6.,6.,2.),v=(.03,-.02,.005),q=(.001,-.002,.003,math.sqrt(1-.001**2-.002**2-.003**2)))
    def test_fast_scoring_matches_verified_law_but_cannot_issue(self):
        p=self.packet();anchor=[6.02,5.98,2.03]
        actual=backup_point(p.p,p.v,p.q,anchor,5.)['command'];approx=approximate_backup_word(p,anchor,5.)
        self.assertLess(max(abs(a-b) for a,b in zip(actual,approx)),1e-6)
    def test_horizontal_braking_anchor_keeps_exact_target_word(self):
        p=self.packet();target=struct.unpack('f',struct.pack('f',5.01))[0]
        result=calibration_hold_xy_candidate(p,5.,target,[6.02,5.98])
        self.assertEqual(result['anchor'][:2],[6.02,5.98])
        actual=backup_point(p.p,p.v,p.q,result['anchor'],5.)['command']
        self.assertEqual(actual[0],target)
        self.assertTrue(result['exact_guard_issued_word_equality_still_required'])
    def test_outside_camera_intent_advances_lateral_component_first(self):
        bounds=[(0.,0.)]*6
        class Guard:
            anchor=[0.,0.,0.];hold=5.
            proof=SimpleNamespace(shapes={'planar_x':np.eye(6),'vertical':np.eye(5),'yaw':np.eye(4)})
            def _membership(self,p,a):return {'channels':{'planar_x':{'state_bounds':bounds},'planar_y':{'state_bounds':bounds},'yaw':{'state_bounds':[(0.,0.)]*4},'vertical':{'vertices':[[0.]*5]}}}
            def _geometry(self,a):return {'reserved':True}
        p=SimpleNamespace(p=(0.,0.,0.),v=(0.,0.,0.),q=(0.,0.,0.,1.))
        result=propose_anchor(Guard(),p,[5.,-1.2,.3,0.],[0.,0.,0.])
        self.assertTrue(result['coverage_first_lateral_intent'])
        self.assertAlmostEqual(result['candidate_anchor'][0],0.)
        self.assertGreater(result['candidate_anchor'][1],0.)
if __name__=='__main__':unittest.main(verbosity=2)
