from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction as F
import math,sys,unittest
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
from trim_target import choose_trim_word
from mapper import ForwardMapper,extend_guard_free_snapshot
class PublicProposalTests(unittest.TestCase):
    def records(self,acceleration=.02,initial_velocity=0.):
        return [{'global_tick':i,'public_packet':{'v':[0,0,initial_velocity+acceleration*i*.01]},'issued_command':[4.5,0,0,0]} for i in range(100)]
    def test_untrusted_force_balance_estimator(self):
        proposal=choose_trim_word(self.records())
        self.assertAlmostEqual(proposal['public_vz_slope_mps2'],.02,places=13)
        self.assertAlmostEqual(proposal['zero_acceleration_word_estimate_N'],4.49,places=13)
        self.assertTrue(proposal['proposal_only_not_a_trim_proof'])
        self.assertFalse(proposal['sampled_mass_motor_energy_used'])
        self.assertGreater(proposal['target_word'],4.49)
    def test_no_constant_velocity_braking_overshoot_term(self):
        a=choose_trim_word(self.records(initial_velocity=.01))
        b=choose_trim_word(self.records(initial_velocity=-.01))
        self.assertEqual(a['target_word'],b['target_word'])
    def test_mapper_only_requests_forward_FOV_and_bounded_range(self):
        class Map:
            shape=(3,3,3);origin=(-1.,-1.,-1.);resolution=1.
            def __init__(self):self.free={};self.calls=[]
            def observe(self,ray,cell,**kwargs):self.calls.append(ray.world_direction);self.free[cell]=True;return SimpleNamespace(free_witness=True)
        class Supplier:
            def ray(self,packet,direction,**kwargs):return SimpleNamespace(world_direction=direction,measured_depth=.5)
        known=Map();mapper=ForwardMapper(known,horizontal_fov=math.pi/2,vertical_fov=math.pi/2,max_range=3.,rays_per_frame=100)
        packet=SimpleNamespace(p=(0,0,0),q=(0,0,0,1))
        result=mapper.observe(Supplier(),packet,time_s=0.)
        self.assertGreater(len(result['added']),0)
        for x,y,z in known.calls:self.assertGreater(x,0);self.assertLess(abs(y),x);self.assertLess(abs(z),x)
    def test_new_map_snapshot_cells_need_full_remaining_lease(self):
        class Known:
            free={(1,1,1):None,(2,2,2):None,(3,3,3):None}
            def known_free(self,c,*,time_s):
                return c==(1,1,1) or c==(2,2,2) and time_s<.5 or c==(3,3,3) and time_s>2.
        guard=SimpleNamespace(deadline_tick=100,_free_through_deadline=((0,0,0),))
        count=extend_guard_free_snapshot(guard,Known(),global_tick=0,epoch_tick=0)
        self.assertEqual(count,1);self.assertEqual(set(guard._free_through_deadline),{(0,0,0),(1,1,1)})
if __name__=='__main__':unittest.main(verbosity=2)
