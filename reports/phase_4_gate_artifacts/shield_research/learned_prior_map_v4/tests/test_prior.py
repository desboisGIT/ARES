from pathlib import Path
import sys,unittest
BASE=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parents[1]
for p in [BASE,BASE/'bounded_service_v1',BASE/'terminal_reservation_v1',BASE/'learned_mission_bridge_v2',HERE]:sys.path.insert(0,str(p))
from prior_map import observe_prior,generate_pair,route_in_prior
from bounded_service import BoundedSupplier,preflight_scan
from sensor_renderer import TruthState,TestOracle
from scene_v2 import ClutterRenderer
from renewable_trim_stream_v3.snapshot import MapLeaseCache
class PriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer=ClutterRenderer();cls.supplier=BoundedSupplier(cls.renderer,TruthState((6.125,6.125,2.125),(0.,0.,0.)),seed=71)
        cls.packet=cls.supplier.capture(capture_time_s=0.,expires_s=55.)
        cls.old=preflight_scan(cls.supplier,cls.packet,time_s=0.)
        cls.new=observe_prior(cls.supplier,cls.packet,time_s=0.)
        cls.snapshot=MapLeaseCache(cls.new).snapshot(start_tick=0,through_tick=5500)
    def test_only_skip_is_removed_and_blocked_cells_stay_unknown(self):
        self.assertTrue(set(self.old.free)<=set(self.new.free))
        added=set(self.new.free)-set(self.old.free)
        self.assertTrue(added);self.assertTrue(all(cell[0]==28 for cell in added))
        self.assertIn((28,25,8),self.new.free);self.assertNotIn((28,21,8),self.new.free)
        self.assertTrue(TestOracle.free_map_valid(self.renderer,self.new.free))
    def test_seeded_pair_is_independently_public_tube_feasible(self):
        a=[6.,6.,1.98];waypoints,meta=generate_pair(a,90200,self.snapshot)
        self.assertGreaterEqual(meta['leg_lengths_m'][0],.5);self.assertGreaterEqual(meta['leg_lengths_m'][1],.9)
        self.assertTrue(route_in_prior([a,*waypoints],self.snapshot))
        self.assertEqual(waypoints,generate_pair(a,90200,self.snapshot)[0])
if __name__=='__main__':unittest.main(verbosity=2)
