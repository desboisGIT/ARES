"""Truth is used only by simulated sensor backends and this test oracle."""
from dataclasses import asdict,replace
from fractions import Fraction as F
from itertools import product
import math
import unittest

from ares.safety_contract import robust_free_witness,validate_contract
from bounded_service import BoundedSupplier,RayPacket,WitnessMap,preflight_scan
from sensor_renderer import AnalyticRenderer,TruthState,TestOracle,exact,noisy_vector

ANCHOR=(6.125,6.125,2.125)
SIGNS=list(product((-1,1),repeat=3))


class BoundedServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer=AnalyticRenderer()
        cls.truth=TruthState(ANCHOR,(.125,-.25,.0625),angular_velocity=(.1,-.2,.3))
        cls.supplier=BoundedSupplier(cls.renderer,cls.truth,seed=71)
        cls.packet=cls.supplier.capture(capture_time_s=0.,expires_s=30.)
        cls.map=preflight_scan(cls.supplier,cls.packet,time_s=.1)

    def test_seeded_packets_nonzero_norm_bounded_and_no_plant_fields(self):
        other=BoundedSupplier(self.renderer,self.truth,seed=71).capture(capture_time_s=0.,expires_s=30.)
        self.assertEqual(other,self.packet)
        self.assertTrue(TestOracle.state_error_valid(self.truth,self.packet))
        self.assertNotEqual(self.packet.p,self.truth.position)
        self.assertNotEqual(self.packet.v,self.truth.velocity)
        self.assertNotEqual(self.packet.w,self.truth.angular_velocity)
        self.assertEqual(set(asdict(self.packet)),{'p','v','q','w','contract','angular_velocity_error_radps','orientation_contract'})
        self.assertNotIn('mass',str(asdict(self.packet)))
        box=self.packet.body_rate_box(time_s=.1)
        self.assertTrue(all(lo<=v<=hi for lo,v,hi in zip(box['lower'],self.truth.angular_velocity,box['upper'])))

    def test_orientation_contract_required_and_raw_quaternion_preserved(self):
        self.assertEqual(self.packet.orientation_contract,'exact_current_raw_quaternion')
        self.assertEqual(self.packet.q,self.truth.quaternion)
        negative=replace(self.packet,q=tuple(-v for v in self.packet.q))
        self.assertEqual(negative.q,tuple(-v for v in self.packet.q))
        for tag in ('estimated_quaternion','normalized_quaternion','',None):
            with self.subTest(tag=tag),self.assertRaises(ValueError):
                replace(self.packet,orientation_contract=tag)
        saved=self.map.to_dict();saved['known_free']=saved['known_free'][:1]
        saved['known_free'][0]['packet']['state']['orientation_contract']='estimated_quaternion'
        with self.assertRaises(ValueError):WitnessMap.from_dict(saved,time_s=.1)
        del saved['known_free'][0]['packet']['state']['orientation_contract']
        with self.assertRaises(ValueError):WitnessMap.from_dict(saved,time_s=.1)

    def test_advertised_state_radii_fit_exact_rational_proof_bounds(self):
        for advertised,decimal,proof in ((self.packet.contract.position_error_m,.005,F(1,200)),
                                        (self.packet.contract.velocity_error_mps,.01,F(1,100)),
                                        (self.packet.angular_velocity_error_radps,.001,F(1,1000))):
            self.assertEqual(advertised,math.nextafter(decimal,0.))
            self.assertLessEqual(F.from_float(advertised),proof)
            self.assertGreater(F.from_float(decimal),proof)
        self.assertTrue(TestOracle.state_error_valid(self.truth,self.packet))

    def test_all_independent_state_ray_and_depth_error_corners(self):
        # 8×8×8×2=1024 signed state/ray/depth cases, plus independent w corners.
        for pc,vc,rc,sign in product(SIGNS,SIGNS,SIGNS,(-1,1)):
            packet=self.supplier.capture(capture_time_s=0.,expires_s=30.,position_corner=pc,velocity_corner=vc,
                                         angular_velocity_corner=rc)
            self.assertTrue(TestOracle.state_error_valid(self.truth,packet))
            direction=tuple(a-b for a,b in zip((6.625,6.625,2.625),packet.p))
            ray=self.supplier.ray(packet,direction,ray_id=19,ray_corner=rc,depth_sign=sign)
            witness=robust_free_witness(packet.p,direction,ray.measured_depth,
                (6.5,6.5,2.5),(6.75,6.75,2.75),1.,packet.contract,time_s=.1)
            self.assertTrue(witness.free_witness)
            actual_direction=noisy_vector(direction,.001,'71:19:ray',rc)
            hit=self.renderer.first_hit(self.truth.position,actual_direction)
            self.assertLessEqual(abs(exact(ray.measured_depth)-hit),exact(.01))
            self.assertLessEqual(sum((exact(a)-exact(b))**2 for a,b in zip(actual_direction,direction)),exact(.001)**2)
            self.assertFalse(witness.flight_certificate)

    def test_map_derived_from_rays_is_valid_and_reserves_neighborhood(self):
        self.assertGreater(len(self.map.free),100)
        self.assertTrue(TestOracle.free_map_valid(self.renderer,self.map.free))
        radius=.17+.15+.005 # Full profile max body radius + margin + ep outer ball.
        result=self.map.reservation(self.packet.p,radius,time_s=.1)
        self.assertTrue(result['conditional_reservation_free'])
        self.assertGreaterEqual(result['required_cell_count'],27)
        self.assertFalse(result['flight_certificate'])
        for cell,(ray,witness,s) in self.map.free.items():
            self.assertTrue(witness.free_witness)
            self.assertTrue(ray.state.contract.capture_time_s==0.)
            self.assertFalse(witness.flight_certificate)

    def test_public_map_reload_rechecks_witness_and_expiry(self):
        saved=self.map.to_dict()
        rebuilt=WitnessMap.from_dict(saved,time_s=.1)
        self.assertEqual(set(rebuilt.free),set(self.map.free))
        self.assertFalse(WitnessMap.from_dict(saved,time_s=31.).free)
        self.assertFalse(self.packet.body_rate_box(time_s=31.)['conditional_input_valid'])

    def test_closed_left_face_contact_rejects_unknown_occupied_and_support(self):
        center=(6.25,6.125,2.125);radius=.25
        observed=WitnessMap();observed.free=dict(self.map.free)
        self.assertTrue(observed.reservation(center,radius,time_s=.1)['conditional_reservation_free'])
        del observed.free[(23,24,8)] # Left cell ends exactly at x=6, one-point sphere contact.
        result=observed.reservation(center,radius,time_s=.1)
        self.assertFalse(result['conditional_reservation_free'])
        self.assertIn((23,24,8),result['missing_cells'])
        cropped=WitnessMap(shape=(24,48,16),origin=(6.,0.,0.))
        cropped.free={(cell[0]-24,*cell[1:]):entry for cell,entry in self.map.free.items() if cell[0]>=24}
        result=cropped.reservation(center,radius,time_s=.1)
        self.assertFalse(result['conditional_reservation_free'])
        self.assertIn((-1,24,8),result['missing_cells'])
        supplier=BoundedSupplier(self.renderer,TruthState((.625,6.125,2.125),(0.,0.,0.)),seed=19)
        packet=supplier.capture(capture_time_s=0.,expires_s=30.)
        near_wall=preflight_scan(supplier,packet,time_s=.1,half_width_cells=2)
        result=near_wall.reservation((.5,6.125,2.125),.25,time_s=.1)
        self.assertFalse(result['conditional_reservation_free'])
        self.assertIn((0,24,8),result['missing_cells']) # Occupied wall upper face x=.25.

    def test_invalid_packet_fields_and_deserialized_rate_bounds_reject(self):
        for value in (-.001,True,float('nan'),float('inf'),'0.001',None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                replace(self.packet,angular_velocity_error_radps=value)
        for field,values in {'w':[(0.,0.),(0.,False,0.),(0.,float('inf'),0.),(0.,'x',0.)],
                             'q':[(0.,0.,0.),(0.,0.,0.,0.),(0.,False,0.,1.),
                                  (0.,0.,float('nan'),1.),(0.,0.,0.,2.)]}.items():
            for value in values:
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    replace(self.packet,**{field:value})
        self.assertEqual(replace(self.packet,q=(0.,0.,0.,-1.)).q,(0.,0.,0.,-1.))
        saved=self.map.to_dict();saved['known_free']=saved['known_free'][:1]
        saved['known_free'][0]['packet']['state']['angular_velocity_error_radps']=-.001
        with self.assertRaises(ValueError):WitnessMap.from_dict(saved,time_s=.1)
        huge=replace(self.packet,w=(1e308,0.,0.),angular_velocity_error_radps=1e308)
        self.assertFalse(huge.body_rate_box(time_s=.1)['conditional_input_valid'])

    def test_invalid_grid_metadata_rejects(self):
        for options in ({'shape':(1,2)},{'shape':(1,2,0)},{'shape':(1,2,True)},
                        {'shape':(1.,2,3)},{'resolution':0.},{'resolution':-.25},
                        {'resolution':float('inf')},{'resolution':True},{'resolution':'.25'},
                        {'origin':(0.,0.)},{'origin':(0.,False,0.)},
                        {'origin':(0.,float('nan'),0.)},{'origin':(1e308,0.,0.)}):
            with self.subTest(options=options),self.assertRaises(ValueError):WitnessMap(**options)
        with self.assertRaises(ValueError):self.map.reservation((0.,False,0.),.25,time_s=.1)
        with self.assertRaises(ValueError):self.map.reservation((0.,0.,0.),True,time_s=.1)

    def test_unknown_and_occupied_targets_are_safely_rejected(self):
        self.assertFalse(self.map.known_free((28,24,8),time_s=.1))
        self.assertFalse(self.map.reservation((7.125,6.125,2.125),.32,time_s=.1)['conditional_reservation_free'])
        for cell in ((0,24,8),(47,24,8),(36,24,8),(40,24,8)):
            self.assertNotIn(cell,self.map.free)
            self.assertIn(cell,self.map.rejections)
        self.assertFalse(self.map.reservation((9.125,6.125,2.125),.32,time_s=.1)['conditional_reservation_free'])

    def test_expired_gaussian_and_unsupported_contracts_reject(self):
        self.assertTrue(validate_contract(self.packet.contract,time_s=30.).conditional_input_valid)
        self.assertFalse(validate_contract(self.packet.contract,time_s=math.nextafter(30.,math.inf)).conditional_input_valid)
        self.assertFalse(self.map.reservation(self.packet.p,.32,time_s=31.)['conditional_reservation_free'])
        for changes in ({'source_kind':'gaussian'},{'bound_basis':'confidence_interval'},
                        {'coordinate_frame':'body'},{'voxel_model':'partly_occupied'},
                        {'depth_model':'not_first_hit'}):
            packet=replace(self.packet,contract=replace(self.packet.contract,**changes))
            witness=robust_free_witness(packet.p,(1.,0.,0.),2.,(6.5,6.,2.),(6.75,6.25,2.25),.5,
                                        packet.contract,time_s=.1)
            self.assertFalse(witness.conditional_input_valid)
            self.assertFalse(witness.free_witness)
        self.assertFalse(self.packet.boxes(time_s=-.1).conditional_input_valid)
        self.assertTrue(self.packet.boxes(time_s=.1).conditional_input_valid)
        with self.assertRaises(ValueError):
            self.supplier.ray(replace(self.packet,p=(0.,0.,0.)),(1.,0.,0.),ray_id=1)

    def test_physical_3d_variations_with_full_profile_plant(self):
        import torch
        from ares.contracts import load_json
        from ares.diffaero_adapter import DiffAeroAdapter
        from ares.flight_randomization import sample_flight_profile
        from pathlib import Path
        root=Path(__file__).resolve().parents[4]
        profile=load_json(root/'ares/data/configs/drone_sim_v2.json')
        profiles=[sample_flight_profile(profile,410+i,('low','high','random')[i%3]) for i in range(8)]
        p=[];v=[];q=[];w=[]
        for i,sign in enumerate(SIGNS):
            p.append([a+s*.0625 for a,s in zip(ANCHOR,sign)])
            v.append([s*.125 for s in sign]);w.append([s*.025 for s in sign])
            roll,pitch,yaw=[math.radians(s*a) for s,a in zip(sign,(5,3,7))]
            sr,cr,sp,cp,sy,cy=math.sin(roll/2),math.cos(roll/2),math.sin(pitch/2),math.cos(pitch/2),math.sin(yaw/2),math.cos(yaw/2)
            q.append([sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy])
        plant=DiffAeroAdapter(profiles,device='cpu',dt=.01,integration_dt=.002)
        plant.reset(position=p,velocity=v,quaternion=q,angular_velocity=w)
        with torch.no_grad():plant.step(torch.tensor([[.5*9.80665,0.,0.,0.]]*8))
        for i in range(8):
            truth=TruthState(tuple(plant.position[i].tolist()),tuple(plant.velocity[i].tolist()),
                             tuple(plant.quaternion[i].tolist()),tuple(plant.angular_velocity[i].tolist()))
            supplier=BoundedSupplier(self.renderer,truth,seed=410+i)
            packet=supplier.capture(capture_time_s=.01,expires_s=.11)
            self.assertTrue(TestOracle.state_error_valid(truth,packet))
            self.assertEqual(packet.q,truth.quaternion)
            self.assertTrue(packet.boxes(time_s=.02).conditional_input_valid)
            direction=tuple(a-b for a,b in zip((6.625,6.625,2.625),packet.p))
            ray=supplier.ray(packet,direction,ray_id=0)
            self.assertGreater(ray.measured_depth,1.)


if __name__=='__main__':unittest.main(verbosity=2)
