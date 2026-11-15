from pathlib import Path
from fractions import Fraction as F
from dataclasses import replace,FrozenInstanceError
from types import SimpleNamespace
import sys,json,unittest,random
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'bounded_service_v1'))
from renewable_trim_stream_v3.producer import StreamingTrimWindow
from renewable_trim_stream_v3.trim_stats import TrimStatisticsEvaluator,H
from renewable_trim_stream_v3.snapshot import MapLeaseSnapshot,MapLeaseCache
from renewable_trim_stream_v3.epochs import prepare_epoch_switch
from reanchoring_shield_v1 import ProofBundle,ContractFailure
from bounded_service import WitnessMap,StatePacket
from ares.safety_contract import MeasurementContract
from trim_history_v2 import rk4_trim_bound
import producerwide_v2 as full

CONTRACT=MeasurementContract(position_error_m=.004999999999999999,velocity_error_mps=.009999999999999998,world_ray_error=.001,depth_error_m=.01,capture_time_s=0.,expires_s=30.,supplier='archived_public_SIL',provenance='unit_replay_of_bounded_public_history',source_kind='hard_external',bound_basis='deterministic_guarantee',voxel_model='homogeneous_static_until_expiry',depth_model='first_hit_range')
def packet(raw,tick):return StatePacket(**{n:tuple(raw[n]) for n in ['p','v','q','w']},contract=replace(CONTRACT,capture_time_s=float(tick*H)))
class StreamingTests(unittest.TestCase):
 @classmethod
 def setUpClass(c):
  c.proof=ProofBundle(BASE.parents[1]/'phase_4_shield_artifacts/cpu_rpi_certificate_v1');c.raw=json.loads((BASE/'renewable_trim_v1/producerwide_case0.json').read_text());c.public=c.raw['public_window'];c.target=F(c.public['target_word']);c.known=WitnessMap.from_dict(json.loads((BASE/'bounded_service_v1/known_free_map.json').read_text()),time_s=0.);c.cache=MapLeaseCache(c.known);c.snapshot=c.cache.snapshot(start_tick=0,through_tick=3000)
 def replay(self,duration=1):
  w=StreamingTrimWindow(self.target,self.proof,history_seconds=duration)
  for r in self.public['records']:
   word=tuple(map(F,r['word']));d=dict(conditional_motion_certificate=True,proposal_accepted=True,command_point_error_verified=True,proof_sha256=self.proof.source_sha256,tick=r['tick']-r['decision_epoch_global_tick'],command=word,anchor=r['receipt_anchor'])
   self.assertTrue(w.note_issued(tick=r['tick'],packet=packet(r['packet'],r['tick']),decision=d,acknowledged_command=word,decision_epoch_global_tick=r['decision_epoch_global_tick']))
  f=self.public['final'];p=packet(f['packet'],f['tick']);w.note_final_packet(tick=f['tick'],packet=p);return w,p
 def full_trim(self,values):
  p=self.proof.certificate['physical_parameter_intervals_outward'];return rk4_trim_bound(times=[i*H for i in range(len(values))],measured_vertical_velocity=values,velocity_error=F(1,100),control_period=H,physics_period=F(1,500),hold_thrust=self.target,public_force_upper=F(13,2),mass_lower=F(p['mass'][0]),drag_upper=F(p['drag'][1]),authority_drop_per_period_upper=self.proof.authority_drop,motor_lag_upper=F(p['motor_lag'][1]),transport_upper=F(2000001,100000000),constant_command_settle=F(7,10),stage_up_axis_error=F(1,5000),stage_angular_rate_upper=1,velocity_roundoff_per_period=F(53,100000000),motor_roundoff_per_period=F(1,50000))
 def test_exact_trim_statistics_equivalent_for_all_explicit_durations(self):
  rng=random.Random(842)
  for duration in [F(1,2),F(1),F(3,2),F(2)]:
   e=TrimStatisticsEvaluator(self.proof,self.target,duration)
   for _ in range(8):
    values=[F(rng.randrange(-2000,2001),100000) for i in range(e.intervals+1)];old=self.full_trim(values);new=e.evaluate(start_vz=values[0],end_vz=values[-1],max_abs_vz=max(map(abs,values)))
    for key in new:self.assertEqual(old[key],new[key],key)
  with self.assertRaises(ValueError):TrimStatisticsEvaluator(self.proof,self.target,F(3,4))
 def test_streamed_summary_exactly_matches_full_history(self):
  for duration in [.5,1,1.5,2]:
   w,end=self.replay(duration);s=w.summary();nodes=self.public['records'][-w.history_intervals:];packets=[r['packet'] for r in nodes]+[self.public['final']['packet']];full_domain=full.verify_stage_domain(packets)
   self.assertEqual(s['stage_up_axis_error'],full_domain['stage_up_axis_error']);self.assertEqual(s['stage_and_macro_domains_pass'],full_domain['stage_and_macro_domains_pass']);self.assertEqual(s['max_abs_vz'],max(abs(F.from_float(p['v'][2])) for p in packets));self.assertTrue(w.can_attempt());self.assertEqual(w.record_count,len(self.public['records']))
 def test_snapshot_immutable_nondyadic_endpoints_and_grid_changes(self):
  snapshot=self.cache.snapshot(start_tick=451,through_tick=1451);cell=next(iter(snapshot.free));self.assertTrue(snapshot.known_free(cell,time_s=4.51));self.assertTrue(snapshot.epoch_view(451).known_free(cell,time_s=0.))
  with self.assertRaises(FrozenInstanceError):snapshot.start_tick=0
  old=self.known.origin;self.known.origin=(1.,0.,0.)
  try:
   with self.assertRaises(ValueError):self.cache.snapshot(start_tick=451,through_tick=1451)
   with self.assertRaises(ValueError):self.cache.refresh()
  finally:self.known.origin=old
 def test_cache_and_epoch_constructor_never_revalidate_rays(self):
  old=self.known.known_free;self.known.known_free=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('ray contract revalidated'))
  try:
   self.cache.refresh();snapshot=self.cache.snapshot(start_tick=451,through_tick=1451);view=snapshot.epoch_view(451)
   for cell in snapshot.free:self.assertTrue(view.known_free(cell,time_s=0.));self.assertTrue(view.known_free(cell,time_s=10.))
  finally:self.known.known_free=old
 def proposal(self):
  w,p=self.replay(1);E=F(self.raw['renewal']['public_initial_energy_upper_J']);r=w.propose(proof=self.proof,known_map=self.snapshot,public_energy_upper=E,service_lease_expires_tick=3000);self.assertTrue(r['admitted']);old=SimpleNamespace(pending=None,blocked=False,tick=p.contract.capture_time_s*100,queue=[tuple(map(F,x)) for x in r['issued_queue']],attestation={k:True for k in ['explicit_norm_sqrt_runtime_verified','verified_motor_decay','verified_power_and_battery','guarded_IEEE_basic_arithmetic','same_full_rigid_body_RK4_model','no_GPU_or_compiled_path','exact_current_orientation_feedback']});return w,p,E,r,old
 def test_consume_fresh_proposal_once_without_recompute(self):
  w,p,E,r,old=self.proposal();w.propose=lambda **kw:(_ for _ in ()).throw(AssertionError('full history replayed'))
  a=prepare_epoch_switch(old,w,proof=self.proof,known_map=self.snapshot,public_energy_upper=E,service_lease_expires_tick=3000,current_absolute_packet=p,proposal=r);self.assertTrue(a['first_decision']['conditional_motion_certificate']);self.assertEqual(a['epoch_global_tick'],270);self.assertIsNotNone(a['new_guard'].pending);self.assertIsNone(old.pending)
  with self.assertRaises(ContractFailure):prepare_epoch_switch(old,w,proof=self.proof,known_map=self.snapshot,public_energy_upper=E,service_lease_expires_tick=3000,current_absolute_packet=p,proposal=r)
 def test_stale_energy_packet_map_and_tampered_row_rejected(self):
  for change in ['energy','packet','map','row']:
   w,p,E,r,old=self.proposal();snapshot=self.snapshot
   if change=='energy':E+=F(1,1000)
   if change=='packet':p=replace(p,v=(p.v[0]+.0001,*p.v[1:]))
   if change=='map':snapshot=self.cache.snapshot(start_tick=0,through_tick=2999)
   if change=='row':r['hold_thrust']+=F(1,100)
   with self.assertRaises(ContractFailure):prepare_epoch_switch(old,w,proof=self.proof,known_map=snapshot,public_energy_upper=E,service_lease_expires_tick=3000,current_absolute_packet=p,proposal=r)
 def test_failed_attempts_and_source_rejections_are_retained(self):
  w=StreamingTrimWindow(self.target,self.proof);r=w.propose(proof=self.proof,known_map=self.snapshot,public_energy_upper=4000,service_lease_expires_tick=3000);self.assertFalse(r['admitted']);self.assertEqual(len(w.attempts),1)
  raw=self.public['records'][0];p=packet(raw['packet'],0);bad=replace(p,contract=replace(p.contract,source_kind='gaussian',bound_basis='RMS'))
  self.assertFalse(w.note_issued(tick=0,packet=bad,decision={},acknowledged_command=[float(self.target),0.,0.,0.]));self.assertEqual(w.record_count,0);self.assertEqual(len(w.reset_events),1)
 def test_snapshot_reuse_and_metadata_tampering_rejected(self):
  original=self.cache.snapshot(start_tick=0,through_tick=3000)
  self.assertIs(original,self.cache.snapshot(start_tick=451,through_tick=3000))
  with self.assertRaises(ValueError):replace(original,origin=(1.,0.,0.))
  with self.assertRaises(ValueError):replace(original,free=frozenset())
 def test_stale_queue_service_and_new_endpoint_reset_token(self):
  for change in ['queue','service','history']:
   w,p,E,r,old=self.proposal();service=3000
   if change=='queue':old.queue=[(q[0]+F(1,100),*q[1:]) for q in old.queue]
   if change=='service':service=3001
   if change=='history':w.note_final_packet(tick=270,packet=replace(p,w=(p.w[0]+.0001,*p.w[1:])))
   with self.assertRaises(ContractFailure):prepare_epoch_switch(old,w,proof=self.proof,known_map=self.snapshot,public_energy_upper=E,service_lease_expires_tick=service,current_absolute_packet=p,proposal=r)
if __name__=='__main__':unittest.main()
