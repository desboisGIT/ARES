from pathlib import Path
from fractions import Fraction as F
import sys,json,unittest,random
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'coupled_reanchoring_v1'));sys.path.insert(0,str(BASE/'bounded_service_v1'))
from producer import *
import energy
from reanchoring_shield_v1 import ProofBundle
from bounded_service_v1.bounded_service import WitnessMap
class RenewableTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.proof=ProofBundle(BASE.parents[1]/'phase_4_shield_artifacts/cpu_rpi_certificate_v1');cls.initial=json.loads((BASE/'initial_contract_v1.json').read_text());cls.capture=json.loads((BASE/'public_trim_history_v1.json').read_text())
 def test_small_rate_discrete_stages_and_zero_case(self):
  r=small_rate_stage_lemma();self.assertLess(r['macro_angular_endpoint_bound'],F(1,100));self.assertLessEqual(max(r['stage_bounds']),F(1,100));self.assertEqual(r['forcing_upper'],F(341,10000))
  z=small_rate_stage_lemma(max_rate=0);old=rate_stage_lemma(rational_domain());self.assertEqual(z['stage_bounds'],old['stage_bounds']);self.assertEqual(z['macro_angular_endpoint_bound'],old['macro_angular_endpoint_bound'])
  with self.assertRaises(ValueError):small_rate_stage_lemma(max_rate=F(1001,1000000))
 def test_real_public_candidates_constant_words_and_noise_domain(self):
  eligible=0
  for row in self.initial['cases']:
   r=recalibration_candidate(row['public_final_packet'],row['hold_thrust'],row['hold_thrust']);self.assertTrue(r['constant_target_word_verified']);self.assertEqual(q(r['command'][0]),q(row['hold_thrust']));self.assertTrue(all(e<=F.from_float(1e-6) for e in r['exact_error']));self.assertTrue(all(q(float(a))==q(a) for a in r['anchor']));eligible+=r['small_rate_window_eligible']
  self.assertGreater(eligible,0)
  for i in range(8):
   packets=[{n:row[n][i] for n in ['p','v','q','w']} for row in self.capture['measured_history']];self.assertTrue(verify_stage_domain(packets)['stage_and_macro_domains_pass'])
 def test_public_mean_word_and_original_primitive_energy_bounds(self):
  words=[[r['hold_thrust'],0,0,0] for r in self.initial['cases']];target=target_from_public_commands(words);self.assertTrue(stored32(target['target_word']))
  rng=random.Random(123);primitive=energy.primitive
  for upper in [F('4.5'),F('5.474'),F('6.5')]:
   b=energy.updater_bound(upper)
   for _ in range(50):
    t=primitive.word(primitive.bits(float(upper)*rng.random()))
    if F.from_float(t)>upper:continue
    for E in [0.,8000.,10000.]:self.assertLessEqual(F.from_float(primitive.energy_step32(E,t))-F.from_float(E),b['energy_jump_upper'])
 def test_tighter_bootstrap_and_ledger_no_energy_or_motor_oracle(self):
  b=energy.bootstrap_bound(maximum_issued_thrust=self.capture['maximum_issued_thrust_N'],steps=self.capture['full_rigid_body_steps_per_case'],mass_gravity_upper=self.proof.mg_upper);self.assertLess(b['energy_upper'],3352);self.assertLess(b['motor_upper'],F('5.474'))
  ledger=energy.PublicEnergyLedger(b['energy_upper'],b['motor_upper']);queue=[[5.,0.,0.,0.]]*2
  for _ in range(100):out=ledger.advance([5.,0.,0.,0.],queue)
  self.assertTrue(out['renewal_energy_admissible']);self.assertLess(out['energy_upper'],3500)
  with self.assertRaises(ValueError):ledger.advance([7.,0.,0.,0.],queue)
 def test_window_requires_true_fresh_receipts_and_complete_history(self):
  hold=self.initial['cases'][0]['hold_thrust'];s=self.initial['cases'][0]['public_final_packet'];window=PublicTrimWindow(hold,self.proof.source_sha256)
  word=[hold,float(energy.primitive.word(energy.primitive.bits(.0001))),0.,0.];receipt=dict(conditional_motion_certificate=True,proposal_accepted=True,command_point_error_verified=True,proof_sha256=self.proof.source_sha256,tick=0,command=word)
  self.assertTrue(window.note_issued(tick=0,packet=s,decision=receipt,acknowledged_command=word));bad=dict(receipt,tick=1,proposal_accepted=False)
  self.assertFalse(window.note_issued(tick=1,packet=s,decision=bad,acknowledged_command=word));self.assertEqual(window.records,[])
  receipt['tick']=2;self.assertTrue(window.note_issued(tick=2,packet=s,decision=receipt,acknowledged_command=word));window.note_final_packet(tick=3,packet=s)
  with self.assertRaises(ValueError):window.propose(proof=self.proof,known_map=None,public_energy_upper=4000,service_lease_expires_tick=3000)
 def test_expired_old_map_rejects_renewed_horizon(self):
  old=WitnessMap.from_dict(json.loads((BASE/'bounded_service_v1/known_free_map.json').read_text()),time_s=0.)
  row=self.initial['cases'][0];r=renewed_reservation(self.proof,old,row['anchor'],tuple(map(F,row['trim_force_interval'])),tick=2700,deadline_tick=3700)
  self.assertFalse(r['reserved'])
 def test_epoch_views_validate_absolute_time_and_shorten_expiry(self):
  from epochs import rebase_packet,EpochMapView
  from run_coupled import public_initial_packet
  from dataclasses import replace
  row=self.initial['cases'][0];packet=public_initial_packet(row);absolute=replace(packet,contract=replace(packet.contract,capture_time_s=2.7,expires_s=30.))
  relative=rebase_packet(absolute,global_tick=270,epoch_global_tick=270)
  self.assertEqual(relative.p,absolute.p);self.assertEqual(relative.contract.capture_time_s,0.)
  self.assertLessEqual(q(relative.contract.expires_s),F(273,10))
  with self.assertRaises(Exception):rebase_packet(absolute,global_tick=271,epoch_global_tick=270)
  known=WitnessMap.from_dict(json.loads((BASE/'bounded_service_v1/known_free_map.json').read_text()),time_s=0.);view=EpochMapView(known,270);cell=next(iter(known.free))
  self.assertTrue(view.known_free(cell,time_s=0.));self.assertFalse(view.known_free(cell,time_s=28.))
 def test_global_receipt_clock_and_energy_renewal_rejection(self):
  hold=self.initial['cases'][0]['hold_thrust'];packet=self.initial['cases'][0]['public_final_packet'];window=PublicTrimWindow(hold,self.proof.source_sha256);word=[hold,0.,0.,0.]
  for i in range(270):
   receipt=dict(conditional_motion_certificate=True,proposal_accepted=True,command_point_error_verified=True,proof_sha256=self.proof.source_sha256,tick=i,command=word,anchor=packet['p'])
   self.assertTrue(window.note_issued(tick=1000+i,packet=packet,decision=receipt,acknowledged_command=word,decision_epoch_global_tick=1000))
  window.note_final_packet(tick=1270,packet=packet);bad=window.propose(proof=self.proof,known_map=None,public_energy_upper=8001,service_lease_expires_tick=3000)
  self.assertFalse(bad['admitted']);self.assertEqual(bad['reason'],'energy_or_service_lease')
 def test_optional_wide_rate_exact_stages_and_sharp_norm(self):
  import producerwide_v2 as wide
  r=wide.small_rate_stage_lemma();self.assertEqual(r['W'],F(1,25));self.assertLess(r['macro_angular_endpoint_bound'],r['W']);self.assertEqual(r['forcing_upper'],F(426,625))
  def f32(x):return energy.primitive.word(energy.primitive.bits(x))
  xyz=list(map(f32,[.0014,-.0014,.0002]));import math
  packet=dict(p=[6.,6.,2.],v=[0.,0.,0.],q=xyz+[f32(math.sqrt(1-sum(x*x for x in xyz)))],w=[.02,-.02,.01])
  domain=wide.verify_stage_domain([packet,packet]);self.assertTrue(domain['stage_and_macro_domains_pass']);self.assertLess(domain['stage_up_axis_error'],F(1,5000));self.assertLess(max(domain['quaternion_stage_norm2_errors']),F(3,100000000))
  narrow=recalibration_candidate(packet,5.,5.);broad=wide.recalibration_candidate(packet,5.,5.)
  self.assertFalse(narrow['small_rate_window_eligible']);self.assertTrue(broad['small_rate_window_eligible']);self.assertTrue(broad['constant_target_word_verified']);self.assertEqual(narrow['command'],broad['command'])
  with self.assertRaises(ValueError):wide.small_rate_stage_lemma(max_rate=F(201,10000))
if __name__=='__main__':unittest.main()
