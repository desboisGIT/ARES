"""Finite coupled producer demonstration; truth confined to sensor/plant oracle.

No live-guard epoch mutation. Public renewal is returned only if its new family
and future static-map reservation pass. All failed attempts remain in report.
"""
from pathlib import Path
from fractions import Fraction as F
import sys,json,hashlib,time
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[2]
for folder in [ROOT,BASE,HERE,BASE/'coupled_reanchoring_v1',BASE/'bounded_service_v1',BASE/'learned_command_bridge_v1']:sys.path.insert(0,str(folder))
import torch
from producer import PublicTrimWindow,recalibration_candidate,target_from_public_commands
from energy import bootstrap_bound,PublicEnergyLedger
from reanchoring_shield_v1 import ProofBundle,InitialAdmission,ReanchoringShield
from run_coupled import PlantAndOracle,public_initial_packet,ATTESTATION
from bounded_service import WitnessMap,BoundedSupplier
from sensor_renderer import AnalyticRenderer,TruthState
from ares.contracts import load_json
from ares.flight_randomization import sample_flight_profile

def encode(x):
 if isinstance(x,F):return str(x)
 if isinstance(x,(list,tuple)):return [encode(v) for v in x]
 if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
 return x

def run(case=0,steps=400,switch_steps=20):
 torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_flush_denormal(False)
 proof=ProofBundle(ROOT/'reports/phase_4_shield_artifacts/cpu_rpi_certificate_v1');capture=load_json(BASE/'public_trim_capture_v1.json');public=load_json(BASE/'public_trim_history_v1.json');row=InitialAdmission.load(proof,BASE/'public_trim_history_v1.json',case);admission=row.row
 known=WitnessMap.from_dict(load_json(BASE/'bounded_service_v1/known_free_map.json'),time_s=0.);renderer=AnalyticRenderer();mode,seed=capture['profile_modes_and_seeds'][case];profile=sample_flight_profile(load_json(ROOT/'ares/data/configs/drone_sim_v2.json'),seed,mode)
 private=PlantAndOracle(profile,capture['simulation_truth_for_replay_only'],case,renderer,known)
 guard=ReanchoringShield(proof,known,hold_thrust=admission['hold_thrust'],trim_force_interval=admission['trim_force_interval'],motor_residual_interval=admission['motor_residual_interval'],issued_queue=admission['issued_queue'],deadline_tick=1000,service_lease_expires_tick=3000,initial_energy_upper_J=4338,arithmetic_backend_attestation=ATTESTATION,initial_admission=row)
 boot=bootstrap_bound(maximum_issued_thrust=public['maximum_issued_thrust_N'],steps=public['full_rigid_body_steps_per_case'],mass_gravity_upper=proof.mg_upper);ledger=PublicEnergyLedger(boot['energy_upper'],boot['motor_upper'])
 # Public seed command at admitted anchor: use it and the two known queue words
 # as the first recent-command average, never mass/weight or motor truth.
 packet=public_initial_packet(admission);seedpoint=__import__('verified_backup_point_v2').backup_point(packet.p,packet.v,packet.q,admission['anchor'],admission['hold_thrust'])
 target=target_from_public_commands([seedpoint['command'],*admission['issued_queue']])['target_word'];window=PublicTrimWindow(target,proof.source_sha256)
 trace=[];renewal=None;start=time.perf_counter();error=None
 for tick in range(steps):
  # Truth is used only inside this test sensor boundary, never the producer.
  if tick:
   state=private.backend.plant.state[0].tolist();truth=TruthState(tuple(state[:3]),tuple(state[7:10]),tuple(state[3:7]),tuple(state[10:13]));packet=BoundedSupplier(renderer,truth,seed=f'refresh:{case}:{tick}').capture(capture_time_s=float(F(tick,100)),expires_s=30.)
  try:
   candidate=recalibration_candidate(packet,guard.hold,target);decision=guard.decide(packet,tick=tick,proposed_anchor=candidate['anchor']);oldqueue=list(guard.queue)
   before,numeric,issued=private.step(decision['command']);guard.acknowledge_issued(issued);private.update_energy();oracle=private.check(before,guard,proof)
   ledger.advance(issued,oldqueue);counted=window.note_issued(tick=tick,packet=packet,decision=decision,acknowledged_command=issued)
   trace.append(dict(tick=tick,proposal_accepted=decision['proposal_accepted'],small_rate_eligible=candidate['small_rate_window_eligible'],window_count=len(window.records),counted=counted,actual_C_included=oracle['actual_C_included'],physical_collision=oracle['physical_collision'],unknown_entry=oracle['unknown_entry'],public_Eupper=ledger.energy))
   if oracle['physical_collision'] or oracle['unknown_entry'] or not oracle['actual_C_included']:raise RuntimeError('physical/oracle audit failure')
   if len(window.records)>=270:
    state=private.backend.plant.state[0].tolist();truth=TruthState(tuple(state[:3]),tuple(state[7:10]),tuple(state[3:7]),tuple(state[10:13]));final=BoundedSupplier(renderer,truth,seed=f'refresh-final:{case}:{tick+1}').capture(capture_time_s=float(F(tick+1,100)),expires_s=30.)
    window.note_final_packet(tick=tick+1,packet=final);renewal=window.propose(proof=proof,known_map=known,public_energy_upper=ledger.energy,service_lease_expires_tick=3000)
    if renewal['admitted']:break
  except Exception as exc:error=dict(tick=tick,type=type(exc).__name__,message=str(exc));break
  switch_report=None
 if renewal and renewal['admitted'] and switch_steps:
  from epochs import prepare_epoch_switch,rebase_packet
  oldguard=guard;oldtick=guard.tick;oldwords=list(guard.queue)
  prepared=prepare_epoch_switch(oldguard,window,proof=proof,known_map=known,public_energy_upper=ledger.energy,service_lease_expires_tick=3000,current_absolute_packet=final)
  guard=prepared['new_guard'];epoch=prepared['epoch_global_tick'];post=[]
  assert guard.queue==oldwords
  for local in range(switch_steps):
   if local==0:decision=prepared['first_decision']
   else:
    state=private.backend.plant.state[0].tolist();truth=TruthState(tuple(state[:3]),tuple(state[7:10]),tuple(state[3:7]),tuple(state[10:13]));absolute=BoundedSupplier(renderer,truth,seed=f'post-refresh:{case}:{local}').capture(capture_time_s=float(F(epoch+local,100)),expires_s=30.)
    relative=rebase_packet(absolute,global_tick=epoch+local,epoch_global_tick=epoch)
    decision=guard.decide(relative,tick=local,proposed_anchor=guard.anchor)
   oldqueue=list(guard.queue);before,numeric,issued=private.step(decision['command']);guard.acknowledge_issued(issued);private.update_energy();ledger.advance(issued,oldqueue);oracle=private.check(before,guard,proof)
   post.append(dict(global_tick=epoch+local,local_tick=local,actual_C_included=oracle['actual_C_included'],physical_collision=oracle['physical_collision'],unknown_entry=oracle['unknown_entry'],queue_matches_public=oracle['queue_matches_public']))
   assert oracle['actual_C_included'] and not oracle['physical_collision'] and not oracle['unknown_entry'] and oracle['queue_matches_public']
  switch_report=dict(epoch_global_tick=epoch,new_guard_steps=guard.tick,post_switch_steps=post,old_guard_tick_unchanged=oldguard.tick==oldtick,old_guard_queue_unchanged=oldguard.queue==oldwords,no_physical_state_or_motor_or_energy_reset=True,new_public_energy_upper=ledger.energy)
 report=dict(schema_version='ares.renewable_trim_public_producer_demo.v1',case=case,requested_steps=steps,completed_steps=len(trace),target_word=target,bootstrap_public_bounds=boot,all_attempts_retained=True,trace=trace,public_window=dict(target_word=window.target,proof_sha256=window.proof_sha256,records=window.records,final=window.final),renewal=renewal,error=error,no_guard_epoch_switch_performed=switch_report is None,epoch_switch=switch_report,source_sha256={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['producer.py','energy.py','epochs.py','run_producer.py']},controller_or_producer_receives_sampled_mass_mg_motor_energy=False,truth_only_sensor_restore_oracle=True,physical_collisions=sum(r['physical_collision'] for r in trace),unknown_entries=sum(r['unknown_entry'] for r in trace),membership_misses=sum(not r['actual_C_included'] for r in trace),flight_certificate=False,elapsed_s=time.perf_counter()-start)
 (HERE/f'producer_case{case}.json').write_text(json.dumps(encode(report),indent=2)+'\n');print(json.dumps({k:encode(v) for k,v in report.items() if k not in ['trace','renewal','public_window']},indent=2));print('renewal',None if renewal is None else {k:encode(renewal[k]) for k in ['admitted','reason','epoch_tick','deadline_tick'] if k in renewal});return report
if __name__=='__main__':run()
