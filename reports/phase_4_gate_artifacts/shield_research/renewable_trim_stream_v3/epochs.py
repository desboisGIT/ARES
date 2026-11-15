"""Consume one fresh streaming proposal; no old epoch/plant edits."""
from fractions import Fraction as F
from dataclasses import replace
from pathlib import Path
import sys,copy,time
BASE=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE))
from reanchoring_shield_v1 import InitialAdmission,ReanchoringShield,ContractFailure
from zero_rate_history_v1 import q
from ares.safety_contract import validate_contract
from .producer import digest,snapshot_packet
from .trim_stats import H
from .snapshot import rounded,MapLeaseSnapshot

def rebase_packet(packet,*,global_tick,epoch_global_tick):
 if type(global_tick) is not int or type(epoch_global_tick) is not int or not 0<=epoch_global_tick<=global_tick:raise ValueError('ordered nonnegative global ticks required')
 now=float(global_tick*H);c=packet.contract
 if c.capture_time_s!=now or not validate_contract(c,time_s=now).conditional_input_valid:raise ContractFailure('absolute current hard packet required before rebasing')
 expiry=q(c.expires_s)-epoch_global_tick*H
 if expiry<0:raise ContractFailure('absolute lease expired')
 return replace(packet,contract=replace(c,capture_time_s=float((global_tick-epoch_global_tick)*H),expires_s=rounded(expiry,False),provenance=c.provenance+f'; explicit_clock_origin_global_tick={epoch_global_tick}'))

class StreamingAdmission(InitialAdmission):pass

def prepare_epoch_switch(old_guard,window,*,proof,known_map,public_energy_upper,service_lease_expires_tick,current_absolute_packet,old_epoch_global_tick=0,proposal):
 started=time.perf_counter()
 if old_guard.pending is not None or old_guard.blocked:raise ContractFailure('old transaction must finish before renewal')
 if not isinstance(known_map,MapLeaseSnapshot):raise ContractFailure('immutable map snapshot required')
 cache=window._cached
 if cache is None:raise ContractFailure('no validated streaming proposal cached')
 token,binding,cached,row_hash=cache
 if token in window._consumed or proposal.get('binding_token')!=token:raise ContractFailure('missing/consumed proposal token')
 supplied={k:v for k,v in proposal.items() if k!='proposal_latency_s'}
 if digest(supplied)!=row_hash:raise ContractFailure('proposal data changed after validation')
 current=window._binding(known_map=known_map,energy=public_energy_upper,service=service_lease_expires_tick)
 if current!=binding or proof.source_sha256!=window.proof_sha256:raise ContractFailure('proposal is stale after history/energy/map/service change')
 epoch=cached['epoch_tick']
 if old_guard.tick+old_epoch_global_tick!=epoch:raise ContractFailure('old/global clocks disagree')
 raw,pdigest=snapshot_packet(current_absolute_packet,epoch)
 if pdigest!=binding['endpoint_digest']:raise ContractFailure('fresh endpoint packet differs from validated endpoint')
 if old_guard.queue!=[tuple(map(q,w)) for w in cached['issued_queue']]:raise ContractFailure('public queue continuity lost')
 if not cached['admitted'] or not known_map.covers(epoch,cached['deadline_tick']):raise ContractFailure('new family/reservation or lease invalid')
 admission=StreamingAdmission();admission.row=copy.deepcopy(cached);admission.public_history_sha256=digest(dict(receipts=binding['receipt_chain_hash'],endpoint=pdigest,source_manifest=window.manifest_digest));admission.mathematical_proof_sha256=proof.source_sha256
 packet=rebase_packet(current_absolute_packet,global_tick=epoch,epoch_global_tick=epoch);view=known_map.epoch_view(epoch)
 guard=ReanchoringShield(proof,view,hold_thrust=cached['hold_thrust'],trim_force_interval=cached['trim_force_interval'],motor_residual_interval=cached['motor_residual_interval'],issued_queue=cached['issued_queue'],deadline_tick=1000,service_lease_expires_tick=service_lease_expires_tick-epoch,initial_energy_upper_J=cached['public_initial_energy_upper_J'],arithmetic_backend_attestation=old_guard.attestation,initial_admission=admission)
 decision=guard.decide(packet,tick=0,proposed_anchor=cached['anchor'])
 if not decision['conditional_motion_certificate'] or not decision['proposal_accepted']:raise ContractFailure('fresh new-epoch first decision rejected')
 window._consumed.add(token)
 return dict(new_guard=guard,first_decision=decision,rebased_current_packet=packet,epoch_global_tick=epoch,new_admission=admission,old_guard_unchanged=True,physical_state_reset=False,public_queue_continuity=True,first_command_pending=True,consumed_binding_token=token,prepare_latency_s=time.perf_counter()-started,flight_certificate=False)
