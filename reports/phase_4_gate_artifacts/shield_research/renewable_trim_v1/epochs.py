"""Explicit epoch-view/switch extension; no old guard or physical state edits."""
from pathlib import Path
from fractions import Fraction as F
from dataclasses import replace
import sys,json,hashlib,math
BASE=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'bounded_service_v1'))
from producer import q,H
from reanchoring_shield_v1 import InitialAdmission,ReanchoringShield,ContractFailure
from ares.safety_contract import validate_contract

def encode(x):
 if isinstance(x,F):return str(x)
 if isinstance(x,(list,tuple)):return [encode(v) for v in x]
 if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
 return x

def rounded(x,upper):
 y=float(x)
 while (q(y)<x if upper else q(y)>x):y=math.nextafter(y,math.inf if upper else -math.inf)
 return y

class EpochMapView:
 def __init__(self,original,epoch_global_tick):
  if type(epoch_global_tick) is not int or epoch_global_tick<0:raise ValueError('nonnegative global tick required')
  self.original=original;self.epoch=epoch_global_tick;self.free=original.free;self.shape=original.shape;self.resolution=original.resolution;self.origin=original.origin
 def known_free(self,cell,*,time_s):
  absolute=self.epoch*H+q(time_s)
  return self.original.known_free(cell,time_s=rounded(absolute,False)) and self.original.known_free(cell,time_s=rounded(absolute,True))

def rebase_packet(packet,*,global_tick,epoch_global_tick):
 if type(global_tick) is not int or type(epoch_global_tick) is not int or not 0<=epoch_global_tick<=global_tick:raise ValueError('ordered nonnegative global ticks required')
 now=float(global_tick*H);c=packet.contract
 if c.capture_time_s!=now or not validate_contract(c,time_s=now).conditional_input_valid:raise ContractFailure('absolute current hard packet required before rebasing')
 relative_expiry=q(c.expires_s)-epoch_global_tick*H
 if relative_expiry<0:raise ContractFailure('absolute hard packet lease expired at new epoch')
 new=replace(c,capture_time_s=float((global_tick-epoch_global_tick)*H),expires_s=rounded(relative_expiry,False),provenance=c.provenance+f'; explicit_clock_origin_global_tick={epoch_global_tick}')
 return replace(packet,contract=new)

class RenewedAdmission(InitialAdmission):
 @classmethod
 def from_window(cls,window,*,proof,known_map,public_energy_upper,service_lease_expires_tick):
  row=window.propose(proof=proof,known_map=known_map,public_energy_upper=public_energy_upper,service_lease_expires_tick=service_lease_expires_tick)
  if not row['admitted']:raise ContractFailure('new public-history family/reserve not admitted: '+row['reason'])
  history=json.dumps(encode(dict(records=window.records,final=window.final)),sort_keys=True,separators=(',',':')).encode()
  result=cls();result.row=row;result.public_history_sha256=hashlib.sha256(history).hexdigest();result.mathematical_proof_sha256=proof.source_sha256
  return result

def prepare_epoch_switch(old_guard,window,*,proof,known_map,public_energy_upper,service_lease_expires_tick,current_absolute_packet,old_epoch_global_tick=0):
 """Return new guard + already-pending first decision; old guard untouched.

 Caller must issue first_decision.command once and acknowledge it on new_guard,
 then replace its active reference. Global policy/sensor clocks never reset.
 No physical state, motor, transport queue or energy updater is reset here.
 """
 if old_guard.pending is not None or old_guard.blocked:raise ContractFailure('old command transaction must finish before renewal')
 if window.final is None:raise ContractFailure('final post-ack packet unavailable')
 epoch=window.final['tick']
 if old_guard.tick+old_epoch_global_tick!=epoch:raise ContractFailure('old guard and global history clocks disagree')
 a=RenewedAdmission.from_window(window,proof=proof,known_map=known_map,public_energy_upper=public_energy_upper,service_lease_expires_tick=service_lease_expires_tick);row=a.row
 if old_guard.queue!=[tuple(map(q,w)) for w in row['issued_queue']]:raise ContractFailure('public queue continuity lost at renewal')
 if any(tuple(getattr(current_absolute_packet,n))!=tuple(row['public_final_packet'][n]) for n in ['p','v','q','w']):raise ContractFailure('switch packet differs from admitted final public packet')
 packet=rebase_packet(current_absolute_packet,global_tick=epoch,epoch_global_tick=epoch);view=EpochMapView(known_map,epoch)
 guard=ReanchoringShield(proof,view,hold_thrust=row['hold_thrust'],trim_force_interval=row['trim_force_interval'],motor_residual_interval=row['motor_residual_interval'],issued_queue=row['issued_queue'],deadline_tick=1000,service_lease_expires_tick=service_lease_expires_tick-epoch,initial_energy_upper_J=row['public_initial_energy_upper_J'],arithmetic_backend_attestation=old_guard.attestation,initial_admission=a)
 decision=guard.decide(packet,tick=0,proposed_anchor=row['anchor'])
 if not decision['conditional_motion_certificate'] or not decision['proposal_accepted']:raise ContractFailure('new epoch first backup not admitted')
 return dict(new_guard=guard,first_decision=decision,rebased_current_packet=packet,epoch_global_tick=epoch,new_admission=a,old_guard_unchanged=True,physical_state_reset=False,public_queue_continuity=True,first_command_pending=True,flight_certificate=False)
