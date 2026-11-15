"""Streaming exact public-history trim, with endpoint-bound renewal proposals."""
from pathlib import Path
from fractions import Fraction as F
from collections import deque
from dataclasses import dataclass,asdict
from functools import lru_cache
from types import MappingProxyType
import sys,json,hashlib,copy,time,math
BASE=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'renewable_trim_v1'));sys.path.insert(0,str(BASE/'terminal_reservation_v1'))
from zero_rate_history_v1 import q
from current_terminal_membership_v1 import current_terminal_membership
from ares.safety_contract import validate_contract
from reservation import free_grid_reserve
from producerwide_v2 import recalibration_candidate,target_from_public_commands,stored32
from energy import ceil_grid
from .trim_stats import TrimStatisticsEvaluator,domain_constants,H
from .snapshot import MapLeaseSnapshot,canonical

KEYS=('qx','qy','qz','abs_vz','p_outer','v_outer','w_outer')
def digest(data):return hashlib.sha256(json.dumps(canonical(data),sort_keys=True,separators=(',',':')).encode()).hexdigest()
@lru_cache(maxsize=1)
def source_manifest():
 paths=[BASE/'verified_backup_point_v2.py',BASE/'trim_history_v2.py',BASE/'current_terminal_membership_v1.py',BASE/'renewable_trim_v1/producerwide_v2.py',BASE/'renewable_trim_v1/energy.py']+list(Path(__file__).parent.glob('*.py'))
 return MappingProxyType({str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

def snapshot_packet(packet,tick):
 if type(tick) is not int or tick<0 or not hasattr(packet,'contract'):raise ValueError('absolute typed hard packet and nonnegative global tick required')
 now=float(tick*H);c=packet.contract;status=validate_contract(c,time_s=now)
 if (not status.conditional_input_valid or c.capture_time_s!=now or q(c.position_error_m)>F(1,200) or q(c.velocity_error_mps)>F(1,100) or q(packet.angular_velocity_error_radps)>F(1,1000) or packet.orientation_contract!='exact_current_raw_quaternion'):raise ValueError('fresh deterministic packet source/feedback bound failed')
 raw=dict(p=tuple(packet.p),v=tuple(packet.v),q=tuple(packet.q),w=tuple(packet.w),contract=asdict(c),angular_velocity_error_radps=packet.angular_velocity_error_radps,orientation_contract=packet.orientation_contract)
 return raw,digest(raw)

def packet_stats(raw):
 d,rate,norm_error,coordinate_drift,_=domain_constants();ep,ev,ew=F(1,200),F(1,100),F(1,1000);shell=F(8,2**24)
 quat=tuple(map(q,raw['q']));n2=sum(x*x for x in quat);valid=quat[3]>0 and (1-shell)**2<=n2<=(1+shell)**2
 p,v,w=[tuple(map(q,raw[n])) for n in ['p','v','w']]
 for j in range(3):
  vd=d['raw13_real_ranges'][7+j];radius=q(d['projective_qxyz_radii'][j]);valid &= abs(p[j])+ep<=16 and q(vd['lo'])<=v[j]-ev<=v[j]+ev<=q(vd['hi']) and abs(w[j])+ew<=rate['W'] and quat[j]**2<=radius**2*n2
 stats=dict(qx=abs(quat[0])/(1-shell),qy=abs(quat[1])/(1-shell),qz=abs(quat[2])/(1-shell),abs_vz=abs(v[2]),p_outer=max(map(abs,p))+ep,v_outer=max(map(abs,v))+ev,w_outer=max(map(abs,w))+ew)
 return bool(valid),v[2],stats

@dataclass(frozen=True)
class Node:
 tick:int
 packet_json:str
 packet_digest:str
 vz:F
 stats:tuple
 domain_valid:bool
 word:tuple
 receipt_digest:str
 def packet(self):return json.loads(self.packet_json)
 def public(self):return dict(tick=self.tick,packet=self.packet(),word=self.word,receipt_digest=self.receipt_digest,packet_digest=self.packet_digest)

class StreamingTrimWindow:
 def __init__(self,target_word,proof,history_seconds=F(1)):
  self.target=q(target_word);self.proof_sha256=proof.source_sha256
  if not stored32(self.target) or not 4<=self.target<=F(13,2):raise ValueError('valid stored constant trim target required')
  self.evaluator=TrimStatisticsEvaluator(proof,self.target,history_seconds);self.history_seconds=self.evaluator.duration;self.history_intervals=self.evaluator.intervals;self.required_intervals=70+self.history_intervals
  self.manifest=source_manifest();self.manifest_digest=digest(dict(self.manifest));self.attempts=[];self.reset_events=[];self._chain=hashlib.sha256(b'ares.streaming.public.receipts.v3').hexdigest();self._serial=0;self._cached=None;self._consumed=set();self._clear_state()
 def _clear_state(self):
  self._nodes=[];self._history=deque();self._max={k:deque() for k in KEYS};self._bad=deque();self._final=None;self._final_raw=None;self._final_digest=None;self._cached=None
 @property
 def record_count(self):return len(self._nodes)
 @property
 def records(self):return tuple(n.public() for n in self._nodes)
 @property
 def final(self):return None if self._final is None else dict(tick=self._final.tick,packet={k:self._final_raw[k] for k in ['p','v','q','w']})
 @property
 def receipt_chain_hash(self):return self._chain
 def clear(self,reason='explicit_reset',tick=None):
  event=dict(reason=reason,tick=tick,previous_receipt_chain=self._chain,accepted_intervals=len(self._nodes));self.reset_events.append(event);self._chain=digest(event);self._clear_state()
 def note_issued(self,*,tick,packet,decision,acknowledged_command,decision_epoch_global_tick=0):
  started=time.perf_counter()
  try:
   word=tuple(map(q,acknowledged_command));raw,pdigest=snapshot_packet(packet,tick)
   okay=(len(word)==4 and all(stored32(x) for x in word) and word[0]==self.target and max(map(abs,word[1:]))<=F(1,50) and decision.get('conditional_motion_certificate') is True and decision.get('proposal_accepted') is True and decision.get('command_point_error_verified') is True and decision.get('proof_sha256')==self.proof_sha256 and type(decision_epoch_global_tick) is int and decision_epoch_global_tick>=0 and decision.get('tick')==tick-decision_epoch_global_tick and tuple(map(q,decision.get('command',())))==word)
   if self._nodes and tick!=self._nodes[-1].tick+1:okay=False
   if not okay:raise ValueError('receipt_rejected_fallback_word_rate_or_clock')
   valid,vz,stats=packet_stats(raw);receipt=dict(global_tick=tick,epoch=decision_epoch_global_tick,proof_sha256=self.proof_sha256,anchor=decision.get('anchor'),word=word,packet_digest=pdigest);rdigest=digest(receipt)
   self._chain=digest(dict(previous=self._chain,receipt=rdigest));node=Node(tick,json.dumps(canonical(raw),sort_keys=True,separators=(',',':')),pdigest,vz,tuple(stats[k] for k in KEYS),valid,word,rdigest);self._nodes.append(node);self._history.append(node);cut=tick-self.history_intervals+1
   for k,value in stats.items():
    queue=self._max[k]
    while queue and queue[-1][1]<=value:queue.pop()
    queue.append((tick,value))
    while queue and queue[0][0]<cut:queue.popleft()
   if not valid:self._bad.append(tick)
   while self._bad and self._bad[0]<cut:self._bad.popleft()
   while self._history and self._history[0].tick<cut:self._history.popleft()
   self._final=None;self._final_raw=None;self._final_digest=None;self._cached=None;self.last_note_latency_s=time.perf_counter()-started;return True
  except (ValueError,TypeError,OverflowError) as error:
   self.clear(str(error),tick);self.last_note_latency_s=time.perf_counter()-started;return False
 def note_final_packet(self,*,tick,packet):
  if not self._nodes or tick!=self._nodes[-1].tick+1:raise ValueError('fresh next-tick endpoint required')
  raw,pdigest=snapshot_packet(packet,tick);valid,vz,stats=packet_stats(raw);self._final=Node(tick,'',pdigest,vz,tuple(stats[k] for k in KEYS),valid,(),pdigest);self._final_raw=raw;self._final_digest=pdigest;self._cached=None
 def can_attempt(self):return self._final is not None and len(self._nodes)>=self.required_intervals and len(self._history)==self.history_intervals
 def summary(self):
  if not self.can_attempt():raise ValueError('insufficient explicit-duration history')
  final=dict(zip(KEYS,self._final.stats));maxima={k:max(queue[0][1],final[k]) for k,queue in self._max.items()};_,_,norm_error,drift,_=domain_constants();up=norm_error+2*(maxima['qx']+drift)**2+2*(maxima['qy']+drift)**2
  return dict(start_vz=self._history[0].vz,end_vz=self._final.vz,max_abs_vz=maxima['abs_vz'],maxima=maxima,stage_up_axis_error=up,stage_and_macro_domains_pass=not self._bad and self._final.domain_valid and up<=F(1,5000),history_duration=self.history_seconds,maximum_packet_gap=H,history_intervals=self.history_intervals,packet_count=self.history_intervals+1,receipt_chain_hash=self._chain)
 def _binding(self,*,known_map,energy,service):
  return dict(endpoint_tick=self._final.tick,endpoint_digest=self._final_digest,receipt_chain_hash=self._chain,queue=[self._nodes[-1].word,self._nodes[-2].word],energy=q(energy),service=service,map_digest=known_map.digest,proof=self.proof_sha256,target=self.target,history_seconds=self.history_seconds,source_manifest=self.manifest_digest)
 def propose(self,*,proof,known_map,public_energy_upper,service_lease_expires_tick):
  started=time.perf_counter();self._serial+=1;row=dict(admitted=False,reason='insufficient_history',history_seconds=self.history_seconds,history_intervals=self.history_intervals,attempt_index=self._serial,receipt_chain_hash=self._chain,flight_certificate=False)
  if self.can_attempt():
   if not isinstance(known_map,MapLeaseSnapshot):raise TypeError('prevalidated immutable MapLeaseSnapshot required')
   E=q(public_energy_upper);tick=self._final.tick;deadline=tick+1000
   if proof.source_sha256!=self.proof_sha256:raise ValueError('proof source changed')
   binding=self._binding(known_map=known_map,energy=E,service=service_lease_expires_tick);token=digest(dict(binding=binding,attempt=self._serial));row.update(epoch_tick=tick,deadline_tick=deadline,binding_token=token,binding=binding,map_snapshot_digest=known_map.digest)
   summary=self.summary();row['domain_check']=summary
   if not 0<=E<=8000 or service_lease_expires_tick<deadline or q(self._final_raw['contract']['expires_s'])<deadline*H or not known_map.covers(tick,deadline):row['reason']='energy_or_source_or_map_lease'
   elif not summary['stage_and_macro_domains_pass'] or not self.evaluator.target_domains_pass:row['reason']='stage_or_motor_or_word_numeric_domain'
   else:
    trim=self.evaluator.evaluate(start_vz=summary['start_vz'],end_vz=summary['end_vz'],max_abs_vz=summary['max_abs_vz']);force=ceil_grid(self.evaluator.mass_hi/F(19,20)*trim['conditional_trim_acceleration_bound']);motor=self.evaluator.motor_dyadic;last={k:self._final_raw[k] for k in ['p','v','q','w']};anchor=last['p'];queue=[self._nodes[-1].word,self._nodes[-2].word]
    membership=current_terminal_membership(**last,anchor=anchor,hold_thrust=self.target,issued_queue=queue,trim_force_interval=(-force,force),motor_residual_interval=(-motor,motor),shapes=proof.shapes);reserve=reserve_snapshot(proof,known_map,anchor,(-force,force),tick,deadline) if membership['current_set_included'] else dict(reserved=False,reason='membership')
    admitted=bool(membership['current_set_included'] and reserve['reserved']);row.update(admitted=admitted,reason='renewal_admitted' if admitted else 'current_family_or_reserve',hold_thrust=self.target,anchor=anchor,public_final_packet=last,trim_force_interval=(-force,force),motor_residual_interval=(-motor,motor),issued_queue=queue,public_initial_energy_upper_J=E,membership=membership,reservation=reserve,trim_lemma=trim,mathematical_proof_sha256=proof.source_sha256,attempted_C_channel_maxima={k:v['maximum'] for k,v in membership['channels'].items()})
   if row['admitted']:self._cached=(token,binding,copy.deepcopy(row),digest(row))
  row['proposal_latency_s']=time.perf_counter()-started;self.attempts.append(copy.deepcopy(row));return row

def reserve_snapshot(proof,snapshot,anchor,trim,tick,deadline):
 if not snapshot.covers(tick,deadline):return dict(reserved=False,reason='snapshot_lease')
 drift=proof.mg_upper*proof.authority_drop*(deadline-tick)/F(19,20)**2;loD,hiD=q(trim[0])-drift,q(trim[1]);a=list(map(q,anchor));rx=proof.radii['planar_x'][0];rz=proof.radii['vertical'][0]
 lower=[a[0]-rx,a[1]-rx,a[2]+loD/F(3,4)-rz];upper=[a[0]+rx,a[1]+rx,a[2]+hiD/F(3,4)+rz]
 if any(lo-F(1,200)<-16 or hi+F(1,200)>16 for lo,hi in zip(lower,upper)):return dict(reserved=False,reason='numeric_world_domain')
 return free_grid_reserve(lower,upper,origin=snapshot.origin,shape=snapshot.shape,resolution=snapshot.resolution,free_cells=snapshot.free,body_radius=F(171,1000),interstep_reserve=F(1,200))
