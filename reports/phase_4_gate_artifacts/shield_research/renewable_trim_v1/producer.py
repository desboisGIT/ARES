"""Conditional renewable trim from public packets + acknowledged command words.

No sampled mass/mg/motor/E is accepted. This extension does not mutate/replace a
live guard; it proposes an admission only after current-family/geometry checks.
"""
from pathlib import Path
from fractions import Fraction as F
import sys,math,struct
BASE=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'terminal_reservation_v1'))
from verified_backup_point_v2 import backup_point,output32,I
from zero_rate_history_v1 import q,rational_domain,rate_stage_lemma,broad_motor_arithmetic,decay_upper
from trim_history_v2 import rk4_trim_bound
from current_terminal_membership_v1 import current_terminal_membership
from reservation import free_grid_reserve
from energy import ceil_grid

MAX_RATE=F(1,1000);H=F(1,100);SETTLE=70;HISTORY=200

def stored32(x):
 x=q(x)
 try:return F.from_float(struct.unpack('f',struct.pack('f',float(x)))[0])==x
 except (OverflowError,ValueError):return False

def packet_dict(packet):
 return {n:list(packet[n] if isinstance(packet,dict) else getattr(packet,n)) for n in ['p','v','q','w']}

def target_from_public_commands(commands):
 if not commands or any(len(w)!=4 or not stored32(w[0]) for w in commands):raise ValueError('nonempty known stored SI words required')
 mean=sum((q(w[0]) for w in commands),F(0))/len(commands);target,error=output32(I(mean))
 if not 4<=q(target)<=F(13,2):raise ValueError('public mean target outside supported trim interval')
 return dict(target_word=target,mean_exact=mean,target_mean_error=error)

def recalibration_candidate(packet,old_hold,target_word):
 s=packet_dict(packet);p,v,quat=[list(map(q,s[n])) for n in ['p','v','q']];old=q(old_hold);T=q(target_word)
 if not stored32(T) or not stored32(old) or not 4<=T<=F(13,2):raise ValueError('old hold/new target must be valid stored Float32 words')
 x,y,z,w=quat;n2=sum(t*t for t in quat)
 if n2<=0 or w<=0:raise ValueError('positive projective q chart required')
 bz=(w*w-x*x-y*y+z*z)/n2
 if bz<=0:raise ValueError('upright body-z component required')
 fz=T/bz
 exact=[p[0]+F(5,3)*v[0],p[1]+F(5,3)*v[1],p[2]+(fz-old+F(5,4)*v[2])/F(3,4)]
 anchor=[float(a) for a in exact]
 point=backup_point(s['p'],s['v'],s['q'],anchor,float(old))
 constant=q(point['command'][0])==T
 return dict(anchor=anchor,command=point['command'],exact_error=point['exact_error'],constant_target_word_verified=constant,small_rate_window_eligible=constant and max(abs(q(x)) for x in point['command'][1:])<=MAX_RATE,requires_fresh_C_and_geometry_admission=True,uses_sampled_parameters=False,flight_certificate=False)

def small_rate_stage_lemma(domain=None,max_rate=MAX_RATE):
 d=rational_domain() if domain is None else domain;base=rate_stage_lemma(d);u=q(max_rate)
 if not 0<=u<=MAX_RATE:raise ValueError('bounded issued rates must not exceed.001/component')
 W,h=F(1,100),F(1,500);amin,amax=27*h,34*h;r=W*W+34*u
 stages=[W,(1-amin/2)*W+h*r/2,(1-amin/2+amin**2/4)*W+h*(amax/4+F(1,2))*r,(1-amin+amin**2/2-amin**3/4)*W+h*(amax**2/4+amax/2+1)*r]
 R=1-amin+amin**2/2-amin**3/6+amin**4/24;end=R**5*W+h*r*sum((R**k for k in range(5)),F(0))+F(1,1000000)
 if max(stages)>W or end>=W:raise ArithmeticError('small-rate discrete invariant check failed')
 return dict(**{k:v for k,v in base.items() if k not in ['stage_bounds','macro_angular_endpoint_bound']},stage_bounds=stages,macro_angular_endpoint_bound=end,issued_rate_component_upper=u,forcing_upper=r)

def verify_stage_domain(packets):
 if len(packets)<2:raise ValueError('complete public history required')
 d=rational_domain();rate=small_rate_stage_lemma(d);ep,ev,ew=F(1,200),F(1,100),F(1,1000);shell=F(8,2**24);checks=[];xy=[F(0),F(0)]
 for s in packets:
  s=packet_dict(s);quat=list(map(q,s['q']));n2=sum(t*t for t in quat);checks.append(quat[3]>0 and (1-shell)**2<=n2<=(1+shell)**2)
  for j in range(3):
   p,v,w=map(q,[s['p'][j],s['v'][j],s['w'][j]]);vd=d['raw13_real_ranges'][7+j];radius=q(d['projective_qxyz_radii'][j])
   checks.extend([abs(p)+ep<=16,q(vd['lo'])<=v-ev<=v+ev<=q(vd['hi']),abs(w)+ew<=rate['W'],quat[j]**2<=radius**2*n2])
   if j<2:xy[j]=max(xy[j],abs(quat[j])/(1-shell))
 omega=rate['omega_l2_upper'];h=F(1,500);N=1/(1-h*omega);D=h*omega*N/2;dq=2*D/(1-D);xy=[a+5*dq+D for a in xy]
 up=max(N*N-1,1-(1-D)**2)+2*xy[0]**2+2*xy[1]**2;checks.append(up<=F(1,5000))
 return dict(stage_and_macro_domains_pass=all(checks),failed_checks=sum(not x for x in checks),stage_up_axis_error=up,stage_omega_upper=omega,rate_lemma=rate)

def renewed_reservation(proof,known_map,anchor,trim,*,tick,deadline_tick):
 if type(tick) is not int or type(deadline_tick) is not int or not tick<deadline_tick<=tick+1000:raise ValueError('renewal must reserve at most1000 future periods')
 drift=proof.mg_upper*proof.authority_drop*(deadline_tick-tick)/F(19,20)**2;loD,hiD=q(trim[0])-drift,q(trim[1]);a=list(map(q,anchor));rx=proof.radii['planar_x'][0];rz=proof.radii['vertical'][0]
 lower=[a[0]-rx,a[1]-rx,a[2]+loD/F(3,4)-rz];upper=[a[0]+rx,a[1]+rx,a[2]+hiD/F(3,4)+rz]
 if any(lo-F(1,200)<-16 or hi+F(1,200)>16 for lo,hi in zip(lower,upper)):return dict(reserved=False,reason='numeric_domain')
 now,end=float(tick*H),float(deadline_tick*H)
 if q(now)>tick*H:now=math.nextafter(now,-math.inf)
 if q(end)<deadline_tick*H:end=math.nextafter(end,math.inf)
 free=tuple(c for c in known_map.free if known_map.known_free(c,time_s=now) and known_map.known_free(c,time_s=end))
 return free_grid_reserve(lower,upper,origin=known_map.origin,shape=known_map.shape,resolution=known_map.resolution,free_cells=free,body_radius=F(171,1000),interstep_reserve=F(1,200))

class PublicTrimWindow:
 """Collect trusted guard receipts, then recompute a proposed renewable family.

    A receipt must come from the existing verified guard, not an untrusted bool
    supplier. Every rejected/fallback/word-mismatch receipt clears the window.
    No epoch switch is performed by this class.
 """
 def __init__(self,target_word,proof_sha256):
  self.target=q(target_word);self.proof_sha256=proof_sha256;self.records=[];self.final=None
  if not stored32(self.target) or not 4<=self.target<=F(13,2):raise ValueError('supported constant target word required')
 def clear(self):self.records=[];self.final=None
 def note_issued(self,*,tick,packet,decision,acknowledged_command,decision_epoch_global_tick=0):
  word=tuple(map(q,acknowledged_command));s=packet_dict(packet)
  okay=(type(tick) is int and len(word)==4 and all(stored32(x) for x in word) and word[0]==self.target and max(map(abs,word[1:]))<=MAX_RATE and decision.get('conditional_motion_certificate') is True and decision.get('proposal_accepted') is True and decision.get('command_point_error_verified') is True and decision.get('proof_sha256')==self.proof_sha256 and type(decision_epoch_global_tick) is int and decision_epoch_global_tick>=0 and decision.get('tick')==tick-decision_epoch_global_tick and tuple(map(q,decision.get('command',())))==word)
  if self.records and tick!=self.records[-1]['tick']+1:okay=False
  if not okay:self.clear();return False
  self.records.append(dict(tick=tick,packet=s,word=word,decision_epoch_global_tick=decision_epoch_global_tick,receipt_anchor=decision.get('anchor')));self.final=None;return True
 def note_final_packet(self,*,tick,packet):
  if not self.records or tick!=self.records[-1]['tick']+1:raise ValueError('post-ack current packet at next macro tick required')
  self.final=dict(tick=tick,packet=packet_dict(packet))
 def propose(self,*,proof,known_map,public_energy_upper,service_lease_expires_tick):
  if self.final is None or len(self.records)<SETTLE+HISTORY:raise ValueError('at least.7s settle+2s complete public history required')
  if proof.source_sha256!=self.proof_sha256:raise ValueError('proof receipt source mismatch')
  E=q(public_energy_upper);tick=self.final['tick'];deadline=tick+1000
  if not 0<=E<=8000 or service_lease_expires_tick<deadline:return dict(admitted=False,reason='energy_or_service_lease',flight_certificate=False)
  # Keep the latest contiguous270 issued intervals plus final packet. The first
  #70 establish constant-T settling; the remaining200 supply vertical balance.
  rec=self.records[-270:];packets=[r['packet'] for r in rec]+[self.final['packet']];history=packets[70:];domain=verify_stage_domain(history)
  if not domain['stage_and_macro_domains_pass']:return dict(admitted=False,reason='stage_domain',domain_check=domain,flight_certificate=False)
  params=proof.certificate['physical_parameter_intervals_outward'];mass_lo,mass_hi=map(F,params['mass']);lag=F(params['motor_lag'][1]);drag=F(params['drag'][1]);broad=broad_motor_arithmetic()
  numeric_domain=rational_domain();complete_settle=F(66,100);residual=(F(13,2)+self.target)*decay_upper(complete_settle/lag)+broad['standalone_settle_motor_error_per_H']/(1-broad['macro_decay_upper'])
  absolute_motor=(F(19,20)*self.target-residual,self.target+residual);mr=numeric_domain['motor_real_range']
  motor_pass=q(mr['lo'])<=absolute_motor[0]<=absolute_motor[1]<=q(mr['hi'])
  words_pass=all(q(group[0]['lo'])<=self.target<=q(group[0]['hi']) and all(q(d['lo'])<=-MAX_RATE<=MAX_RATE<=q(d['hi']) for d in group[1:]) for group in [numeric_domain['current_word_real_ranges'],*numeric_domain['history_word_real_ranges']])
  domain.update(post_settle_absolute_motor_interval=absolute_motor,motor_and_word_numeric_domains_pass=motor_pass and words_pass)
  if not motor_pass or not words_pass:return dict(admitted=False,reason='motor_or_word_numeric_domain',domain_check=domain,flight_certificate=False)
  trim=rk4_trim_bound(times=[i*H for i in range(201)],measured_vertical_velocity=[s['v'][2] for s in history],velocity_error=F(1,100),control_period=H,physics_period=F(1,500),hold_thrust=self.target,public_force_upper=F(13,2),mass_lower=mass_lo,drag_upper=drag,authority_drop_per_period_upper=proof.authority_drop,motor_lag_upper=lag,transport_upper=F(2000001,100000000),constant_command_settle=F(7,10),stage_up_axis_error=F(1,5000),stage_angular_rate_upper=1,velocity_roundoff_per_period=F(53,100000000),motor_roundoff_per_period=broad['standalone_settle_motor_error_per_H'])
  force=ceil_grid(mass_hi/F(19,20)*trim['conditional_trim_acceleration_bound']);motor=ceil_grid(trim['motor_tracking_force_bound']);last=history[-1];anchor=last['p'];queue=[rec[-1]['word'],rec[-2]['word']]
  membership=current_terminal_membership(**last,anchor=anchor,hold_thrust=self.target,issued_queue=queue,trim_force_interval=(-force,force),motor_residual_interval=(-motor,motor),shapes=proof.shapes)
  reserve=renewed_reservation(proof,known_map,anchor,(-force,force),tick=tick,deadline_tick=deadline) if membership['current_set_included'] else dict(reserved=False,reason='membership')
  admitted=membership['current_set_included'] and reserve['reserved']
  return dict(admitted=bool(admitted),reason='renewal_admitted' if admitted else 'current_family_or_reserve',epoch_tick=tick,deadline_tick=deadline,hold_thrust=self.target,anchor=anchor,public_final_packet=last,trim_force_interval=(-force,force),motor_residual_interval=(-motor,motor),issued_queue=queue,public_initial_energy_upper_J=E,membership=membership,reservation=reserve,domain_check=domain,trim_lemma=trim,requires_atomic_epoch_switch_and_retained_old_backup=True,mathematical_proof_sha256=proof.source_sha256,flight_certificate=False)
