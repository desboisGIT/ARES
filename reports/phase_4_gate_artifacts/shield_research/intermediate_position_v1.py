"""Conditional reserve for every ORIGINAL RK4 stage/endpoint/collision chord.

Discrete numerical objects only. No physical continuous interpolation assertion.
No package edits, fitted acceleration constants, or flight certificate.
"""
from pathlib import Path
from fractions import Fraction as F
import sys,json,hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'roundoff_enclosure_v1'))
from roundoff import *
from zero_rate_history_v1 import rational_domain,rate_stage_lemma

def q(x):return x if isinstance(x,F) else F.from_float(float(x))
def bounds(d,cast=False):return ScalarRoundInterval.input(q(d['lo']),q(d['hi']),cast=cast)
def exact_max_interval(d):return max(abs(q(d['lo'])),abs(q(d['hi'])))

def real_reserve(domain):
 H,h=F(1,100),F(1,500);W=F(1);desired=F(51,100)
 params=domain['physical_parameter_real_ranges']
 if any(q(params[n]['lo'])<=0 for n in ['m','g','tau']) or any(q(d['lo'])<=0 for d in params['j']):raise ValueError('positive physical parameters required')
 if q(params['drag']['lo'])<0 or not 0<=q(params['battery']['lo'])<=q(params['battery']['hi'])<=1:raise ValueError('drag/battery domain invalid')
 if q(domain['motor_real_range']['lo'])<=0 or any(q(group[0]['lo'])<=0 for group in [domain['current_word_real_ranges'],*domain['history_word_real_ranges']]):raise ValueError('positive motor/target domains required')
 if max(exact_max_interval(d) for d in domain['raw13_real_ranges'][:3])>16:raise ValueError('world position domain exceeds arithmetic scope')
 oldrate=rate_stage_lemma(domain)
 # Same physical/nominal parameter-set argument, now with a held command.
 assert oldrate['lambda_lo']>=27 and oldrate['lambda_hi']<=34 and oldrate['remainder_coefficient']<=1
 assert max(exact_max_interval(d) for d in domain['raw13_real_ranges'][10:13])<W
 maxcmd=max(exact_max_interval(d) for group in [domain['current_word_real_ranges'],*domain['history_word_real_ranges']] for d in group[1:])
 assert maxcmd<F(1,2)<desired
 # The ideal fractional queue is a convex combination. Tiny actual queue
 # deviations are separately handled by the complete stage arithmetic trace.
 force=34*desired+W**2;amin,amax=27*h,34*h
 stage=[W,(1-amin/2)*W+h*force/2,(1-amin/2+amin**2/4)*W+h*(amax/4+F(1,2))*force,(1-amin+amin**2/2-amin**3/4)*W+h*(amax**2/4+amax/2+1)*force]
 R=1-amin+amin**2/2-amin**3/6+amin**4/24
 end=R**5*W+h*force*sum((R**k for k in range(5)),F(0))
 gyro_l1=F(6)*F('0.003000001')*W**2
 assert max(stage)<=W and end<W and gyro_l1<100
 omega=3*W;N=1/(1-h*omega)
 initial_motor=exact_max_interval(domain['motor_real_range'])
 target=max(exact_max_interval(group[0]) for group in [domain['current_word_real_ranges'],*domain['history_word_real_ranges']])
 assert initial_motor<F(13,2) and target<F(13,2)
 # Positive REAL lag recurrence and nonnegative battery<=1 imply all stage
 # thrusts remain below6.5; thrust is constant during each rigid-body substep.
 force_upper=F(13,2)
 params=domain['physical_parameter_real_ranges'];mass=q(params['m']['lo']);g=q(params['g']['hi']);drag=q(params['drag']['hi'])
 V0=max(exact_max_interval(d) for d in domain['raw13_real_ranges'][7:10])
 accel_const=force_upper/mass*N**2+g;drag_coefficient=drag/mass*N**4
 assert H*drag_coefficient<1
 V=(V0+H*accel_const)/(1-H*drag_coefficient)
 # A causal stage induction uses V0+t*A, 0<=t<=H, with
 # A=accel_const+drag_coefficient*V. Every RK4 coefficient is nonnegative.
 assert V0+H*(accel_const+drag_coefficient*V)==V
 arithmetic_padding=F(1,10000);reserve=H*V+arithmetic_padding
 assert reserve<F(1,200)
 return dict(H=H,h=h,W=W,command_component_upper=desired,rate_stage_component_bounds=stage,rate_macro_endpoint_upper=end,omega_l2_upper=omega,quaternion_stage_norm_upper=N,initial_velocity_component_upper=V0,motor_force_uniform_upper=force_upper,velocity_stage_component_upper=V,acceleration_stage_component_upper=accel_const+drag_coefficient*V,position_arithmetic_padding=arithmetic_padding,position_reserve_per_axis=reserve,chosen_reserve_per_axis=F(1,200),chosen_Euclidean_reserve=F(9,1000),proof_uses_discrete_RK4_only=True)

def stage_arithmetic(domain,branch):
 COUNTS.clear();checks={};p={name:bounds(domain['physical_parameter_real_ranges'][name],True) for name in ['m','g','drag','tau','battery']}
 p['j']=[bounds(d,True) for d in domain['physical_parameter_real_ranges']['j']]
 p['jn']=[sr(F('.002')),sr(F('.002')),sr(F('.003'))];p['j_inv']=[sr(1)/x for x in p['j']]
 h=sr(F(1,500));dt=sr(F(1,100));delay=ScalarRoundInterval.input(F('.005'),F('.02'),cast=True)/dt;decay=(-h/p['tau']).exp(0)
 radius=list(map(q,domain['projective_qxyz_radii']));xyz=[ScalarRoundInterval(I.rational(-r,r),upper(8*U*r)) for r in radius]
 sums=I.rational(0)
 for x in xyz:sums=sums+x.real.square()
 qw=I(sqrt_rational(1-frac(float(sums.hi)),False),1.)
 state=[bounds(d) for d in domain['raw13_real_ranges'][:3]]+xyz+[ScalarRoundInterval(qw,upper(8*U))]+[bounds(d) for d in domain['raw13_real_ranges'][7:]]
 motor=bounds(domain['motor_real_range']);history=[[bounds(d) for d in group] for group in [domain['current_word_real_ranges'],*domain['history_word_real_ranges']]]
 held=interpolate_history(history,branch,delay.error);held[0]=held[0]*p['battery']
 assert 0<float(held[0].actual().lo)<float(held[0].actual().hi)<22
 assert max(maximum(x.actual()) for x in held[1:])<.51
 records=[]
 def record(s,step,stage):
  records.append(dict(substep=step,stage=stage,position_error=[x.error for x in s[:3]],velocity_error=[x.error for x in s[7:10]],position_real_ranges=[x.real.record() for x in s[:3]]))
 for i in range(5):
  motor=motor_update(motor,held[0],decay,True);upstream=[motor/(p['m']*p['g'])]+held[1:]
  def dyn(s):return original_dynamics(s,upstream,p,checks)
  record(state,i+1,'k1_input');k1=scale(dyn(state),h)
  x2=add(state,scale(k1,sr(F(1,2))));record(x2,i+1,'k2_input');k2=scale(dyn(x2),h)
  x3=add(state,scale(k2,sr(F(1,2))));record(x3,i+1,'k3_input');k3=scale(dyn(x3),h)
  x4=add(state,k3);record(x4,i+1,'k4_input');k4=scale(dyn(x4),h)
  addition=add(add(add(k1,scale(k2,sr(2))),scale(k3,sr(2))),k4)
  new=add(state,[x/sr(6) for x in addition]);record(new,i+1,'endpoint_pre_normalization')
  qq,ncheck=normalize(new[3:7],True,True);checks.setdefault('normalization_checks',[]).append(ncheck);state=new[:3]+qq+new[7:]
 maxerror=max(q(e) for r in records for e in r['position_error']);padding=F(1,10000)
 assert maxerror<padding
 return dict(branch=branch,all_25_stage_endpoint_nodes_recorded=len(records)==25,maximum_position_arithmetic_error=maxerror,strictly_within_padding=maxerror<padding,padding=padding,records=records,operation_counts=dict(COUNTS),checks=checks)

def encode(x):
 if isinstance(x,F):return {'rational':str(x),'display_only':float(x)}
 if isinstance(x,list):return list(map(encode,x))
 if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
 return x

def run():
 domain=rational_domain();real=real_reserve(domain);arithmetic=[stage_arithmetic(domain,b) for b in [0,1]]
 report=dict(schema_version='ares.conditional_intermediate_position_reserve.v1',flight_certificate=False,real_geometry=real,arithmetic=arithmetic,all_discrete_nodes_and_collision_chords_covered=True,physical_continuous_curve_covered=False,arithmetic_primitive_contract_required=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),domain_source_sha256=hashlib.sha256((HERE/'roundoff_enclosure_v1/conditional_budgets.json').read_bytes()).hexdigest(),roundoff_engine_sha256=hashlib.sha256((HERE/'roundoff_enclosure_v1/roundoff.py').read_bytes()).hexdigest(),scope='Conditional original eagerCPU, full published terminal guard domain and physical ranges, all four RK4 stage inputs and endpoint per substep, linear chords joining nodes in the SAME macro-step cover; no physical continuous interpolant claim',chord_proof='Each covered node is in p_macro + [-reserve,reserve]^3; the box is convex so every linear chord between covered nodes is contained. In particular each consecutive macro-endpoint collision chord has both ends in its starting macro cover; do not infer coverage of arbitrary chords spanning several macro covers.',Euclidean_reserve_proof='3*(.005)^2 < (.009)^2, verified as exact rational inequality')
 assert 3*F(1,200)**2<F(9,1000)**2
 (HERE/'intermediate_position_v1.json').write_text(json.dumps(encode(report),indent=2)+'\n');print(json.dumps(encode(real),indent=2));print([(r['branch'],float(r['maximum_position_arithmetic_error'])) for r in arithmetic]);return report
if __name__=='__main__':run()
