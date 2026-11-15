"""Conditional discrete zero-rate stage/domain lemma from public packets only.

No physical sampled mass, motor, energy, gravity or inertia state is accepted.
Exact Fraction decisions; external deterministic packet and command contracts
remain assumptions. Never proves bootstrap safety or a flight certificate.
"""
from fractions import Fraction as F
from pathlib import Path
import json,sys,math,struct,hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'roundoff_enclosure_v1'))
from roundoff import ScalarRoundInterval as SR,I,sr,motor_update,gamma,U,FTZ,upper,frac

def q(x):
 if isinstance(x,F):return x
 if isinstance(x,int):return F(x)
 if not math.isfinite(float(x)):raise ValueError('nonfinite public input')
 return F.from_float(float(x))
def word(x):return q(struct.unpack('f',struct.pack('f',x))[0])
def decay_upper(x):
 term=total=F(1)
 for k in range(1,65):term*=x/k;total+=term
 return 1/total

def rational_domain():
 data=json.loads((HERE/'roundoff_enclosure_v1/conditional_budgets.json').read_text())
 row=next(r for r in data['rows'] if r.get('verified_decay') and r.get('world_position_abs_bound')==16 and r.get('current_command_is_stored_binary32_word') and r.get('error_shell_refinement') and r.get('range_unit_refinement'))
 return row['checks']['entry_domains']

def rate_stage_lemma(domain,W=F(1,100),h=F(1,500)):
 J=[(q(v['lo']),q(v['hi'])) for v in domain['physical_parameter_real_ranges']['j']]
 Jn=[(min(F(str(x)),word(x)),max(F(str(x)),word(x))) for x in [.002,.002,.003]]
 lows=[30*Jn[i][0]/J[i][1] for i in range(3)];highs=[30*Jn[i][1]/J[i][0] for i in range(3)]
 assert min(lows)>=27 and max(highs)<=34
 rem=[]
 for i,j,k in [(0,1,2),(1,2,0),(2,0,1)]:
  vals=[abs((Jn[k][c]-J[k][a])-(Jn[j][d]-J[j][b]))/J[i][0] for a in [0,1] for b in [0,1] for c in [0,1] for d in [0,1]];rem.append(max(vals))
 assert max(rem)<=1
 amin,amax=27*h,34*h;r=W*W
 # On a in[0,.1], all listed base coefficients are positive and decreasing;
 # R'(a)<=-1+a+a^3/6<0. Bounds use exact triangle inequalities for forcing.
 assert 0<amin<=amax<=F(1,10)
 assert -1+amax+amax**3/6<0  # R decreases on the whole interval.
 assert 1-amax-amax**3/4>0  # All stage base coefficients positive.
 forcing_lowers=[(1-amax-amax**3/4)/6,(2-amax)/6,(2-amax)/6,F(1,6)]
 assert min(forcing_lowers)>0 and -F(1,2)+amax/6<=0
 stage=[W,(1-amin/2)*W+h*r/2,(1-amin/2+amin**2/4)*W+h*(amax/4+F(1,2))*r,(1-amin+amin**2/2-amin**3/4)*W+h*(amax**2/4+amax/2+1)*r]
 R=1-amin+amin**2/2-amin**3/6+amin**4/24
 # Individual forcing coefficients in the endpoint are nonnegative; their
 # sum is 1-a/2+a²/6-a³/24 <=1. Five identical bounds compose geometrically.
 endpoint=R**5*W+h*r*sum((R**i for i in range(5)),F(0))+F(1,1000000)
 gyro_l1=6*max(x[1] for x in Jn)*W*W
 assert max(stage)<=W and endpoint<W and gyro_l1<100
 return dict(W=W,stage_bounds=stage,macro_angular_endpoint_bound=endpoint,lambda_lo=min(lows),lambda_hi=max(highs),remainder_coefficient=max(rem),gyro_compensation_norm_upper=gyro_l1,omega_l2_upper=3*W,endpoint_forcing_coefficient_over_h_lowers=forcing_lowers)

def broad_motor_arithmetic():
 # Independent settling-domain calculation avoids assuming the final RPI
 # motor interval before entry. Stored delay and battery words are inputs.
 # gamma6 for two-term queue interpolation, followed by battery multiply.
 queue_error=gamma(6)*F(13,2)+12*FTZ
 target_error=queue_error+U*(F(13,2)+queue_error)+FTZ
 assert target_error<F(4,1000000)
 lag=SR.input(F('0.0149999987'),F('0.040000003'),cast=True)
 decay=(sr(-F(1,500))/lag).exp(0)
 motor=SR.input(0,F(6501,1000));target=SR(I.rational(0,F(13,2)),upper(target_error))
 for _ in range(5):motor=motor_update(motor,target,decay)
 epsilon=F(1,50000)
 assert q(motor.error)<epsilon
 a=decay_upper(F(1,100)/F('0.040000003'))
 assert F(13,2)+epsilon/(1-a)<F(6501,1000)
 assert (1-decay_upper(F(1,500)/F('0.040000003')))*F(19,5)>epsilon/(1-a)
 return dict(standalone_settle_motor_error_per_H=epsilon,calculated_motor_error=q(motor.error),constant_target_arithmetic_upper=target_error,bootstrap_motor_interval_upper=F(6501,1000),macro_decay_upper=a)

def history_case(history,hold,case,domain,rate,motor):
 ep,ev,ew=F(1,200),F(1,100),F(1,1000);shell=8*U
 times=[F(row['time']) for row in history]
 if len(times)<2 or any(b-a!=F(1,100) for a,b in zip(times,times[1:])):raise ValueError('complete exact macro-grid history required')
 if not 4<=q(hold)<=F(13,2):raise ValueError('settling thrust must be within[4,6.5]')
 checks=[];maxq=[F(0)]*3;maxp=maxv=maxw=F(0)
 for row in history:
  p,v,quat,w=[list(map(q,row[name][case])) for name in ['p','v','q','w']]
  n2=sum((x*x for x in quat),F(0));checks.append(quat[3]>0 and (1-shell)**2<=n2<=(1+shell)**2)
  for j in range(3):
   maxq[j]=max(maxq[j],abs(quat[j])/(1-shell));maxp=max(maxp,abs(p[j])+ep);maxv=max(maxv,abs(v[j])+ev);maxw=max(maxw,abs(w[j])+ew)
   checks.append(abs(p[j])+ep<=16)
   d=domain['raw13_real_ranges'][7+j];checks.append(q(d['lo'])<=v[j]-ev and v[j]+ev<=q(d['hi']))
   checks.append(abs(w[j])+ew<=rate['W'])
   radius=q(domain['projective_qxyz_radii'][j]);checks.append(quat[j]**2<=radius**2*n2)
 # Whole substep quaternion stage bound, REAL unit initial condition. No ODE
 # solution: geometric polynomial bound applies directly to RK4 stages.
 h=F(1,500);omega=rate['omega_l2_upper'];N=1/(1-h*omega);D=h*omega*N/2
 delta=2*D/(1-D)
 xyz=[x+5*delta+D for x in maxq]
 axis=max(N*N-1,1-(1-D)**2)+2*xyz[0]**2+2*xyz[1]**2
 checks.append(axis<=F(1,5000));checks.append(omega<=1)
 hold=q(hold);H=F(1,100);lag=F('0.040000003');delay=F('0.02000001');settle=F(7,10)
 periods=(settle-delay-H)//H;duration=periods*H
 residual=(F(13,2)+hold)*decay_upper(duration/lag)+motor['standalone_settle_motor_error_per_H']/(1-motor['macro_decay_upper'])
 motorlo=F(19,20)*hold-residual;motorhi=hold+residual
 mr=domain['motor_real_range'];checks.append(q(mr['lo'])<=motorlo and motorhi<=q(mr['hi']))
 for group in [domain['current_word_real_ranges'],*domain['history_word_real_ranges']]:
  checks.append(q(group[0]['lo'])<=hold<=q(group[0]['hi']))
  checks.extend(q(d['lo'])<=0<=q(d['hi']) for d in group[1:])
 return dict(case=case,conditional_history_inputs_and_stages_pass=all(checks),all_checks=len(checks),failed_checks=sum(not x for x in checks),stage_up_axis_error_upper=axis,stage_angular_rate_upper=omega,maximum_packet_outer_position=maxp,maximum_packet_outer_velocity=maxv,maximum_packet_outer_angular_velocity=maxw,motor_lower_after_settle=motorlo,motor_upper_after_settle=motorhi,constant_thrust=q(hold),settling_complete_macro_periods=periods,raw_q_projective_component_upper=maxq,hard_public_packet_error_assumptions=dict(ep=ep,ev=ev,ew=ew))

def encode(x):
 if isinstance(x,F):return {'rational':str(x),'display_only':float(x)}
 if isinstance(x,list):return list(map(encode,x))
 if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
 return x

def run():
 capture_path=HERE/'public_trim_capture_v1.json';data=json.loads(capture_path.read_text());domain=rational_domain();rate=rate_stage_lemma(domain);motor=broad_motor_arithmetic()
 rows=[history_case(data['measured_history'],r['public_hold_thrust_N'],r['case'],domain,rate,motor) for r in data['cases']]
 report=dict(schema_version='ares.conditional_zero_rate_public_history.v1',all_cases_retained=True,cases=rows,all_eight_conditional_cases_pass=all(r['conditional_history_inputs_and_stages_pass'] for r in rows),rate_lemma=rate,settling_motor_arithmetic=motor,flight_certificate=False,bootstrap_safety_proved=False,external_hard_packet_service_truth_not_proved=True,numeric_primitive_contract_remains_required=True,assumptions=['Every p/v/w packet error is deterministically bounded by component ep=.005,ev=.01,ew=.001 including measurement arithmetic','q packet is exact current Float32 raw quaternion, at the same macro timestamp','All issued commands and both prior queues have the same stored Fhold and exactly zero rates throughout stated settle+history after transport','Initial motor is within[0,6.5] at the constant-command window start, battery represented authority remains[.95,1]','Physical and nominal constants within the independently supplied profile-set/CPU arithmetic contracts','History is complete on the exact .01s macro grid; no packet is filtered','Published full-plant numeric budget applies only after independently establishing its entry domain'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),capture_sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),no_sampled_physical_or_motor_truth_inputs=True)
 (HERE/'zero_rate_history_v1.json').write_text(json.dumps(encode(report),indent=2)+'\n');print(json.dumps(encode({k:v for k,v in report.items() if k!='cases'}),indent=2));print([(r['case'],r['conditional_history_inputs_and_stages_pass'],float(r['stage_up_axis_error_upper'])) for r in rows])
 return report
if __name__=='__main__':run()
