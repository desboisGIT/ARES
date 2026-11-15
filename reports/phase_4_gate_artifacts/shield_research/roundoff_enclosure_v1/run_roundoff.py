from pathlib import Path
from fractions import Fraction as F
import json,math,time,hashlib,numpy as np,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from roundoff import *
OUT=Path(__file__).resolve().parent
TAYLOR=OUT.parent/'taylor_remainder_v1'
source=json.loads((TAYLOR/'increment_inflated_flow_bounds.json').read_text())
r=source['state_radii'];ur=list(map(F,['.116','.224','.224','.0243']))

def run(branch,range_refinement=True,error_shell_refinement=True,verified_decay=False,current_words=False,position_abs_bound=None):
 COUNTS.clear();start=time.monotonic();checks={}
 try:
  def param(lo,hi,widen=True):
   if widen:
    lo=frac(float(np.nextafter(np.float32(lo),np.float32(-np.inf))))
    hi=frac(float(np.nextafter(np.float32(hi),np.float32(np.inf))))
   else:lo,hi=F(str(lo)),F(str(hi))
   return ScalarRoundInterval.input(lo,hi,cast=True)
  p={'m':param(.45,.55),'g':param(9.78,9.83),'drag':param(.05,.2),'tau':param(.015,.04),
     'j':[param(.0018,.0022),param(.0018,.0022),param(.0027,.0033)],
     'jn':[sr(F('.002')),sr(F('.002')),sr(F('.003'))],'battery':param(.95,1.,False)}
  p['j_inv']=[sr(1)/x for x in p['j']]
  h=sr(F(1,500));dt=sr(F(1,100));delay=param(.005,.02,False)/dt
  # Monotone RNE parameter casts and positive division imply these exact
  # endpoints. Thus all actual floor/ceil choices use the three supplied slots.
  assert F.from_float(f32(.02))==2*F.from_float(f32(.01))
  assert 2*F.from_float(f32(.005))==F.from_float(f32(.01))
  checks['delay_actual_frames_proved_by_RNE_monotonic_endpoints']=[.5,2.]
  checks['delay_quotient_absolute_error']=delay.error
  decay=(-h/p['tau']).exp(0 if verified_decay else 1)
  if not(0<float(decay.actual().lo)<float(decay.actual().hi)<1):raise EnclosureFailure('actual motor decay not in(0,1)')
  checks['decay']=decay.record()
  m=p['m'].real;g=p['g'].real;b=p['battery'].real;mg=m*g;teq=mg/b
  eq=(I.rational(F('9.80665'))-2*teq)/I.rational(F(3,2))
  checks['reference_equilibrium_offset_interval']=eq.record()
  position=[]
  for i in range(3):
   interval=I.rational(6)+I(-r[i],r[i])+(eq if i==2 else I.rational(0))
   if position_abs_bound is not None:interval=I(-position_abs_bound,position_abs_bound)
   position.append(ScalarRoundInterval(interval))
  xyz=[ScalarRoundInterval(I(-r[6+i],r[6+i]),upper(8*U*frac(r[6+i]))) for i in range(3)]
  total=I.rational(0)
  for x in xyz:total=total+x.real.square()
  qw=I(sqrt_rational(1-frac(float(total.hi)),False),1.)
  initial_qw_error=upper(8*U*frac(maximum(qw)))
  q=xyz+[ScalarRoundInterval(qw,initial_qw_error)]
  velocity=[ScalarRoundInterval(I(-r[3+i],r[3+i])) for i in range(3)]
  omega=[ScalarRoundInterval(I(-r[9+i],r[9+i])) for i in range(3)]
  state=position+q+velocity+omega
  motor=ScalarRoundInterval(mg+I(-r[12],r[12]))
  current=[ScalarRoundInterval(teq+I(-float(ur[0]),float(ur[0])),upper(U*frac(maximum(teq+I(-float(ur[0]),float(ur[0]))))+FTZ))]
  current +=[ScalarRoundInterval.input(-x,x,cast=True) for x in ur[1:]]
  queue=[]
  for slot in (13,17):
   queue.append([ScalarRoundInterval(teq+I(-r[slot],r[slot]))]+[ScalarRoundInterval(I(-r[slot+i],r[slot+i])) for i in range(1,4)])
  if current_words:
   current=[ScalarRoundInterval(x.real,0.) for x in current]
  held=interpolate_history([current]+queue,branch,delay.error)
  held[0]=held[0]*p['battery']
  if float(held[0].actual().lo)<=0 or float(held[0].actual().hi)>=22:
   raise EnclosureFailure('actual thrust clamp branches not excluded')
  if any(maximum(x.actual())>=6 for x in held[1:]):raise EnclosureFailure('actual body-rate clamp not excluded')
  checks['held_applied_SI_command']=[x.record() for x in held]
  checks['entry_domains']={'raw13_real_ranges':[x.real.record() for x in state],
   'projective_qxyz_radii':r[6:9],'motor_real_range':motor.real.record(),
   'current_word_real_ranges':[x.real.record() for x in current],
   'history_word_real_ranges':[[x.real.record() for x in q] for q in queue],
   'physical_parameter_real_ranges':{name:([x.real.record() for x in value] if isinstance(value,list) else value.real.record()) for name,value in p.items() if name!='j_inv'}}
  checkpoints=[]
  for step in range(5):
   state,motor=original_substep(state,motor,held,decay,p,h,checks,range_refinement,error_shell_refinement)
   checkpoints.append({'substep':step+1,'raw13_error':[x.error for x in state],'motor_error_N':motor.error,
                       'raw13_real_ranges':[x.real.record() for x in state]})
  projected_q=[upper((frac(state[i].error)+8*U*frac(maximum(state[i].real)))/(1-8*U)) for i in range(3,6)]
  raw21=[x.error for x in state[:3]]+[x.error for x in state[7:10]]+projected_q+[x.error for x in state[10:13]]+[motor.error]+[x.error for x in current]+[0.]*4
  result={'projected_output_qxyz_error':projected_q,'same_center_projective_state21_error':raw21,'status':'conditional_forward_budget_available','raw13_error':[x.error for x in state],
          'motor_error_N':motor.error,'new_command_cast_error':[x.error for x in current],
          'raw13_real_ranges':[x.real.record() for x in state],'motor_real_range':motor.real.record(),
          'checkpoints':checkpoints,'checks':checks}
 except Exception as error:
  result={'status':'enclosure_failure','exception_type':type(error).__name__,'reason':str(error),'checks':checks}
 result.update(branch=branch,range_unit_refinement=range_refinement,error_shell_refinement=error_shell_refinement,
               verified_decay=verified_decay,current_command_is_stored_binary32_word=current_words,world_position_abs_bound=position_abs_bound,elapsed_s=time.monotonic()-start,operation_counts=dict(COUNTS))
 return result

result={'schema_version':'ares.conditional_float32_forward_budget.v1','flight_certificate':False,
 'platform_primitive_contract_audited':False,'gpu_claim':False,
 'scope':'Original eager CPU13Dvendor/adapter forward expression; original optimizedquatmul/fullrotation/RK4, original normalizedq. Conditional bounds only; initial raw inputs are stored values except declared quaternion norm shell.',
 'state_domain_source':{'path':str(TAYLOR/'increment_inflated_flow_bounds.json'),'sha256':hashlib.sha256((TAYLOR/'increment_inflated_flow_bounds.json').read_bytes()).hexdigest()},
 'primitive_assumptions':{'RNE_binary32_basic_full_precision':True,'unit_roundoff_exact':str(U),
  'absolute_output_FTZ_per_operation':str(FTZ),'input_DAZ_disabled_or_separately_bounded':True,
  'short_dot':'nproducts/nadds <=gamma_(2n)*sumabs +4n*FTZ, n3or4, includesFMA/reassociation',
  'sqrt_error_vs_exact_ULP':'.5','exp_extra_ULP_vs_RNE':{'native_candidate':1,'verified_decay_backend':0},
  'norm4':'sum4squares withgamma8+8FTZ, thenassumedsqrt; no installedtorch.linalg vector_norm audit',
  'overflow_nan_inf_excluded_by_domains':True,'no_reduced_precision_matmul_or_autocast':True},
 'initial_quaternion_norm_shell':str(8*U),'initial_reference_quaternion':'exact projectivepi(raw), componenterrors<=8u*maxabs(referenceqi)',
 'command_box_includes_verified_pointlaw_1e_minus6':['.116','.224','.224','.0243'],
 'parameter_real_ranges':'binary32-outward-widened physicalprofile endpoints; castsconservativelycountedagain; latencyandbatterydeclaredbounds',
 'world_position_domains':'anchor6 equilibriumfamily variants; optional uniform |p_i|<=16 primary, perrow',
 'world_anchor':[6.,6.,6.],'rows':[]}
for options in [(False,False,False,False),(True,False,False,False),(True,True,False,False),(True,True,True,False),(True,True,True,True),(True,True,True,True,16.)]:
 for branch in (0,1):
  row=run(branch,*options);result['rows'].append(row)
  print(options,branch,row['status'],row.get('reason',row.get('raw13_error')),flush=True)
result['source_sha256']={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['roundoff.py','run_roundoff.py']}
(OUT/'conditional_budgets.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
