import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/"reports/phase_4_gate_artifacts/shield_research/taylor_remainder_v1"))
"""Joint flow Taylor remainder and global parameter-family local Gamma bounds."""
from pathlib import Path
from fractions import Fraction as F
import json,time,hashlib,numpy as np
import interval_taylor as t
from sector_flow import flow_map,GCENTER,GDELTA
OUT=Path(__file__).resolve().parent
SOURCE=Path.cwd()/'reports/phase_4_gate_artifacts/shield_research/taylor_remainder_v1'
source=json.loads((SOURCE/'inflated_structured_noise_Q_taylor_bounds.json').read_text())
r=source['state_component_radii_binary64_outward'];ep=.005;ev=.01
controller,_,_=t.backup_map(r,ep,ev,0,structured_controller=True)
# Capture actual validated controller deviation bounds, not sampled commands or
# a point gain. Rational decimal ceilings make the declared SI box explicit.
command_exact=list(map(F,['.115899','.223856','.223856','.024255336']))
u=[t.outer(x,True) for x in command_exact];joint=r+u+[0.,0.];d=list(map(t.frac,joint));rows=[];signed={}
print('command_deviation_SI_box',u,flush=True)
for branch in (0,1):
 start=time.monotonic()
 try:
  output,checks,p=flow_map(r,u,branch)
  H=np.array([x.h.absmax() for x in output]);eps=[]
  for i in range(21):eps.append(t.outer(sum((t.frac(H[i,j,k])*d[j]*d[k]/2 for j in range(25) for k in range(25)),F(0)),True))
  zero,zero_checks,_=flow_map([0.]*21,[0.]*4,branch)
  gamma_lo=np.array([x.g.lo[21:25] for x in zero]);gamma_hi=np.array([x.g.hi[21:25] for x in zero])
  # Exact causal/queue structure removes only analytically zero entries.
  allowed={0:{2,5,12,13},1:{1,4,6,9,14},2:{0,3,7,10,15},3:{8,11,16}}
  for i in range(21):
   for j in range(4):
    if i not in allowed[j] or branch==1:gamma_lo[i,j]=gamma_hi[i,j]=0.
  gamma_lo[13:17]=np.eye(4);gamma_hi[13:17]=np.eye(4)
  for suffix,arr in [('H_lower',[x.h.lo for x in output]),('H_upper',[x.h.hi for x in output]),('J_lower',[x.g.lo for x in output]),('J_upper',[x.g.hi for x in output])]:signed[f'branch{branch}_{suffix}']=np.array(arr)
  row={'branch':branch,'status':'conditional_real_joint_flow_hessian_domain_pass','epsilon_joint_state_command':eps,
       'local_Gamma_parameter_interval_lower':gamma_lo.tolist(),'local_Gamma_parameter_interval_upper':gamma_hi.tolist(),
       'zero_origin_checks':zero_checks,'branch_domain_checks':checks,'elapsed_s':time.monotonic()-start}
  print(branch,'flow_epsmax',max(eps),'motor_queue_eps',eps[12:],'sec',row['elapsed_s'],flush=True)
 except t.DomainFailure as e:row={'branch':branch,'status':'domain_failure','reason':str(e)}
 rows.append(row)
np.savez_compressed(OUT/'sector_quantized_flow_derivative_intervals.npz',**signed)
result={'scope':'REAL physical flow with independent public SI new-command input; controller NL/noise propagated through localGamma ONCE, plus this joint flow remainder. Not a machine execution or RPI certificate.',
        'exact_real_quaternion_increment_factorization':True,'flight_certificate':False,'machine_execution_budget_proved':False,
        'source_controller_bounds':{'path':str(SOURCE/'inflated_structured_noise_Q_taylor_bounds.json'),'sha256':hashlib.sha256((SOURCE/'inflated_structured_noise_Q_taylor_bounds.json').read_bytes()).hexdigest()},
        'command_quantization_allowance_per_axis':'1/1000000','command_domain_provenance':'Exact inflated-Q linear-form supports plus proved joint27D structured-H remainder and linear noise bounds; assess_phase4 external companion audit.',
        'state_radii':r,'command_deviation_radii_exact':[str(x) for x in command_exact],'command_deviation_radii_binary64_outward':u,
        'input_order':'state0..20, new SI command deviations21..24,unused25..26','runs':rows,
        'source_sha256':{'sector_flow.py':hashlib.sha256((OUT/'sector_flow.py').read_bytes()).hexdigest(),'interval_taylor.py':hashlib.sha256((SOURCE/'interval_taylor.py').read_bytes()).hexdigest()},'sector_gain_center_rational':str(GCENTER),'sector_gain_radius_rational':str(GDELTA)}
(OUT/'sector_quantized_flow_bounds.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
