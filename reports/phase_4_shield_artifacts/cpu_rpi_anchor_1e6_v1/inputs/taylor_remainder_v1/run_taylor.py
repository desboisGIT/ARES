import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import interval_taylor as t
from fractions import Fraction as F
import numpy as np,json,time,hashlib,datetime
OUT=Path(__file__).resolve().parent
GROUPS={'planar_x':([0,3,7,10,15,19],['.111','.153','.0223','.58','.427','.595']),
 'planar_y':([1,4,6,9,14,18],['.111','.153','.0223','.58','.427','.595']),
 'vertical':([2,5,12,13,17],['.339079','.396802','1.782115','1.438910','1.933647']),
 'yaw':([8,11,16,20],['.011805','.414826','.363444','.510264'])}
RADII={'planar_x':F(1,10),'planar_y':F(1,10),'vertical':F(1,10),'yaw':F(1,20)}
component=[F(0)]*21
for name,(indices,coefficients) in GROUPS.items():
 for i,c in zip(indices,coefficients):component[i]=RADII[name]*F(c)
radius=[t.outer(x,True) for x in component]

def exact_eps(H,gradient,ep,ev):
 # Coefficients/domain radii enter exact rational sums, with outward binary64
 # reporting afterwards; no optimistic floating accumulation of epsilon.
 d=list(map(t.frac,radius));noise=[t.frac(ep)]*3+[t.frac(ev)]*3
 e=[];n=[]
 for i in range(21):
  q=sum((t.frac(H[i,j,k])*d[j]*d[k]/2 for j in range(21) for k in range(21)),F(0))
  e.append(t.outer(q,True))
  q=sum((t.frac(gradient[i,21+j])*noise[j] for j in range(6)),F(0))
  n.append(t.outer(q,True))
 return e,n

result={'schema_version':'ares.backup_interval_taylor.v1','created_utc':datetime.datetime.now(datetime.UTC).isoformat(),
 'scope':'REAL centered nonlinear21Dbackup map plus6bounded publicp/vnoiseinputs; NOT floatingplant execution or RPIproof.',
 'flight_certificate':False,'machine_execution_budget_proved':False,
 'exact_real_substep_s':'1/500','steps_per_control':5,'positive_qw_chart':True,
 'component_bounds_basis':'Externally exact-audited inverse-P diagonal projection bounds supplied by assess_phase4; conservative rational upper coefficients.',
 'groups':GROUPS,'product_P_radii':{k:str(v) for k,v in RADII.items()},'state_component_radii_exact':[str(v) for v in component],
 'state_component_radii_binary64_outward':radius,'runs':[]}
signed={}
started=time.monotonic()
for ep,ev,label in [(0.,0.,'zero_noise_diagnostic'),(.0005,.001,'precision_diagnostic'),(.005,.01,'declared_service_bounds')]:
 for branch in (0,1):
  begin=time.monotonic()
  try:
   output,checks,param=t.backup_map(radius,ep,ev,branch)
   H=np.array([x.h.absmax() for x in output]);G=np.array([x.g.absmax() for x in output])
   prefix=f'{label}_branch{branch}'
   signed[prefix+'_H_lower']=np.array([x.h.lo for x in output]);signed[prefix+'_H_upper']=np.array([x.h.hi for x in output])
   signed[prefix+'_J_lower']=np.array([x.g.lo for x in output]);signed[prefix+'_J_upper']=np.array([x.g.hi for x in output])
   eps,noise=exact_eps(H,G,ep,ev)
   row={'label':label,'branch':branch,'delay_s':[.005,.01] if branch==0 else [.01,.02],
        'position_noise_component_bound_m':ep,'velocity_noise_component_bound_mps':ev,
        'status':'conditional_real_interval_hessian_domain_pass','epsilon_state_nonlinear':eps,
        'epsilon_measurement_noise':noise,'jacobian_absolute_upper':G.tolist(),
        'hessian_absolute_upper':H.tolist(),'branch_domain_checks':checks,
        'parameter_intervals':{k:([v.record() for v in value] if isinstance(value,list) else value.record()) for k,value in param.items()},
        'elapsed_s':time.monotonic()-begin}
   print(label,branch,'stateeps_max',max(eps),'noiseeps_max',max(noise),'sec',row['elapsed_s'],flush=True)
  except t.DomainFailure as error:
   row={'label':label,'branch':branch,'status':'branch_domain_failure','reason':str(error),'elapsed_s':time.monotonic()-begin}
   print(label,branch,row,flush=True)
  result['runs'].append(row)
np.savez_compressed(OUT/'signed_derivative_intervals.npz',**signed)
result['signed_derivative_intervals_sha256']=hashlib.sha256((OUT/'signed_derivative_intervals.npz').read_bytes()).hexdigest()
result['elapsed_s']=time.monotonic()-started
result['source_sha256']={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['interval_taylor.py','run_taylor.py']}
(OUT/'taylor_bounds.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
