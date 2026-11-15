"""Nonlinear interval bounds on the independent linear-noise-Q candidate box."""
from pathlib import Path
from fractions import Fraction as F
import json,time,hashlib,numpy as np
import interval_taylor as t
OUT=Path(__file__).resolve().parent
radius=[F(0)]*21
for indices,values in [([0,3,7,10,15,19],['.077783','.070442','.010529','.197912','.288661','.326455']),
 ([1,4,6,9,14,18],['.077783','.070442','.010529','.197912','.288661','.326455']),
 ([2,5,12,13,17],['.067302','.055462','.173839','.202997','.254497'])]:
 for i,v in zip(indices,values):radius[i]=F(v)
for i,v in zip([8,11,16,20],['.011805','.414826','.363444','.510264']):radius[i]=F(v)/20
for i in range(21):radius[i]*=F(3) if i in [8,11,16,20] else F(3,2)
r=[t.outer(x,True) for x in radius];d=list(map(t.frac,r));ep=.005;ev=.01
result={'scope':'Real nonlinear derivative/remainder bounds on exact-audited Q-to-box projection; this does NOT certify nonlinear invariance or machine arithmetic.',
 'domain_inflation':{'x_y_z':'3/2','yaw':'3'},'structured_unit_quaternion_controller':True,'flight_certificate':False,'state_component_radii_exact':[str(x) for x in radius],'state_component_radii_binary64_outward':r,
 'position_noise_m':ep,'velocity_noise_mps':ev,'runs':[]}
signed={}
for branch in (0,1):
 start=time.monotonic()
 try:
  y,checks,p=t.backup_map(r,ep,ev,branch,structured_controller=True)
  H=np.array([a.h.absmax() for a in y]);G=np.array([a.g.absmax() for a in y]);eps=[];noise=[]
  for i in range(21):
   eps.append(t.outer(sum((t.frac(H[i,j,k])*d[j]*d[k]/2 for j in range(21) for k in range(21)),F(0)),True))
   noise.append(t.outer(sum((t.frac(G[i,21+j])*t.frac(ep if j<3 else ev) for j in range(6)),F(0)),True))
  for suffix,arr in [('H_lower',[x.h.lo for x in y]),('H_upper',[x.h.hi for x in y]),('J_lower',[x.g.lo for x in y]),('J_upper',[x.g.hi for x in y])]:signed[f'branch{branch}_{suffix}']=np.array(arr)
  row={'branch':branch,'status':'conditional_real_interval_hessian_domain_pass','epsilon_state_nonlinear':eps,
       'epsilon_measurement_noise':noise,'branch_domain_checks':checks,'elapsed_s':time.monotonic()-start}
  print(branch,'stateepsmax',max(eps),'noiseepsmax',max(noise),'sec',row['elapsed_s'],flush=True)
 except t.DomainFailure as e:row={'branch':branch,'status':'domain_failed','reason':str(e)}
 result['runs'].append(row)
np.savez_compressed(OUT/'inflated_structured_noise_Q_signed_derivative_intervals.npz',**signed)
result['source_sha256']={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['interval_taylor.py','run_inflated_q_domain.py','projection_audit.json']}
(OUT/'inflated_structured_noise_Q_taylor_bounds.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
