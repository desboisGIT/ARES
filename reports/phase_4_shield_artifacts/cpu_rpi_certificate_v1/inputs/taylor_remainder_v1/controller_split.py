"""Controller C z + D e + joint nonlinear remainder (noise counted once)."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,numpy as np
import interval_taylor as t
OUT=Path(__file__).resolve().parent
base=json.loads((OUT/'structured_noise_Q_taylor_bounds.json').read_text())
base_exact=list(map(F,base['state_component_radii_exact']));signed={};rows=[]
for label,factor,yaw_factor in [('base_Q_domain',F(1),F(1)),('inflated_Q_domain',F(3,2),F(3))]:
 exact=[a*(yaw_factor if i in [8,11,16,20] else factor) for i,a in enumerate(base_exact)]
 radius=[t.outer(a,True) for a in exact];ep=.005;ev=.01
 full,checks,_=t.backup_map(radius,ep,ev,0,structured_controller=True)
 u=full[13:17];H=np.array([a.h.absmax() for a in u]);joint=list(map(t.frac,radius))+[t.frac(ep)]*3+[t.frac(ev)]*3
 eps=[t.outer(sum((t.frac(H[i,j,k])*joint[j]*joint[k]/2 for j in range(27) for k in range(27)),F(0)),True) for i in range(4)]
 origin,_,_=t.backup_map([0.]*21,0.,0.,0,structured_controller=True)
 C_lo=np.array([a.g.lo[:21] for a in origin[13:17]]);C_hi=np.array([a.g.hi[:21] for a in origin[13:17]])
 D_lo=np.array([a.g.lo[21:] for a in origin[13:17]]);D_hi=np.array([a.g.hi[21:] for a in origin[13:17]])
 # Structural zeros of the REAL local controller, not sampled thresholding.
 c_allowed=[{2,5},{1,4,6},{0,3,7},{8}];d_allowed=[{2,5},{1,4},{0,3},set()]
 for i in range(4):
  for j in range(21):
   if j not in c_allowed[i]:C_lo[i,j]=C_hi[i,j]=0.
  for j in range(6):
   if j not in d_allowed[i]:D_lo[i,j]=D_hi[i,j]=0.
 for suffix,arr in [('H_lower',[a.h.lo for a in u]),('H_upper',[a.h.hi for a in u]),('J_lower',[a.g.lo for a in u]),('J_upper',[a.g.hi for a in u])]:signed[label+'_'+suffix]=np.array(arr)
 row={'label':label,'state_component_radii_exact':[str(x) for x in exact],'state_component_radii_binary64_outward':radius,
      'noise_component_bounds':[ep]*3+[ev]*3,'controller_joint_Taylor_remainder':eps,
      'local_C_parameter_interval_lower':C_lo.tolist(),'local_C_parameter_interval_upper':C_hi.tolist(),
      'local_D_parameter_interval_lower':D_lo.tolist(),'local_D_parameter_interval_upper':D_hi.tolist(),
      'command_deviation_interval':[a.v.record() for a in u],'branch_checks':checks}
 rows.append(row);print(label,'jointNL',eps,'publicCommandPrimalBounds',[float(a.v.absmax()) for a in u],flush=True)
np.savez_compressed(OUT/'controller_split_signed_intervals.npz',**signed)
result={'scope':'REAL u=C(theta)z+D(theta)e+r_ctrl; joint27D Taylor remainder excludes BOTH linear state and linear measurement terms. For linear-noise Q base ΓD e is already counted; add Γ r_ctrl ONLY plus joint physical-flow remainder. No flight certificate/machine bound.',
        'flight_certificate':False,'cases':rows,
        'source_sha256':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['interval_taylor.py','controller_split.py','structured_noise_Q_taylor_bounds.json']}}
(OUT/'controller_split_bounds.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
