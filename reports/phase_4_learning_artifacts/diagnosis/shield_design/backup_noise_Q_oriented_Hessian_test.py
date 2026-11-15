"""Floating-point research bound retaining product-ellipsoid correlations."""
from pathlib import Path
import numpy as np,json,time
root=Path.cwd();out=root/'reports/phase_4_learning_artifacts/diagnosis/shield_design';base=root/'reports/phase_4_gate_artifacts/shield_research/taylor_remainder_v1';h=json.loads((base/'noise_Q_taylor_bounds.json').read_text());signed=np.load(base/'noise_Q_signed_derivative_intervals.npz');meta=json.loads((out/'backup_common_P_feasibility.json').read_text());rows=[]; qdata=json.loads((out/'backup_linear_noise_RPI_feasibility.json').read_text())
for run in h['runs']:
 if 'epsilon_state_nonlinear' not in run:continue
 key='branch'+str(run['branch']);lo=signed[key+'_H_lower'][:,:21,:21];hi=signed[key+'_H_upper'][:,:21,:21];center=(lo+hi)/2;radius=(hi-lo)/2;groups=[]
 for name,info in meta['groups'].items():
  sc=np.array(info['state_scaling']);p=np.array(info['scaled_P']);
  if name=='yaw':q=.05**2*np.diag(sc)@np.linalg.inv(p)@np.diag(sc)
  else:
   record=next(a for a in qdata['records'] if a['group']==('planar_x' if name=='planar_y' else name) and a['tau']==.99);q=np.diag(sc)@np.array(record['Q_scaled'])@np.diag(sc)
   if name=='planar_y':
    D=np.diag([1,1,-1,-1,-1,-1]);q=D@q@D
  groups.append((info['indices'],np.linalg.cholesky(q)))
 eps=np.zeros(21)
 for i in range(21):
  total=0.
  for k,(idx,L) in enumerate(groups):
   for ell,(jdx,R) in enumerate(groups[k:],start=k):
    mid=L.T@center[i][np.ix_(idx,jdx)]@R;err=np.abs(L.T)@radius[i][np.ix_(idx,jdx)]@np.abs(R);bound=np.linalg.norm(mid,2)+np.linalg.norm(err,2);total+=bound*(.5 if k==ell else 1.)
  eps[i]=total
 blocks={}
 for name,info in meta['groups'].items():
  p=np.abs(np.array(info['scaled_P']));v=eps[info['indices']]/np.array(info['state_scaling']);blocks[name]=float(np.sqrt(v@p@v))
 result={'label':'noise_Q_domain','branch':run['branch'],'old_component_eps':run['epsilon_state_nonlinear'],'oriented_component_eps':eps.tolist(),'oriented_block_Pnorm_bounds':blocks,'method_is_float_research_not_a_proof':True};rows.append(result);print(key,'queueeps old/new',[run['epsilon_state_nonlinear'][j] for j in [13,14,15,16]],eps[[13,14,15,16]],'Pnorms',blocks,flush=True)
(out/'backup_noise_Q_oriented_Hessian_feasibility.json').write_text(json.dumps({'status':'FLOAT_RESEARCH_ONLY','source_signed_H':str((base/'noise_Q_signed_derivative_intervals.npz').relative_to(root)),'scope':'Noise-aware Q product ellipsoids for planar/vertical plus original yaw P-radius .05. Signed Hessian intervals are bounded on this whole product; output block P-norms are diagnostic comparisons, not a nonlinear RPI certificate. No floating execution proof.','records':rows},indent=2)+'\n')
