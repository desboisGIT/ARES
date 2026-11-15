"""NUMERICAL noise-aware invariant ellipsoid search for local linear maps only."""
import sys
sys.path.insert(0,'/tmp/ares-shield-sdp')
import cvxpy as cp,numpy as np,json,time
from pathlib import Path
from fractions import Fraction as F
ns={};exec(Path('/tmp/ares_backup_bernstein_embedding.py').read_text(),ns);out=ns['OUT'];meta=ns['meta'];start=time.monotonic();records=[]
def controlled_AB(h,c,d):
 raw=ns['controlled'](h,c,d);n=len(c);D=np.array(c[:2],float);B=np.zeros((n+2,2));B[:n]=np.array([row[n] for row in h[:n]],float)[:,None]*float(d[0])*D[None,:];B[n]=D;return np.array(raw,float),B
for name,bb,variants in [('planar_x',ns['bp'],ns['variants']),('vertical',ns['bv'],[(r,b,d) for r in [1/ns['MHI'],1/ns['MLO']] for b in [F(19,20),F(1)] for d in ns['delayvertices']()])]:
 As=[];Bs=[];info=meta['groups'][name];sc=np.array(info['state_scaling']);n=len(sc);S=np.diag(sc);Si=np.diag(1/sc)
 for row in bb:
  for core in row:
   for var in variants:
    h=[r.copy() for r in core]
    if name=='planar_x':
     g,c,d=var
     for i in [0,1]:
      for j in [2,3,4]:h[i][j]*=g
     C=[-6*c*F(3,2),-6*c*F(5,2),F(-12),F(0)]
    else:
     invm,b,d=var
     for i in [0,1]:
      for j in [2,3]:h[i][j]*=invm
     for i in [0,1,2]:h[i][3]*=b
     C=[F(-3,4),F(-5,4),F(0)]
    A,B=controlled_AB(h,C,d);As.append(Si@A@S);Bs.append(Si@B)
 As=np.array(As);Bs=np.array(Bs);W=np.diag([2*.005**2,2*.01**2]);Gs=Bs@W@np.swapaxes(Bs,1,2)
 for tau in [.99,.995,.999]:
  active=set(np.linspace(0,len(As)-1,24,dtype=int).tolist());qvalue=None;history=[]
  for it in range(30):
   Q=cp.Variable((n,n),symmetric=True);constraints=[Q>>1e-8*np.eye(n)]+[Q-As[i]@Q@As[i].T/tau-Gs[i]/(1-tau)>>1e-8*np.eye(n) for i in active];prob=cp.Problem(cp.Minimize(cp.trace(Q)),constraints)
   try:prob.solve(solver='CLARABEL',tol_gap_abs=1e-8,tol_feas=1e-9,tol_gap_rel=1e-8,max_iter=250)
   except Exception as err:history.append({'iteration':it,'exception':str(err)});break
   if Q.value is None:history.append({'iteration':it,'status':prob.status});break
   qvalue=Q.value;res=qvalue[None]-As@qvalue@np.swapaxes(As,1,2)/tau-Gs/(1-tau);margin=np.linalg.eigvalsh(res)[:,0];worst=float(margin.min());history.append({'iteration':it,'active':len(active),'min_float_eigenvalue':worst,'status':prob.status})
   if worst>=-1e-7:break
   active.update(np.argsort(margin)[:12].tolist())
  rec={'group':name,'tau':tau,'iterations':history,'local_linear_only':True,'full_coefficient_vertices_checked':len(As),'input_box':{'p_error_m':.005,'v_error_mps':.01}}
  if qvalue is not None:
   rec.update(Q_scaled=qvalue.tolist(),physical_component_radii=np.sqrt(np.diag(S@qvalue@S)).tolist(),minimum_float_eigenvalue=history[-1].get('min_float_eigenvalue'),status=history[-1].get('status'))
  records.append(rec);print('NOISEELLIPSOID',name,tau,{k:v for k,v in rec.items() if k not in ['Q_scaled','iterations']},flush=True)
artifact={'status':'NUMERICAL_LOCAL_LINEAR_NOISE_ELLIPSOID_RESEARCH_ONLY','real_nonlinear_invariance_proven':False,'exact_matrix_audit_done':False,'model_rounding_covered':False,'method':'Find Q with Q-AQA^T/tau-BWB^T/(1-tau) PSD over joint local Bernstein coefficient matrices, W=2diag(ep²,ev²) encloses the 2D measurement box. Cutting-plane SDP then scans all vertices numerically.','records':records,'elapsed_s':time.monotonic()-start};(out/'backup_linear_noise_RPI_feasibility.json').write_text(json.dumps(artifact,indent=2)+'\n');print('DONE',artifact['elapsed_s'],flush=True)
