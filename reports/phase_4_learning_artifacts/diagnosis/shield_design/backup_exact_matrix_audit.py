"""Exact rational audit of stored numerical candidate matrices, NOT model proof."""
from pathlib import Path
from fractions import Fraction as F
import json,numpy as np,time,hashlib
out=Path('reports/phase_4_learning_artifacts/diagnosis/shield_design');meta=json.loads((out/'backup_common_P_feasibility.json').read_text());data=np.load(out/'backup_common_P_candidate.npz');start=time.monotonic()
def frac(x):return F.from_float(float(x))
def ldl_positive(a):
 n=len(a);l=[[F(int(i==j)) for j in range(n)] for i in range(n)];d=[]
 for j in range(n):
  pivot=a[j][j]-sum(l[j][k]*l[j][k]*d[k] for k in range(j))
  if pivot<=0:return False,j,str(pivot)
  d.append(pivot)
  for i in range(j+1,n):l[i][j]=(a[i][j]-sum(l[i][k]*l[j][k]*d[k] for k in range(j)))/pivot
 return True,None,None
result={}
for name,info in meta['groups'].items():
 idx=info['indices'];n=len(idx);s=list(map(frac,info['state_scaling']));p=[[frac(v) for v in row] for row in info['scaled_P']];assert all(p[i][j]==p[j][i] for i in range(n) for j in range(n));eps=F({'planar_x':'0.003','planar_y':'0.003','vertical':'0.001','yaw':'0.008'}[name]);pshift=[[p[i][j]-(F('0.0001') if i==j else 0) for j in range(n)] for i in range(n)];assert ldl_positive(pshift)[0],name
 for vertex,a0 in enumerate(data['local_vertex_jacobians']):
  a=[[frac(a0[idx[i],idx[j]])*s[j]/s[i] for j in range(n)] for i in range(n)];pa=[[sum(p[i][k]*a[k][j] for k in range(n)) for j in range(n)] for i in range(n)];q=[[p[i][j]-sum(a[k][i]*pa[k][j] for k in range(n))-(eps if i==j else 0) for j in range(n)] for i in range(n)];okay,j,pivot=ldl_positive(q)
  if not okay:raise AssertionError((name,vertex,j,pivot))
 m=max(sum(abs(v) for v in row) for row in p);result[name]={'P_minus_margin_I_positive_definite_exact':True,'P_eigenvalue_lower_bound':'0.0001','all_stored_vertices_pass_exact':True,'vertices':len(data['local_vertex_jacobians']),'P_minus_ATPA_eigenvalue_lower_bound':str(eps),'P_eigenvalue_upper_bound_rational':str(m),'P_norm_contraction_squared_upper_bound_rational':str(1-eps/m),'P_norm_contraction_upper_bound_float_display_only':float(1-eps/m)**.5}
 print(name,'exactPASS',len(data['local_vertex_jacobians']),'margin',eps,flush=True)
artifact={'status':'EXACT_RATIONAL_FINITE_MATRIX_POLYTOPE_CERTIFICATE_ONLY','meaning':'Exact LDL over Fractions of stored binary64 candidate values proves P >= .0001I and P-A^T P A >= stated margin I for every stored vertex; matrix convexity extends this to their convex hull. This does NOT prove the true model Jacobians lie in that hull.','covered_real_model':False,'nonlinear_remainder_covered':False,'floating_execution_rounding_covered':False,'source_candidate_sha256':hashlib.sha256((out/'backup_common_P_candidate.npz').read_bytes()).hexdigest(),'blocks':result,'elapsed_s':time.monotonic()-start};(out/'backup_exact_matrix_audit.json').write_text(json.dumps(artifact,indent=2)+'\n');print('EXACTAUDITseconds',artifact['elapsed_s'])
