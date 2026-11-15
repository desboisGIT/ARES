"""Exact rational local-linear useful-noise RPI certificate. Not full plant RPI."""
from pathlib import Path
from fractions import Fraction as F
from math import isqrt
import json,time
ns={};exec(Path('/tmp/ares_backup_bernstein_embedding.py').read_text(),ns);out=ns['OUT'];m=ns['meta'];noise=json.loads((out/'backup_linear_noise_RPI_feasibility.json').read_text());start=time.monotonic()
def ldl(a):
 n=len(a);l=[[F(int(i==j)) for j in range(n)] for i in range(n)];d=[]
 for j in range(n):
  p=a[j][j]-sum(l[j][k]*l[j][k]*d[k] for k in range(j))
  if p<=0:return False
  d.append(p)
  for i in range(j+1,n):l[i][j]=(a[i][j]-sum(l[i][k]*l[j][k]*d[k] for k in range(j)))/p
 return True
def sqrtup(x,digits=6):
 den=10**digits;n=(x.numerator*den**2+x.denominator-1)//x.denominator;q=isqrt(n)
 if q*q<n:q+=1
 return F(q,den)
results={}
for name,bb,variants in [('planar_x',ns['bp'],ns['variants']),('vertical',ns['bv'],[(r,b,d) for r in [1/ns['MHI'],1/ns['MLO']] for b in [F(19,20),F(1)] for d in ns['delayvertices']()])]:
 candidate=next(r for r in noise['records'] if r['group']==name and r['tau']==.99);q=[[F.from_float(float(v)) for v in row] for row in candidate['Q_scaled']];s=list(map(lambda v:F.from_float(float(v)),m['groups'][name]['state_scaling']));n=len(q);tau=F(99,100);assert ldl(q);count=0
 for row in bb:
  for core in row:
   for variant in variants:
    h=[r.copy() for r in core]
    if name=='planar_x':
     g,c,d=variant
     for i in [0,1]:
      for j in [2,3,4]:h[i][j]*=g
     C=[-6*c*F(3,2),-6*c*F(5,2),F(-12),F(0)]
    else:
     invm,b,d=variant
     for i in [0,1]:
      for j in [2,3]:h[i][j]*=invm
     for i in [0,1,2]:h[i][3]*=b
     C=[F(-3,4),F(-5,4),F(0)]
    # Both measurement coefficients share sign, and the whole output B has
    # rank one. Thus e=(+ep,+ev) maximizes |C_p ep+C_v ev| over all four corners.
    assert C[0]<0 and C[1]<0
    du=C[0]*F(1,200)+C[1]*F(1,100);w=[h[i][n-2]*d[0]*du for i in range(n-2)]+[du,F(0)];w=[w[i]/s[i] for i in range(n)];raw=ns['controlled'](h,C,d);a=[[raw[i][j]*s[j]/s[i] for j in range(n)] for i in range(n)];aq=[[sum(a[i][k]*q[k][j] for k in range(n)) for j in range(n)] for i in range(n)];R=[[q[i][j]-sum(aq[i][k]*a[j][k] for k in range(n))/tau-w[i]*w[j]/(1-tau)-(F(1,10**10) if i==j else F(0)) for j in range(n)] for i in range(n)]
    assert ldl(R),(name,count);count+=1
 radii=[sqrtup(q[j][j]*s[j]**2) for j in range(n)];results[name]={'tau_rational':str(tau),'Q_scaled_exact_binary64':candidate['Q_scaled'],'scales_exact_binary64':m['groups'][name]['state_scaling'],'vertices_passed_exact':count,'residual_matrix_eigenvalue_lower_bound':'1/10000000000','state_physical_indices':m['groups'][name]['indices'],'outward_physical_box_radii_rational':list(map(str,radii)),'outward_physical_box_radii_display':list(map(float,radii)),'measurement_component_bounds':{'position_m':'1/200','velocity_mps':'1/100'},'noise_vertex_reduction':'B columns are exactly collinear with C_p,C_v same sign. The rank-one ww^T worst case is attained at aligned box-noise corner; opposite corner identical and mixed corners PSD dominated.'};print('EXACTNOISERPI',name,'PASS',count,'radii',list(map(float,radii)),flush=True)
artifact={'status':'EXACT_FULL_PARAMETER_LOCAL_LINEAR_USEFUL_NOISE_RPI','precise_claim':'For each channel local linear system over the exact joint(A,B) Bernstein hull, Q-AQA^T/tau-(Be)(Be)^T/(1-tau) >= 1e-10 I holds for all bounded position/velocity input errors and every coefficient. Weighted ellipsoid-sum inclusion makes E(Q) robust positively invariant for this LOCAL LINEAR inclusion. Matrix convexity covers the full independently lifted parameter box, not merely sampled profiles.','nonlinear_full_quadrotor_RPI_proven':False,'machine_rounding_budget_proven':False,'authority_equilibrium_drift_covered':False,'measurement_bounds_are_external_contract':True,'blocks':results,'planar_y':'Same local invariant ellipsoid under sign similarity D=diag(1,1,-1,-1,-1,-1).','yaw':'Finite yaw ellipsoid from original exact common-P contraction has no first-order p/v measurement input; nonlinear couplings not covered.','elapsed_s':time.monotonic()-start};(out/'backup_linear_noise_exact_audit.json').write_text(json.dumps(artifact,indent=2)+'\n');print('DONE',artifact['elapsed_s'],flush=True)
