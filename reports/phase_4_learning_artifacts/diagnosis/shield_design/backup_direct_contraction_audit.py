"""Exact direct-metric contraction + ellipsoid projection certificate, LOCAL."""
from pathlib import Path
from fractions import Fraction as F
from math import isqrt
import json,time
ns={};exec(Path('/tmp/ares_backup_bernstein_embedding.py').read_text(),ns);out=ns['OUT'];meta=ns['meta'];start=time.monotonic()
def ldl(a):
 n=len(a);l=[[F(int(i==j)) for j in range(n)] for i in range(n)];d=[]
 for j in range(n):
  v=a[j][j]-sum(l[j][k]*l[j][k]*d[k] for k in range(j))
  if v<=0:return False
  d.append(v)
  for i in range(j+1,n):l[i][j]=(a[i][j]-sum(l[i][k]*l[j][k]*d[k] for k in range(j)))/v
 return True
def inverse(a):
 n=len(a);a=[row.copy()+[F(int(i==j)) for j in range(n)] for i,row in enumerate(a)]
 for j in range(n):
  pivot=a[j][j];a[j]=[x/pivot for x in a[j]]
  for i in range(n):
   if i!=j:
    c=a[i][j];a[i]=[x-c*y for x,y in zip(a[i],a[j])]
 return [row[n:] for row in a]
def sqrt_upper(x,places=14):
 den=10**places;n=(x.numerator*den**2+x.denominator-1)//x.denominator;q=isqrt(n)
 if q*q<n:q+=1
 return F(q,den)
result={}
for name,bb,variants,decorate in [('planar_x',ns['bp'],ns['variants'],ns['planar_decorate']),('vertical',ns['bv'],[(r,b,d) for r in [1/ns['MHI'],1/ns['MLO']] for b in [F(19,20),F(1)] for d in ns['delayvertices']()],ns['vertical_decorate']),('yaw',ns['by'],ns['delayvertices'](),lambda h,d:ns['controlled'](h,[F(-12),F(0)],d))]:
 info=meta['groups'][name];p=[[F.from_float(float(v)) for v in row] for row in info['scaled_P']];s=[F.from_float(float(v)) for v in info['state_scaling']];n=len(p);rho2=F({'planar_x':'.9933','vertical':'.9981','yaw':'.9892'}[name]);assert ldl(p);pi=inverse(p);projection=[s[j]*sqrt_upper(pi[j][j]) for j in range(n)];count=0
 print('PROJECTION',name,list(map(float,projection)),flush=True)
 for row in bb:
  for core in row:
   for variant in variants:
    raw=decorate(core,variant);a=[[raw[ii][jj]*s[jj]/s[ii] for jj in range(n)] for ii in range(n)];pa=[[sum(p[ii][k]*a[k][jj] for k in range(n)) for jj in range(n)] for ii in range(n)];q=[[rho2*p[ii][jj]-sum(a[k][ii]*pa[k][jj] for k in range(n)) for jj in range(n)] for ii in range(n)];assert ldl(q),(name,count);count+=1
 rho=sqrt_upper(rho2);result[name]={'rho_squared_rational':str(rho2),'rho_upper_rational':str(rho),'rho_upper_display':float(rho),'exact_Bernstein_coefficients_passed':count,'physical_indices':info['indices'],'physical_component_halfwidth_per_P_radius_rational':list(map(str,projection)),'physical_component_halfwidth_per_P_radius_display':list(map(float,projection)),'scaled_P_exact_binary64':info['scaled_P'],'state_scales_exact_binary64':info['state_scaling']};print('DIRECTCONTRACTION',name,'PASS',rho2,float(rho),flush=True)
artifact={'status':'EXACT_LOCAL_DIRECT_METRIC_CONTRACTION_ONLY','claim':'Exact rational LDL proves rho_squared*P-A^T*P*A positive definite on every exact Bernstein coefficient matrix, and hence the entire independently lifted local Jacobian family by matrix convexity. Projection coefficients are rational upper bounds from exact P inverse diagonal.','not_a_nonlinear_RPI_certificate':True,'blocks':result,'planar_y':'Sign similarity from planar_x with D=diag(1,1,-1,-1,-1,-1), same absolute projection bounds.','RPI_next_obligation':'For each product-ellipsoid block k, bound nonlinear/noise/authority-drift/rounding residual epsilon_k on the WHOLE product set and prove rho_k*r_k + epsilon_k <= r_k.','elapsed_s':time.monotonic()-start};(out/'backup_direct_contraction_audit.json').write_text(json.dumps(artifact,indent=2)+'\n');print('DONE',artifact['elapsed_s'],flush=True)
