"""Exact rational certificates for oriented interval-Hessian remainder bounds."""
from pathlib import Path
from fractions import Fraction as F
import json,numpy as np,time
root=Path(__file__).resolve().parent;out=root/'inputs';base=out/'taylor_remainder_v1';meta=json.loads((out/'backup_common_P_feasibility.json').read_text());qd=json.loads((out/'backup_linear_noise_RPI_feasibility.json').read_text());start=time.monotonic();norm_count=0

def ff(x):return F.from_float(float(x))
def matfloat(a):return np.array(a,dtype=float)
def transpose(a):return list(map(list,zip(*a)))
def mm(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def ab(a):return [[abs(x) for x in r] for r in a]
def ldl(a):
 n=len(a);l=[[F(int(i==j)) for j in range(n)] for i in range(n)];d=[]
 for j in range(n):
  p=a[j][j]-sum(l[j][k]*l[j][k]*d[k] for k in range(j))
  if p<=0:return False
  d.append(p)
  for i in range(j+1,n):l[i][j]=(a[i][j]-sum(l[i][k]*l[j][k]*d[k] for k in range(j)))/p
 return True
def inverse(a):
 n=len(a);x=[row.copy()+[F(int(i==j)) for j in range(n)] for i,row in enumerate(a)]
 for j in range(n):
  p=x[j][j];x[j]=[v/p for v in x[j]]
  for i in range(n):
   if i!=j:
    c=x[i][j];x[i]=[v-c*w for v,w in zip(x[i],x[j])]
 return [row[n:] for row in x]
def opnorm_upper(a):
 global norm_count
 n=len(a);m=len(a[0]);maximum=max(abs(v) for r in a for v in r)
 if maximum==0:return F(0)
 approx=float(np.linalg.norm(matfloat(a),2));upper=ff(approx)*(1+F(1,10**12))+F(1,10**24)
 for iteration in range(8):
  block=[[upper if i==j else F(0) for j in range(n+m)] for i in range(n+m)]
  for i in range(n):
   for j in range(m):block[i][n+j]=block[n+j][i]=a[i][j]
  if ldl(block):norm_count+=1;return upper
  upper=upper*(1+F(1,10**8))+F(1,10**22)
 raise AssertionError('could not certify operator norm')

groups=[];factor_audits={}
for name,info in meta['groups'].items():
 sc=list(map(ff,info['state_scaling']));n=len(sc)
 if name=='yaw':
  p=[[ff(v) for v in r] for r in info['scaled_P']];pi=inverse(p);q=[[F(9,400)*sc[i]*pi[i][j]*sc[j] for j in range(n)] for i in range(n)]
 else:
  rec=next(r for r in qd['records'] if r['group']==('planar_x' if name=='planar_y' else name) and r['tau']==.99);sg=[1,1,-1,-1,-1,-1] if name=='planar_y' else [1]*n;q=[[F(9,4)*sc[i]*ff(v)*sc[j]*sg[i]*sg[j] for j,v in enumerate(row)] for i,row in enumerate(rec['Q_scaled'])]
 approx=np.linalg.cholesky(matfloat(q));inflation=F(1,10**12)
 for it in range(8):
  L=[[ff(v)*(1+inflation) for v in row] for row in approx];cov=mm(L,transpose(L));difference=[[cov[i][j]-q[i][j] for j in range(n)] for i in range(n)]
  if ldl(difference):break
  inflation*=10
 else:raise AssertionError('outer factor failure')
 groups.append((info['indices'],L));factor_audits[name]={'outer_factor_exact':[list(map(str,row)) for row in L],'factor_inflation':str(inflation),'exact_LLT_minus_outer_Q_positive_definite':True,'physical_indices':info['indices']}
 print('FACTOR',name,'PASS',inflation,flush=True)


def oriented(hlo,hhi,allgroups,outputs):
 result=[]
 for output in outputs:
  total=F(0)
  for k,(idx,L) in enumerate(allgroups):
   for ell,(jdx,R) in enumerate(allgroups[k:],start=k):
    # Union symmetric entries ensures a symmetric-Hessian outer interval even
    # if separate interval derivative evaluation has differing last bits.
    lo=[[min(ff(hlo[output,i,j]),ff(hlo[output,j,i])) for j in jdx] for i in idx];hi=[[max(ff(hhi[output,i,j]),ff(hhi[output,j,i])) for j in jdx] for i in idx];mid=[[(x+y)/2 for x,y in zip(lr,hr)] for lr,hr in zip(lo,hi)];rad=[[(y-x)/2 for x,y in zip(lr,hr)] for lr,hr in zip(lo,hi)];M=mm(mm(transpose(L),mid),R);E=mm(mm(ab(transpose(L)),rad),ab(R));bound=opnorm_upper(M)+opnorm_upper(E);total+=bound*(F(1,2) if k==ell else 1)
  result.append(total);print('EXACT_REMAINDER output',output,'upper',float(total),'norm_certificates',norm_count,flush=True)
 return result

# Controller joint state+measurement remainder, not mean-value full noise.
controller_source=base/'controller_split_signed_intervals.npz';H=np.load(controller_source);cg=groups+ [([21+i],[[r]]) for i,r in enumerate([F(1,200)]*3+[F(1,100)]*3)];controller=oriented(H['inflated_Q_domain_H_lower'],H['inflated_Q_domain_H_upper'],cg,range(4))
# Validate that rational outer factors remain in the Hessian domain boxes.
sector=out/'sector_flow_v1';flowmeta=json.loads((sector/'sector_quantized_flow_bounds.json').read_text());state_r=list(map(ff,flowmeta['state_radii']))
for idx,L in groups:
 cov=mm(L,transpose(L))
 for i,j in enumerate(idx):assert cov[i][i]<=state_r[j]**2,(j,float(cov[i][i]),float(state_r[j]**2))
command_r=list(map(F,flowmeta['command_deviation_radii_exact']));fg=groups+[([21+i],[[r]]) for i,r in enumerate(command_r)];FH=np.load(sector/'sector_quantized_flow_derivative_intervals.npz');flows=[]
for branch in [0,1]:
 # Motor and queue maps are exactly affine in fixed-parameter joint inputs.
 eps=oriented(FH[f'branch{branch}_H_lower'],FH[f'branch{branch}_H_upper'],fg,range(12))+[F(0)]*9;flows.append(eps)
artifact={'status':'EXACT_RATIONAL_ORIENTED_INTERVAL_HESSIAN_REMAINDER_CERTIFICATES','method':'Each rational outer factor L satisfies LL^T > Q_outer by exact LDL. Each transformed Hessian midpoint M and interval-radius bound R is assigned an operator norm s proven by exact LDL of [[sI,M],[M^T,sI]]. Product ellipsoid/noise-box quadratic Taylor bounds then sum diagonals by1/2 and cross terms by1. No floating eigensign or norm value is trusted without a rational PSD check.','outer_factor_audits':factor_audits,'controller_joint_remainder_rational':list(map(str,controller)),'controller_joint_remainder_display':list(map(float,controller)),'sector_flow_remainder_rational':[[str(x) for x in row] for row in flows],'sector_flow_remainder_display':[[float(x) for x in row] for row in flows],'operator_norm_certificates':norm_count,'outer_factor_inside_Hessian_domain_verified_exact':True,'nonlinear_RPI_certificate':False,'depends_on_interval_Hessian_implementation_correctness':True,'floating_machine_execution_budget_proven':False,'elapsed_s':time.monotonic()-start};(root/'replayed_exact_oriented_remainder_audit.json').write_text(json.dumps(artifact,indent=2)+'\n');print('DONE',artifact['elapsed_s'],flush=True)
