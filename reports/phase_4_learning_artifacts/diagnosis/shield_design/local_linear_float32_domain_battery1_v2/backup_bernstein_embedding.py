"""Exact polynomial/Bernstein enclosure of LOCAL linear backup channel maps.

Not a nonlinear, measured-state, compiled arithmetic or terminal RPI proof.
"""
from fractions import Fraction as F
from math import comb,factorial
from pathlib import Path
import numpy as np,json,time
start=time.monotonic();OUT=Path('reports/phase_4_learning_artifacts/diagnosis/shield_design');meta=json.loads((OUT/'backup_common_P_feasibility.json').read_text())
class Poly:
 def __init__(self,d=0):self.d={k:F(v) for k,v in d.items() if v} if isinstance(d,dict) else ({(0,0):F(d)} if d else {})
 def __add__(self,b):
  if not isinstance(b,Poly):b=Poly(b)
  d=self.d.copy()
  for k,v in b.d.items():d[k]=d.get(k,F(0))+v
  return Poly(d)
 __radd__=__add__
 def __neg__(self):return Poly({k:-v for k,v in self.d.items()})
 def __sub__(self,b):return self+-b if isinstance(b,Poly) else self+(-F(b))
 def __mul__(self,b):
  if not isinstance(b,Poly):b=Poly(b)
  d={}
  for (i,j),x in self.d.items():
   for (k,l),y in b.d.items():d[(i+k,j+l)]=d.get((i+k,j+l),F(0))+x*y
  return Poly(d)
 __rmul__=__mul__
 def evaluate(self,t,u):return sum(float(v)*float(t)**i*float(u)**j for (i,j),v in self.d.items())
def zeros(n,m):return [[Poly() for j in range(m)] for i in range(n)]
def eye(n):return [[Poly(int(i==j)) for j in range(n)] for i in range(n)]
def mm(a,b):return [[sum((a[i][k]*b[k][j] for k in range(len(b))),Poly()) for j in range(len(b[0]))] for i in range(len(a))]
def plus(a,b):return [[x+y for x,y in zip(ar,br)] for ar,br in zip(a,b)]
def times(a,c):return [[x*c for x in row] for row in a]
def rk4(a):
 result=eye(len(a));power=eye(len(a))
 for k in range(1,5):power=mm(power,a);result=plus(result,times(power,F(1,500)**k/F(factorial(k))))
 return result
def power(a,n):
 y=eye(len(a))
 for i in range(n):y=mm(y,a)
 return y
def bern(p,n,m):
 if not p.d:return [[F(0)]*(m+1) for i in range(n+1)]
 assert max(i for i,j in p.d)<=n and max(j for i,j in p.d)<=m
 temp=[[sum((v*F(comb(i,a),comb(n,a)) for (a,b0),v in p.d.items() if b0==b and a<=i),F(0)) for b in range(m+1)] for i in range(n+1)]
 return [[sum((temp[i][b]*F(comb(j,b),comb(m,b)) for b in range(j+1)),F(0)) for j in range(m+1)] for i in range(n+1)]
def matrix_bern(a,n,m):
 tensors=[[bern(v,n,m) for v in row] for row in a]
 return [[[ [tensors[r][c][i][j] for c in range(len(a[0]))] for r in range(len(a))] for j in range(m+1)] for i in range(n+1)]
def delayvertices():return [(F(1,2),F(1,2),F(0)),(F(0),F(1),F(0)),(F(0),F(0),F(1))]
def controlled(h,c,delay):
 n=len(c);a=[[F(0)]*(n+2) for i in range(n+2)];new,one,two=delay
 for i in range(n):
  for j in range(n):a[i][j]=h[i][j]+h[i][n]*new*c[j]
  a[i][n]=h[i][n]*one;a[i][n+1]=h[i][n]*two
 for j in range(n):a[n][j]=c[j]
 a[n+1][n]=F(1)
 return a
def analyze(name,core,n,m,endpoint_variants,decorate):
 b=matrix_bern(core,n,m);p=np.array(meta['groups'][name]['scaled_P']);sc=np.array(meta['groups'][name]['state_scaling']);S=np.diag(sc);Si=np.diag(1/sc);worst=(float('inf'),None);maximum=0.;matrices=[];count=0
 for i in range(n+1):
  for j in range(m+1):
   for variant in endpoint_variants:
    a=decorate(b[i][j],variant);raw=np.array(a,dtype=float);scaled=Si@raw@S;margin=float(np.linalg.eigvalsh(p-scaled.T@p@scaled).min());count+=1
    if margin<worst[0]:worst=(margin,[i,j,variant])
    maximum=max(maximum,float(np.linalg.norm(scaled,2)));matrices.append(raw)
 result={'Bernstein_matrices':count,'degrees':[n,m],'minimum_decrease_eigenvalue_float64':worst[0],'worst_vertex':str(worst[1]),'all_floating_checks_positive':worst[0]>0,'max_scaled_A_norm':maximum};np.savez(OUT/(name+'_Bernstein_vertices.npz'),matrices=np.array(matrices));print(name,result,flush=True);return result,b
# Exact rational physical coefficient intervals, with conservative independence.
def outward32(lo,hi):
 return (F.from_float(float(np.nextafter(np.float32(lo),np.float32(-np.inf)))),
         F.from_float(float(np.nextafter(np.float32(hi),np.float32(np.inf)))))
MLO,MHI=outward32(.45,.55);GLO,GHI=outward32(9.78,9.83);DRLO,DRHI=outward32(.05,.2);TAULO,TAUHI=outward32(.015,.04)
DLO,DHI=DRLO/MHI,DRHI/MLO
j_bounds=[outward32(.0018,.0022),outward32(.0018,.0022),outward32(.0027,.0033)]
j_nom=[(min(F(str(x)),F.from_float(float(np.float32(x)))),max(F(str(x)),F.from_float(float(np.float32(x))))) for x in [.002,.002,.003]]
KLO=min(30*n[0]/j[1] for n,j in zip(j_nom,j_bounds));KHI=max(30*n[1]/j[0] for n,j in zip(j_nom,j_bounds))
delta=Poly(DLO)+Poly({(1,0):DHI-DLO});k=Poly(KLO)+Poly({(0,1):KHI-KLO})
a=zeros(5,5);a[0][1]=Poly(1);a[1][1]=-delta;a[1][2]=Poly(2);a[2][3]=Poly(F(1,2));a[3][3]=-k;a[3][4]=k
planar=power(rk4(a),5);g_ends=[GLO,GHI];c_ends=[F(1,2)/(MHI*GHI),F(1,2)/(MLO*GLO)]
variants=[(g,c,delay) for g in g_ends for c in c_ends for delay in delayvertices()]
def planar_decorate(h,var):
 g,c,d=var;h=[row.copy() for row in h]
 for i in [0,1]:
  for j in [2,3,4]:h[i][j]*=g
 return controlled(h,[-6*c*F(3,2),-6*c*F(5,2),F(-12),F(0)],d)
results={};results['planar_x'],bp=analyze('planar_x',planar,20,20,variants,planar_decorate)
# Exact alternating-series brackets for the real-valued motor exponential.
def expneg(x,n):return sum(((-x)**j/F(factorial(j)) for j in range(n+1)),F(0))
alpha_lo=expneg(F(1,500)/TAULO,31);alpha_hi=expneg(F(1,500)/TAUHI,30);alpha=Poly(alpha_lo)+Poly({(0,1):alpha_hi-alpha_lo})
plant=zeros(3,3);plant[0][1]=Poly(1);plant[1][1]=-delta;plant[1][2]=Poly(1);phi=rk4(plant);sub=zeros(4,4)
for i in [0,1]:
 for j in [0,1]:sub[i][j]=phi[i][j]
 sub[i][2]=phi[i][2]*alpha;sub[i][3]=phi[i][2]*(Poly(1)-alpha)
sub[2][2]=alpha;sub[2][3]=Poly(1)-alpha;sub[3][3]=Poly(1);vertical=power(sub,5)
def vertical_decorate(h,var):
 invm,d=var;h=[row.copy() for row in h]
 for i in [0,1]:
  for j in [2,3]:h[i][j]*=invm
 return controlled(h,[F(-3,4),F(-5,4),F(0)],d)
results['vertical'],bv=analyze('vertical',vertical,20,5,[(r,d) for r in [1/MHI,1/MLO] for d in delayvertices()],vertical_decorate)
# Yaw k range differs only if Jnom/J endpoints use proportional z inertia,
# which they do in this profile, giving the same ratio interval.
yaw=zeros(3,3);yaw[0][1]=Poly(F(1,2));yaw[1][1]=-k;yaw[1][2]=k;yawcore=power(rk4(yaw),5)
results['yaw'],by=analyze('yaw',yawcore,0,20,list(delayvertices()),lambda h,d:controlled(h,[F(-12),F(0)],d))
# Export exact core power coefficients; these plus the displayed symbolic
# affine endpoint transformations are sufficient to reproduce all rational B.
poly_artifact={name:[[[[i,j,str(v)] for (i,j),v in sorted(p.d.items())] for p in row] for row in core] for name,core in [('planar_g1',planar),('vertical_invm1',vertical),('yaw',yawcore)]}
(OUT/'backup_local_exact_power_coefficients.json').write_text(json.dumps(poly_artifact)+'\n')
artifact={'status':'EXACT_SYMBOLIC_LOCAL_POLYNOMIAL_EMBEDDING_WITH_FLOAT_MATRIX_AUDIT','scope':'Exact rational RK4/motor-polynomial formula and Bernstein core coefficients enclose whole independently lifted parameter boxes for the local linear model. Eigenvalue checks of existing P on exported matrices remain float64 until rational LDL audit. Does not cover nonlinear state/measurement/numerical execution.','results':results,'motor_alpha_lower_rational':str(alpha_lo),'motor_alpha_upper_rational':str(alpha_hi),'outward_float32_physical_bounds':{name:[str(a),str(b)] for name,a,b in [('mass',MLO,MHI),('gravity',GLO,GHI),('drag',DRLO,DRHI),('motor_lag',TAULO,TAUHI),('drag_over_mass',DLO,DHI),('angular_rate_gain',KLO,KHI)]},'queue_fraction_contract':[.5,2.],'exact_real_substep_s':'1/500','elapsed_s':time.monotonic()-start};(OUT/'backup_Bernstein_embedding_feasibility.json').write_text(json.dumps(artifact,indent=2)+'\n');print('DONE',artifact['elapsed_s'],flush=True)
