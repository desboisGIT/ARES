"""Real-arithmetic second-order interval AD of the centered backup map.

Research only. No floating plant execution budget, terminal certificate or RPI
claim. Uses outward IEEE binary64 elementary interval arithmetic, exact rational
sqrt endpoint checks, rational small-atan and motor-exp alternating-series bracketing.
"""
import math
from fractions import Fraction as F
import numpy as np

N=27
class DomainFailure(ValueError):pass

def frac(v):return F.from_float(float(v))
def outer(q,up):
 try:x=float(q)
 except OverflowError as e:raise DomainFailure('outward overflow') from e
 if not math.isfinite(x):raise DomainFailure('nonfinite outward bound')
 while (frac(x)<q if up else frac(x)>q):
  x=math.nextafter(x,math.inf if up else -math.inf)
  if not math.isfinite(x):raise DomainFailure('outward overflow')
 return x

def sqrt_rational(exact,up):
 if exact<0:raise DomainFailure('negative sqrt endpoint')
 if exact==0:return 0.
 r=math.sqrt(float(exact))
 while (frac(r)**2<exact if up else frac(r)**2>exact):r=math.nextafter(r,math.inf if up else -math.inf)
 return r

def sqrt_endpoint(x,up):return sqrt_rational(frac(x),up)

def atan_endpoint(x,up):
 # Alternating series converges for the audited near-hover ratio domain.
 q=frac(x)
 if abs(q)>F(1,2):raise DomainFailure('atan interval exceeds rational-series domain |x|<=.5')
 if q<0:return -atan_endpoint(-x,not up)
 lower=sum(((-1)**k*q**(2*k+1)/F(2*k+1) for k in range(16)),F(0))
 upper=lower+q**33/F(33)
 return outer(upper if up else lower,up)

def exp_endpoint(x,up):
 # Exact alternating-series brackets for exp(-a), 0<=a<=1. Motor arguments
 # lie inside this domain. No libm/Decimal accuracy assumption is needed.
 a=-frac(x)
 if not(0<=a<=1):raise DomainFailure('motor exponential outside rational-series domain [-1,0]')
 lower=sum(((-a)**j/F(math.factorial(j)) for j in range(32)),F(0))
 upper=sum(((-a)**j/F(math.factorial(j)) for j in range(31)),F(0))
 return outer(upper if up else lower,up)

class I:
 def __init__(self,lo,hi=None):self.lo=np.asarray(lo,dtype=float);self.hi=np.asarray(lo if hi is None else hi,dtype=float)
 @classmethod
 def rational(cls,lo,hi=None):return cls(outer(F(lo),False),outer(F(lo if hi is None else hi),True))
 def __add__(self,b):
  if isinstance(b,D):return b+self
  b=iv(b);return I(np.nextafter(self.lo+b.lo,-np.inf),np.nextafter(self.hi+b.hi,np.inf))
 __radd__=__add__
 def __neg__(self):return I(-self.hi,-self.lo)
 def __sub__(self,b):return self+-b if isinstance(b,D) else self+-iv(b)
 def __rsub__(self,b):return iv(b)+-self
 def __mul__(self,b):
  if isinstance(b,D):return b*self
  b=iv(b);vs=np.stack(np.broadcast_arrays(self.lo*b.lo,self.lo*b.hi,self.hi*b.lo,self.hi*b.hi))
  return I(np.nextafter(vs.min(0),-np.inf),np.nextafter(vs.max(0),np.inf))
 __rmul__=__mul__
 def reciprocal(self):
  if np.any((self.lo<=0)&(self.hi>=0)):raise DomainFailure('reciprocal reaches zero')
  return I(np.nextafter(1/self.hi,-np.inf),np.nextafter(1/self.lo,np.inf))
 def __truediv__(self,b):return ad(self)/b if isinstance(b,D) else self*iv(b).reciprocal()
 def square(self):
  lo=np.minimum(self.lo*self.lo,self.hi*self.hi);lo=np.where((self.lo<=0)&(self.hi>=0),0,lo)
  return I(np.maximum(0,np.nextafter(lo,-np.inf)),np.nextafter(np.maximum(self.lo*self.lo,self.hi*self.hi),np.inf))
 def sqrt(self):
  if self.lo.shape!=():raise DomainFailure('only scalar sqrt required')
  return I(sqrt_endpoint(float(self.lo),False),sqrt_endpoint(float(self.hi),True))
 def atan(self):return I(atan_endpoint(float(self.lo),False),atan_endpoint(float(self.hi),True))
 def absmax(self):return np.maximum(abs(self.lo),abs(self.hi))
 def reshape(self,*shape):return I(self.lo.reshape(*shape),self.hi.reshape(*shape))
 def record(self):return {'lo':self.lo.tolist(),'hi':self.hi.tolist()}

def iv(x):return x if isinstance(x,I) else I.rational(F(x))
def iouter(a,b):return a.reshape(N,1)*b.reshape(1,N)

class D:
 def __init__(self,v,g=None,h=None):
  self.v=iv(v);self.g=I(np.zeros(N)) if g is None else g;self.h=I(np.zeros((N,N))) if h is None else h
 @classmethod
 def variable(cls,v,index):
  g=np.zeros(N);g[index]=1.;return cls(v,I(g))
 def __add__(self,b):
  b=ad(b);return D(self.v+b.v,self.g+b.g,self.h+b.h)
 __radd__=__add__
 def __neg__(self):return D(-self.v,-self.g,-self.h)
 def __sub__(self,b):return self+-ad(b)
 def __rsub__(self,b):return ad(b)+-self
 def __mul__(self,b):
  b=ad(b)
  return D(self.v*b.v,self.g*b.v+self.v*b.g,
           self.h*b.v+self.v*b.h+iouter(self.g,b.g)+iouter(b.g,self.g))
 __rmul__=__mul__
 def unary(self,value,first,second):return D(value,first*self.g,first*self.h+second*iouter(self.g,self.g))
 def reciprocal(self):
  value=self.v.reciprocal();return self.unary(value,-value.square(),2*value*value*value)
 def __truediv__(self,b):return self*ad(b).reciprocal()
 def __rtruediv__(self,b):return ad(b)*self.reciprocal()
 def square(self):return self.unary(self.v.square(),2*self.v,I.rational(2))
 def sqrt(self):
  value=self.v.sqrt()
  if value.lo<=0:raise DomainFailure('sqrt derivative not separated from zero')
  return self.unary(value,iv(F(1,2))/value,-iv(F(1,4))/(self.v*value))
 def atan(self):
  value=self.v.atan();den=1+self.v.square()
  return self.unary(value,iv(1)/den,-2*self.v/(den*den))
def ad(x):return x if isinstance(x,D) else D(x)
def summation(x):
 r=ad(0)
 for a in x:r=r+a
 return r
def dot(a,b):return summation([x*y for x,y in zip(a,b)])
def add(a,b):return [x+y for x,y in zip(a,b)]
def scale(a,s):return [x*s for x in a]
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def crotate(q,v):
 a=cross(q[:3],v)
 return add(v,scale(add(scale(a,q[3]),cross(q[:3],a)),2))
def conj(q):return [-q[0],-q[1],-q[2],q[3]]
def rotate(q,v):
 xyz=q[:3];w=q[3]
 return add(add(scale(v,w.square()-dot(xyz,xyz)),scale(cross(xyz,v),2*w)),scale(xyz,2*dot(xyz,v)))
def qmul(a,b):
 x1,y1,z1,w1=a;x2,y2,z2,w2=b
 ww=(z1+x1)*(x2+y2);yy=(w1-y1)*(w2+z2);zz=(w1+y1)*(w2-z2)
 xx=ww+yy+zz;qq=(xx+(z1-x1)*(x2-y2))/2
 return [qq-xx+(x1+w1)*(x2+w2),qq-yy+(w1-x1)*(y2+z2),qq-zz+(z1+y1)*(w2-x2),qq-ww+(z1-y1)*(y2-z2)]
def qchart(xyz):return list(xyz)+[(1-summation([x.square() for x in xyz])).sqrt()]

def params():
 def fp(lo,hi):return I(float(np.nextafter(np.float32(lo),np.float32(-np.inf))),float(np.nextafter(np.float32(hi),np.float32(np.inf))))
 nominal=[I.rational(min(F(str(v)),frac(float(np.float32(v)))),max(F(str(v)),frac(float(np.float32(v))))) for v in [.002,.002,.003]]
 p={'m':fp(.45,.55),'g':fp(9.78,9.83),'drag':fp(.05,.2),'tau':fp(.015,.04),
    'j':[fp(.0018,.0022),fp(.0018,.0022),fp(.0027,.0033)],'jn':nominal,'battery':I.rational(F(19,20),1)}
 h=I.rational(F(1,500));arg=-h/p['tau']
 p['alpha']=I(exp_endpoint(float(arg.lo),False),exp_endpoint(float(arg.hi),True))
 return p

def strict_domain(value,lo,hi,name,checks):
 if not(float(value.lo)>lo and float(value.hi)<hi):
  raise DomainFailure(f'{name} not strictly inside ({lo},{hi}): {value.record()}')
 checks[name]=value.record()

def backup_map(state_radius,noise_position,noise_velocity,branch,point=None,parameter_values=None,noise_point=None):
 p=params()
 if parameter_values is not None:
  for key,value in parameter_values.items():
   if key not in ('delay_fraction',):p[key]=[I(x) for x in value] if isinstance(value,list) else I(value)
  arg=-I.rational(F(1,500))/p['tau']
  p['alpha']=I(exp_endpoint(float(arg.lo),False),exp_endpoint(float(arg.hi),True))
 checks={};z=[]
 for i,r in enumerate(state_radius):z.append(D.variable(I(-r,r) if point is None else I(point[i]),i))
 noise=[D.variable(I(-noise_position,noise_position),21+i) for i in range(3)]
 noise +=[D.variable(I(-noise_velocity,noise_velocity),24+i) for i in range(3)]
 if noise_point is not None:noise=[D.variable(I(x),21+i) for i,x in enumerate(noise_point)]
 q=qchart(z[6:9]);strict_domain(q[3].v,0,math.inf,'positive_initial_qw',checks)
 # Parameter-dependent exact real equilibrium is analytically canceled.
 mg=p['m']*p['g'];teq=mg/p['battery'];force_eq=2*teq
 delta_force=[-F(3,2)*(z[i]+noise[i])-F(5,2)*(z[3+i]+noise[3+i]) for i in range(3)]
 force=list(delta_force);force[2]=force[2]+force_eq
 norm=summation([x.square() for x in force]).sqrt();strict_domain(norm.v,1e-6,math.inf,'force_norm',checks)
 bz=crotate(q,[ad(0),ad(0),ad(1)]);desired=scale(force,norm.reciprocal())
 rates=scale(crotate(conj(q),cross(bz,desired)),6)
 heading=crotate(q,[ad(1),ad(0),ad(0)])
 strict_domain(heading[0].v,1e-4,math.inf,'heading_positive_x_and_valid_mask',checks)
 yaw=(-heading[1]/heading[0]).atan();yawrate=6*yaw
 strict_domain(yawrate.v,-1,1,'heading_rate_unclamped',checks)
 rates=add(rates,crotate(conj(q),[ad(0),ad(0),yawrate]))
 for i,r in enumerate(rates):strict_domain(r.v,-6,6,f'body_rate_{i}_unclamped',checks)
 du=F(1,2)*(dot(delta_force,bz)+force_eq*(bz[2]-1))
 strict_domain((du+teq).v,0,22.,'thrust_policy_and_plant_unclamped',checks)
 for slot in (13,17):
  strict_domain((z[slot]+teq).v,0,22.,f'queue_thrust_{slot}_unclamped',checks)
  for axis in range(1,4):strict_domain(z[slot+axis].v,-6,6,f'queue_rate_{slot}_{axis}_unclamped',checks)
 strict_domain((z[12]+mg).v,0,22.,'initial_motor_positive_bounded',checks)
 action=[du]+rates
 # branch0 delay[.005,.01]: .5..1 between current and previous command.
 # branch1 delay[.01,.02]: 0..1 between previous and second previous.
 fraction=I.rational(F(1,2),1) if branch==0 else I.rational(0,1)
 if parameter_values is not None and 'delay_fraction' in parameter_values:fraction=I(parameter_values['delay_fraction'])
 a,b=(action,z[13:17]) if branch==0 else (z[13:17],z[17:21])
 command=[(1-fraction)*x+fraction*y for x,y in zip(a,b)]
 u=command[1:];motor=z[12];h=F(1,500)
 physical=list(z[:3])+q+list(z[3:6])+list(z[9:12])
 max_compensation=0.
 def dynamics(x,motor):
  nonlocal max_compensation
  qq=x[3:7];v=x[7:10];w=x[10:13]
  nominal_cross=cross(w,[p['jn'][i]*w[i] for i in range(3)])
  cross_squared=sum((frac(float(k.v.absmax()))**2 for k in nominal_cross),F(0))
  if cross_squared>=10000:raise DomainFailure('nominal gyroscopic compensation saturation not excluded')
  cross_bound=sqrt_rational(cross_squared,True)
  max_compensation=max(max_compensation,cross_bound)
  mismatch=cross(w,[(p['jn'][i]-p['j'][i])*w[i] for i in range(3)])
  wdot=[30*p['jn'][i]/p['j'][i]*(u[i]-w[i])+mismatch[i]/p['j'][i] for i in range(3)]
  body_z=rotate(qq,[ad(0),ad(0),ad(1)])
  norm4=summation([x.square() for x in qq]).square()
  acc=[p['g']*(body_z[i]-(1 if i==2 else 0))+(motor/p['m'])*body_z[i]
       -(p['drag']/p['m'])*norm4*v[i] for i in range(3)]
  qdot=scale(qmul(qq,list(w)+[ad(0)]),F(1,2))
  return list(v)+qdot+acc+wdot
 for step in range(5):
  motor=p['alpha']*motor+(1-p['alpha'])*p['battery']*command[0]
  k1=scale(dynamics(physical,motor),h)
  k2=scale(dynamics(add(physical,scale(k1,F(1,2))),motor),h)
  k3=scale(dynamics(add(physical,scale(k2,F(1,2))),motor),h)
  k4=scale(dynamics(add(physical,k3),motor),h)
  nxt=add(physical,scale(add(add(k1,scale(k2,2)),add(scale(k3,2),k4)),F(1,6)))
  raw=nxt[3:7];strict_domain(raw[3].v,0,math.inf,f'positive_raw_qw_{step}',checks)
  raw_norm=summation([x.square() for x in raw]).sqrt();strict_domain(raw_norm.v,1e-12,math.inf,f'normalization_norm_{step}',checks)
  xyz=scale(raw[:3],raw_norm.reciprocal())
  normalized=qchart(xyz)
  physical=nxt[:3]+normalized+nxt[7:]
 checks['max_nominal_compensation_norm_upper']=max_compensation
 output=physical[:3]+physical[7:10]+physical[3:6]+physical[10:13]+[motor]+action+list(z[13:17])
 return output,checks,p
