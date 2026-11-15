"""Conditional original eager-CPU binary32 forward error enclosure.

REAL expression range + absolute actual32 deviation. RNE/full precision/FTZ and
primitive exp/sqrt/norm contracts are explicit assumptions, not platform audits.
No simulator edits, factored quaternion map, fitted constants or certificates.
"""
from pathlib import Path
import sys,math,struct
from fractions import Fraction as F
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'taylor_remainder_v1'))
from interval_taylor import I,frac,outer,sqrt_rational,exp_endpoint
U=F(1,2**24);FTZ=F(1,2**126);F32MAX=F.from_float(float(np.finfo(np.float32).max))
class EnclosureFailure(ValueError):pass
COUNTS={}

def count(name):COUNTS[name]=COUNTS.get(name,0)+1
def upper(q):return outer(q,True)
def maximum(x):return float(x.absmax())
def abs_interval(lo,hi):return I(lo,hi)
def up_add(*values):
 q=F(0)
 for x in values:q+=frac(x) if isinstance(x,(float,np.floating)) else F(x)
 return upper(q)
def up_mul(a,b):return upper(frac(a)*frac(b))
def gamma(n):return F(n)*U/(1-F(n)*U)
def f32(value):return struct.unpack('>f',struct.pack('>f',float(value)))[0]
def cast_error_exact(value):
 q=F(value);rounded=f32(float(q));return upper(abs(F.from_float(rounded)-q))
def ulp_upper(maxabs):
 if maxabs==0:return F(1,2**149)
 if not math.isfinite(maxabs) or frac(maxabs)>F32MAX:raise EnclosureFailure('binary32 overflow possible')
 rounded=np.float32(maxabs);nextvalue=np.nextafter(rounded,np.float32(np.inf))
 if not np.isfinite(nextvalue):return F(2)**104
 exponent=math.frexp(float(nextvalue))[1]
 return max(F(1,2**149),F(2)**(exponent-24))
def rounding_error(actual_range):
 a=maximum(actual_range)
 if not math.isfinite(a) or frac(a)*(1+U)>F32MAX:raise EnclosureFailure('binary32 overflow possible')
 return upper(U*frac(a)+FTZ)

def iv(x):return x if isinstance(x,I) else I.rational(F(x))
class ScalarRoundInterval:
 def __init__(self,real,absolute_error=0.):self.real=iv(real);self.error=float(absolute_error)
 @classmethod
 def constant(cls,value):
  q=F(value);count('cast_constant');return cls(I.rational(q),cast_error_exact(q))
 @classmethod
 def input(cls,lo,hi=None,cast=False):
  real=I.rational(F(lo),F(lo if hi is None else hi))
  error=upper(U*frac(maximum(real))+FTZ) if cast else 0.
  if cast:count('cast_input')
  return cls(real,error)
 def actual(self):return self.real+I(-self.error,self.error)
 def __neg__(self):return ScalarRoundInterval(-self.real,self.error)
 def __add__(self,b):
  b=sr(b);count('add')
  real=self.real+b.real;actual=self.actual()+b.actual()
  return ScalarRoundInterval(real,up_add(self.error,b.error,rounding_error(actual)))
 __radd__=__add__
 def __sub__(self,b):return self+-sr(b)
 def __rsub__(self,b):return sr(b)+-self
 def __mul__(self,b):
  b=sr(b);count('multiply')
  prop=up_add(up_mul(maximum(self.real),b.error),up_mul(maximum(b.real),self.error),up_mul(self.error,b.error))
  actual=self.actual()*b.actual()
  return ScalarRoundInterval(self.real*b.real,up_add(prop,rounding_error(actual)))
 __rmul__=__mul__
 def __truediv__(self,b):
  b=sr(b);count('divide')
  if float(b.real.lo)>0:minimum=float(b.real.lo)
  elif float(b.real.hi)<0:minimum=-float(b.real.hi)
  else:raise EnclosureFailure('REAL divisor crosses zero')
  if b.error>=minimum:raise EnclosureFailure('actual32 divisor may cross zero')
  prop=(frac(self.error)+frac(maximum(self.real))*frac(b.error)/frac(minimum))/(frac(minimum)-frac(b.error))
  return ScalarRoundInterval(self.real/b.real,up_add(upper(prop),rounding_error(self.actual()/b.actual())))
 def square(self):
  count('square');actual=self.actual().square()
  prop=2*frac(maximum(self.real))*frac(self.error)+frac(self.error)**2
  return ScalarRoundInterval(self.real.square(),up_add(upper(prop),rounding_error(actual)))
 def sqrt(self):
  count('sqrt_primitive_assumption')
  if float(self.real.lo)<0:raise EnclosureFailure('REAL square-root argument crosses zero')
  real=self.real.sqrt();actual=self.actual();alo=max(0.,float(actual.lo));ahi=float(actual.hi)
  if ahi<0:raise EnclosureFailure('actual square-root input negative')
  actualsqrt=I(sqrt_rational(frac(alo),False),sqrt_rational(frac(ahi),True))
  if float(self.real.lo)>self.error:
   denom=frac(float(real.lo))+frac(sqrt_rational(frac(float(self.real.lo))-frac(self.error),False))
   prop=upper(frac(self.error)/denom)
  else:prop=sqrt_rational(frac(self.error),True)
  # Explicit assumption: sqrt itself is correctly rounded within .5 ULP of
  # exact sqrt on a nonnegative binary32 argument. No installed backend audit.
  primitive=upper(F(1,2)*ulp_upper(maximum(actualsqrt))+FTZ)
  return ScalarRoundInterval(real,up_add(prop,primitive))
 def exp(self,extra_ulp_vs_rne=1):
  count('exp_primitive_assumption' if extra_ulp_vs_rne else 'verified_exp_backend')
  a=self.actual()
  if float(a.lo)<-.25 or float(a.hi)>0:raise EnclosureFailure('motor exp actual input leaves [-.25,0]')
  real=I(exp_endpoint(float(self.real.lo),False),exp_endpoint(float(self.real.hi),True))
  actual=I(exp_endpoint(float(a.lo),False),exp_endpoint(float(a.hi),True))
  prop=up_mul(float(actual.hi),self.error)
  primitive=upper((F(extra_ulp_vs_rne)+F(1,2))*ulp_upper(maximum(actual))+FTZ)
  return ScalarRoundInterval(real,up_add(prop,primitive))
 def record(self):return {'real_lower':float(self.real.lo),'real_upper':float(self.real.hi),'absolute_actual32_error':self.error}
def sr(x):return x if isinstance(x,ScalarRoundInterval) else ScalarRoundInterval.constant(x)
def add(a,b):return [x+y for x,y in zip(a,b)]
def scale(a,s):return [x*s for x in a]
def dot(a,b):
 """Compound full-precision sum of three products, arbitrary FMA association."""
 n=len(a);count(f'dot{n}_gamma{2*n}')
 if n not in (3,4):raise EnclosureFailure('only short dot contracts supported')
 real=I.rational(0);prop=F(0);magnitude=F(0)
 for x,y in zip(a,b):
  real=real+x.real*y.real
  prop+=frac(maximum(x.real))*frac(y.error)+frac(maximum(y.real))*frac(x.error)+frac(x.error)*frac(y.error)
  magnitude+=frac(maximum(x.actual()))*frac(maximum(y.actual()))
 local=gamma(2*n)*magnitude+F(4*n)*FTZ
 return ScalarRoundInterval(real,upper(prop+local))
def sum_squares(a):
 n=len(a);count(f'norm_sum{n}_gamma{2*n}')
 real=I.rational(0);prop=F(0);magnitude=F(0)
 for x in a:
  real=real+x.real.square()
  prop+=2*frac(maximum(x.real))*frac(x.error)+frac(x.error)**2
  magnitude+=frac(maximum(x.actual()))**2
 return ScalarRoundInterval(I(max(0.,float(real.lo)),float(real.hi)),upper(prop+gamma(2*n)*magnitude+F(4*n)*FTZ))
def norm(a):return sum_squares(a).sqrt()
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def rotate(q,v):
 # ORIGINAL vendor homogeneous expression, with no q_in/r factorization.
 w=q[3];xyz=q[:3]
 c=w.square()-sum_squares(xyz)
 return add(add(scale(v,c),scale(cross(xyz,v),sr(2)*w)),scale(xyz,sr(2)*dot(xyz,v)))
def inverse(q):return [-x for x in q[:3]]+[q[3]]
def qmul(a,b):
 # ORIGINAL optimized vendor polynomial, not Hamilton simplification.
 x1,y1,z1,w1=a;x2,y2,z2,w2=b
 ww=(z1+x1)*(x2+y2);yy=(w1-y1)*(w2+z2);zz=(w1+y1)*(w2-z2)
 xx=ww+yy+zz;qq=sr(F(1,2))*(xx+(z1-x1)*(x2-y2))
 return [qq-xx+(x1+w1)*(x2+w2),qq-yy+(w1-x1)*(y2+z2),qq-zz+(z1+y1)*(w2-x2),qq-ww+(z1-y1)*(y2-z2)]
def diag_matvec(diagonal,v):
 # The original matrix product is a3dot, including the exact zero entries.
 zero=sr(0)
 return [dot([diagonal[i] if i==j else zero for j in range(3)],v) for i in range(3)]

def convex_real(a,b,f):
 values=[(1-frac(ff))*frac(aa)+frac(ff)*frac(bb) for ff in [float(f.lo),float(f.hi)] for aa in [float(a.lo),float(a.hi)] for bb in [float(b.lo),float(b.hi)]]
 return I(outer(min(values),False),outer(max(values),True))
def motor_update(old,target,decay,dependency_refinement=True):
 count('motor_original_three_operations')
 raw=target+(old-target)*decay
 if not dependency_refinement:return raw
 amax=maximum(decay.real);amin=float(decay.real.lo)
 difference=maximum(old.real-target.real)
 prop=frac(amax)*frac(old.error)+(1-frac(amin))*frac(target.error)+frac(difference)*frac(decay.error)
 prop+=(frac(old.error)+frac(target.error))*frac(decay.error)
 # Same original three operations, but local arithmetic error is evaluated
 # separately from correlated input perturbations to preserve motor stability.
 aa=ScalarRoundInterval(target.actual());bb=ScalarRoundInterval(old.actual());dd=ScalarRoundInterval(decay.actual())
 local=(aa+(bb-aa)*dd).error
 real=convex_real(target.real,old.real,decay.real)
 return ScalarRoundInterval(real,up_add(upper(prop),local))

def normalize_shell_bound(actual_raw_norm_lower):
 """Rational norm4 gamma8+sqrt/div assumptions imply shell<8u here."""
 q=frac(actual_raw_norm_lower)
 if q<=0:raise EnclosureFailure('actual raw quaternion norm not separated from zero')
 a=gamma(8)+8*FTZ/(q*q)
 if a>=1:raise EnclosureFailure('normalization relative sum error invalid')
 nrel=a/(2-a)+2*U*(1+a)+FTZ/q
 if nrel>=1:raise EnclosureFailure('normalization relative norm error invalid')
 shell=max((1+U)/(1-nrel)-1,1-(1-U)/(1+nrel))+2*FTZ
 if shell>8*U:raise EnclosureFailure('8u normalization shell not established')
 return shell

def actual_norm_lower(q):
 total=F(0)
 for x in q:
  a=x.actual();lo=float(a.lo);hi=float(a.hi)
  minimum=0. if lo<=0<=hi else min(abs(lo),abs(hi))
  total+=frac(minimum)**2
 return sqrt_rational(total,False)
def qw_shell_error(xyz,qw_real,shell):
 eps=F(shell);numerator=2*eps+eps*eps;sum_actual=F(0)
 for x in xyz:
  m=frac(maximum(x.real));e=frac(x.error)
  numerator+=2*m*e+e*e;sum_actual+=(m+e)**2
 lower_actual=(1-eps)**2-sum_actual
 if lower_actual<=0:raise EnclosureFailure('positive actual quaternion hemisphere not established')
 denominator=frac(float(qw_real.lo))+frac(sqrt_rational(lower_actual,False))
 return upper(numerator/denominator)
def normalize(q,range_refinement=True,error_shell_refinement=True):
 lower=actual_norm_lower(q);shell=normalize_shell_bound(lower)
 if float(q[3].actual().lo)<=0:raise EnclosureFailure('actual quaternion hemisphere not positive')
 n=norm(q)
 if float(n.actual().lo)<=1e-12:raise EnclosureFailure('normalization clamp branch may activate')
 result=[x/n for x in q];before=result[3].error
 if range_refinement:
  sq=I.rational(0)
  for x in result[:3]:sq=sq+x.real.square()
  if float(sq.hi)>=1:raise EnclosureFailure('unit-real quaternion xyz box reaches hemisphere boundary')
  real=I(sqrt_rational(1-frac(float(sq.hi)),False),sqrt_rational(1-max(F(0),frac(float(sq.lo))),True))
  lo=max(float(real.lo),float(result[3].real.lo));hi=min(float(real.hi),float(result[3].real.hi))
  if lo>hi:raise EnclosureFailure('empty quaternion unit-range intersection')
  result[3]=ScalarRoundInterval(I(lo,hi),result[3].error)
 if error_shell_refinement:
  result[3].error=min(result[3].error,qw_shell_error(result[:3],result[3].real,8*U))
 return result,{'raw_actual_norm_lower':lower,'derived_norm_shell_upper':upper(shell),
                'used_shell_upper':float(8*U),'qw_forward_error_before':before,'qw_error_after':result[3].error}

def interpolate_history(history,branch,delay_error):
 """Continuous fractional-delay map plus original Float32 mix error.

 The ideal piecewise interpolation is globally Lipschitz in delay measured in
 frames, even at integer boundaries. This avoids assuming a stable floor index.
 """
 f=I.rational(F(1,2),1) if branch==0 else I.rational(0,1)
 left,right=(history[0],history[1]) if branch==0 else (history[1],history[2])
 result=[]
 for axis in range(4):
  real=convex_real(left[axis].real,right[axis].real,f)
  slope=max(maximum(history[i][axis].real-history[i+1][axis].real) for i in range(2))
  input_error=max(x[axis].error for x in history)
  magnitude=max(maximum(x[axis].actual()) for x in history)
  # Six full-precision operations conservatively cover fraction/weight/product
  # and summation associations; branch crossing is separately Lipschitz-bounded.
  error=frac(slope)*frac(delay_error)+frac(input_error)+gamma(6)*2*frac(magnitude)+12*FTZ
  result.append(ScalarRoundInterval(real,upper(error)))
 count('fractional_queue_global_lipschitz_gamma6')
 return result

def original_dynamics(state,upstream,p,checks):
 q=state[3:7];v=state[7:10];w=state[10:13]
 nominal_Jw=diag_matvec(p['jn'],w)
 nominal_cross=cross(w,nominal_Jw)
 saturation_norm=norm(nominal_cross)
 saturation_ratio=saturation_norm/sr(100)
 if float(saturation_ratio.actual().hi)>=1:raise EnclosureFailure('actual gyroscopic saturation branch not excluded')
 checks['maximum_compensation_norm_actual_upper']=max(checks.get('maximum_compensation_norm_actual_upper',0.),float(saturation_norm.actual().hi))
 # Denominator max(norm/100,1) equals represented1 exactly on this domain.
 err=[upstream[i+1]-w[i] for i in range(3)]
 angacc=scale(err,sr(30));compensation=[x/sr(1) for x in nominal_cross]
 torque=add(diag_matvec(p['jn'],angacc),compensation)
 actual_Jw=diag_matvec(p['j'],w);moment=[x-y for x,y in zip(torque,cross(w,actual_Jw))]
 wdot=diag_matvec(p['j_inv'],moment)
 # ORIGINAL relative-thrust multiply path; no mass/gravity cancellation.
 thrust=(upstream[0]*p['g'])*p['m']
 body_z=rotate(q,[sr(0),sr(0),sr(1)])
 thrust_acc=scale(body_z,thrust/p['m'])
 drag_body=diag_matvec([p['drag']]*3,rotate(inverse(q),v))
 drag_world=rotate(q,drag_body)
 acc=[thrust_acc[i]+(-p['g'] if i==2 else sr(0))-drag_world[i]/p['m'] for i in range(3)]
 qdot=scale(qmul(q,w+[sr(0)]),sr(F(1,2)))
 return v+qdot+acc+wdot

def original_substep(state,motor,command,decay,p,h,checks,range_refinement=True,error_shell_refinement=True,dependency_motor=True):
 motor=motor_update(motor,command[0],decay,dependency_motor)
 upstream=[motor/(p['m']*p['g'])]+command[1:]
 def dynamics(x):return original_dynamics(x,upstream,p,checks)
 k1=scale(dynamics(state),h)
 k2=scale(dynamics(add(state,scale(k1,sr(F(1,2))))),h)
 k3=scale(dynamics(add(state,scale(k2,sr(F(1,2))))),h)
 k4=scale(dynamics(add(state,k3)),h)
 # Match the original left-associated vendor RK4 expression.
 addition=add(add(add(k1,scale(k2,sr(2))),scale(k3,sr(2))),k4)
 new=add(state,[x/sr(6) for x in addition])
 q,normcheck=normalize(new[3:7],range_refinement,error_shell_refinement)
 checks.setdefault('normalization_checks',[]).append(normcheck)
 return new[:3]+q+new[7:],motor
