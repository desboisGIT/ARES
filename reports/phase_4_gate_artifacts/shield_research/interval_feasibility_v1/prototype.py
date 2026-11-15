"""Conditional interval research, not a compiled-CUDA numerical certificate.

Outward float64 host endpoints plus conservative FP32 primitive rounding padding.
Assumes fixed expression DAG, round-to-nearest arithmetic and stated elementary
function envelopes. FMA contraction is enclosed by separately rounded interval
subexpressions; arbitrary fast-math reassociation is NOT certified. No ares edits.
"""
import sys
sys.path.insert(0,'/home/ares-user/Documents/projects/ARES')
import math,json,time,hashlib
from pathlib import Path
import numpy as np
U=2.**-24
TINY=float(np.nextafter(np.float32(0),np.float32(1)))
ROUND=True
class I:
 def __init__(self,lo,hi=None):
  self.lo=np.asarray(lo,dtype=np.float64);self.hi=np.asarray(lo if hi is None else hi,dtype=np.float64)
 def __getitem__(self,k):return I(self.lo[k],self.hi[k])
 def __add__(self,o):
  o=iv(o);return rounded(self.lo+o.lo,self.hi+o.hi)
 __radd__=__add__
 def __neg__(self):return I(-self.hi,-self.lo)
 def __sub__(self,o):return self+-iv(o)
 def __rsub__(self,o):return iv(o)+-self
 def __mul__(self,o):
  o=iv(o);p=np.array([self.lo*o.lo,self.lo*o.hi,self.hi*o.lo,self.hi*o.hi]);return rounded(p.min(0),p.max(0))
 __rmul__=__mul__
 def __truediv__(self,o):
  o=iv(o)
  if np.any(o.lo<=0) and np.any(o.hi>=0):
   if np.any((o.lo<=0)&(o.hi>=0)):raise ArithmeticError('interval denominator reaches zero')
  return self*rounded(1/o.hi,1/o.lo)
 def square(self):
  low=np.minimum(self.lo*self.lo,self.hi*self.hi);low=np.where((self.lo<=0)&(self.hi>=0),0,low)
  return rounded(low,np.maximum(self.lo*self.lo,self.hi*self.hi))
 def sqrt(self):
  if np.any(self.hi<0):raise ArithmeticError('negative square root')
  return rounded(np.sqrt(np.maximum(0,self.lo)),np.sqrt(np.maximum(0,self.hi)),4)
 def exp(self):return rounded(np.exp(self.lo),np.exp(self.hi),32)
 def atan(self):return rounded(np.arctan(self.lo),np.arctan(self.hi),32)
 def clamp(self,low,high):return I(np.clip(self.lo,low,high),np.clip(self.hi,low,high))
 def sum(self):
  n=self.lo.size;mag=np.maximum(np.abs(self.lo),np.abs(self.hi)).sum()
  # Covers arbitrary accumulation association of this explicit reduction.
  pad=((n+2)*U+8*np.finfo(float).eps)*mag+TINY*n if ROUND else 8*np.finfo(float).eps*mag
  return I(np.nextafter(self.lo.sum()-pad,-np.inf),np.nextafter(self.hi.sum()+pad,np.inf))
 @property
 def radius(self):return (self.hi-self.lo)/2
 def out(self):return {'lo':self.lo.tolist(),'hi':self.hi.tolist()}
def iv(v):return v if isinstance(v,I) else I(np.asarray(v,dtype=np.float32).astype(np.float64))
def rounded(lo,hi,k=2):
 lo,hi=np.asarray(lo),np.asarray(hi)
 pad=(k*U*np.maximum(np.abs(lo),np.abs(hi))+TINY) if ROUND else 0
 return I(np.nextafter(lo-pad,-np.inf),np.nextafter(hi+pad,np.inf))
def ranges(lo,hi):
 return I(np.nextafter(np.asarray(lo,dtype=np.float32),np.float32(-np.inf)).astype(float),np.nextafter(np.asarray(hi,dtype=np.float32),np.float32(np.inf)).astype(float))
def stack(values):return I(np.array([iv(v).lo for v in values]),np.array([iv(v).hi for v in values]))
def cat(values):return I(np.concatenate([v.lo for v in values]),np.concatenate([v.hi for v in values]))
def cross(a,b):return stack([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
def crotate(q,v):
 xyz=q[:3];a=cross(xyz,v)
 return v+2*(q[3]*a+cross(xyz,a))
def rotate(q,v):
 xyz=q[:3];w=q[3]
 return v*(w.square()-xyz.square().sum())+2*w*cross(xyz,v)+2*xyz*(xyz*v).sum()
def inv(q):return cat([-q[:3],q[3:4]])
def mul(a,b):
 x1,y1,z1,w1=[a[i] for i in range(4)];x2,y2,z2,w2=[b[i] for i in range(4)]
 ww=(z1+x1)*(x2+y2);yy=(w1-y1)*(w2+z2);zz=(w1+y1)*(w2-z2)
 xx=ww+yy+zz;qq=.5*(xx+(z1-x1)*(x2-y2))
 return stack([qq-xx+(x1+w1)*(x2+w2),qq-yy+(w1-x1)*(y2+z2),qq-zz+(z1+y1)*(w2-x2),qq-ww+(z1-y1)*(y2-z2)])
PARAM={'m':ranges(.45,.55),'j':ranges([.0018,.0018,.0027],[.0022,.0022,.0033]),'drag':ranges(.05,.2),'g':ranges(9.78,9.83),'lag':ranges(.015,.04),'tmax':ranges(22.,27.)}
JC=iv([.002,.002,.003]);UP=iv([0.,0.,1.])
def dynamics(x,u,param=PARAM):
 q=x[3:7];v=x[7:10];w=x[10:13]
 err=u[1:]-w
 nominal_cross=cross(w,JC*w)
 denom=(nominal_cross.square().sum().sqrt()/100)
 denom=I(np.maximum(denom.lo,1),np.maximum(denom.hi,1))
 torque=JC*(30*err)+nominal_cross/denom
 wdot=(torque-cross(w,param['j']*w))/param['j']
 # Preserve actual divide/multiply path rather than canceling uncertain m,g.
 thrust=u[0]*param['g']*param['m']
 fdrag=rotate(q,param['drag']*rotate(inv(q),v))
 acc=rotate(q,UP)*(thrust/param['m'])-param['g']*UP-fdrag/param['m']
 qdot=.5*mul(q,cat([w,iv([0.])]))
 return cat([v,qdot,acc,wdot])
def normalize(q,unit_shell):
 norm=q.square().sum().sqrt();norm=I(np.maximum(norm.lo,1e-12),np.maximum(norm.hi,1e-12))
 q=q/norm
 if unit_shell:
  # Conditional fixed-DAG IEEE32 normalization error enclosure, deliberately
  # loose. Cannot transfer to an unverified target vector_norm/reassociation.
  e=64*U if ROUND else 1e-12
  q=I(np.maximum(q.lo,-(1+e)),np.minimum(q.hi,1+e))
  # Unit-sphere constraint projection on positive scalar chart.
  if q.lo[3]>0:
   s=q[:3].square().sum()
   lo=math.sqrt(max(0,(1-e)**2-float(s.hi)));hi=math.sqrt(max(0,(1+e)**2-max(0,float(s.lo))))
   q.lo[3]=max(q.lo[3],lo);q.hi[3]=min(q.hi[3],hi)
 return q

def substep(x,t,u,h,unit_shell,param=PARAM):
 command=u[0].clamp(0,float(param['tmax'].hi))
 command=I(np.minimum(command.lo,param['tmax'].lo),np.minimum(command.hi,param['tmax'].hi))
 decay=(-iv(h)/param['lag']).exp()
 # Exact real multilinear corner extrema preserve repeated-command
 # dependence. Add a GLOBAL bound for all three FP32 arithmetic roundings;
 # contraction to FMA has no larger bound under the fixed-DAG assumptions.
 cs=[float(command.lo),float(command.hi)];ts=[float(t.lo),float(t.hi)];ds=[float(decay.lo),float(decay.hi)]
 values=[c+(tt-c)*d for c in cs for tt in ts for d in ds]
 cmax=max(abs(v) for v in cs);dmax=max(abs(tt-c) for tt in ts for c in cs)
 pad=8*U*(cmax+dmax*(1+max(abs(v) for v in ds)))+8*TINY if ROUND else 16*np.finfo(float).eps*(cmax+dmax)
 t=I(np.nextafter(min(values)-pad,-np.inf),np.nextafter(max(values)+pad,np.inf))
 upstream=cat([stack([t/(param['m']*param['g'])]),u[1:].clamp(-6,6)])
 k1=h*dynamics(x,upstream,param);k2=h*dynamics(x+.5*k1,upstream,param)
 k3=h*dynamics(x+.5*k2,upstream,param);k4=h*dynamics(x+k3,upstream,param)
 new=x+(k1+2*k2+2*k3+k4)/6
 new=cat([new[:3],normalize(new[3:7],unit_shell),new[7:]])
 # Actual simulator semantics defines a numerical state transition, not a
 # unique continuous trajectory. Cover the discrete p endpoints and all RK4
 # stage position boxes, using axis-aligned hull; not an ODE flow certificate.
 stage=[x[:3],(x+.5*k1)[:3],(x+.5*k2)[:3],(x+k3)[:3],new[:3]]
 sweep=I(np.min([v.lo for v in stage],axis=0),np.max([v.hi for v in stage],axis=0))
 return new,t,sweep

def feedback(x,ep,ev,eq):
 obs_p=x[:3]+ranges([-ep]*3,[ep]*3);obs_v=x[7:10]+ranges([-ev]*3,[ev]*3)
 q=x[3:7]+ranges([-eq]*4,[eq]*4)
 a=-1.5*(obs_p-iv([6.,6.,6.]))-2.5*obs_v
 force=a+iv(9.80665)*UP
 norm=force.square().sum().sqrt();norm=I(np.maximum(norm.lo,1e-6),np.maximum(norm.hi,1e-6));fz=force/norm
 bz=crotate(q,UP);tilt=6*crotate(inv(q),cross(bz,fz))
 heading=crotate(q,iv([1.,0.,0.]))
 yaw=(-heading[1]/heading[0]).atan() if heading.lo[0]>0 else I(-math.pi,math.pi)
 yawrate=(6*yaw).clamp(-1,1)
 rates=(tilt+crotate(inv(q),UP*yawrate)).clamp(-6,6)
 thrust=(.5*(force*bz).sum()).clamp(0,24.5)
 return cat([stack([thrust]),rates])

def initial(ep,ev,eq,ew,velocity_center=(0.,0.,0.),quaternion_xyz_center=(0.,0.,0.)):
 qc=np.asarray(quaternion_xyz_center);qxyz=ranges(qc-eq,qc+eq)
 upper=np.maximum(qxyz.lo*qxyz.lo,qxyz.hi*qxyz.hi).sum()
 lower=np.where((qxyz.lo<=0)&(qxyz.hi>=0),0,np.minimum(qxyz.lo*qxyz.lo,qxyz.hi*qxyz.hi)).sum()
 e=64*U if ROUND else 1e-12
 qw=I(math.sqrt(max(0,(1-e)**2-upper)),math.sqrt(max(0,(1+e)**2-lower)))
 vc=np.asarray(velocity_center)
 return cat([ranges(np.array([6.,6.,6.])-ep,np.array([6.,6.,6.])+ep),qxyz,stack([qw]),ranges(vc-ev,vc+ev),ranges([-ew]*3,[ew]*3)])

def run(label,uncertainty,mode,unit_shell=True,duration=.5,rounding=True,velocity_center=(0.,0.,0.),quaternion_xyz_center=(0.,0.,0.)):
 global ROUND;ROUND=rounding
 ep,ev,eq,ew=uncertainty;x=initial(ep,ev,eq,ew,velocity_center,quaternion_xyz_center);t=ranges(4.401,5.4065)
 dt=.01;h=.002;queue=[iv([4.903325,0.,0.,0.]) for _ in range(3)]
 rows=[];states=[];start=time.perf_counter();status='completed'
 try:
  for step in range(round(duration/dt)):
   command=iv([4.903325,0.,0.,0.]) if mode=='fixed_hover' else feedback(x,ep,ev,eq)
   queue.insert(0,command);del queue[3:]
   # Unknown constant transport delay in[.005,.02]s: convex hull all potentially
   # interpolated known queue slots encloses causal fractional-delay commands.
   u=I(np.min([v.lo for v in queue],axis=0),np.max([v.hi for v in queue],axis=0))
   sweeps=[]
   for _ in range(5):
    x,t,sweep=substep(x,t,u,h,unit_shell);sweeps.append(sweep)
    if max(x.radius[3:7])>.4 or max(x.radius[10:13])>10:
     status='quaternion_or_rate_wrapping_limit';break
   if status!='completed':break
   radius=float(np.linalg.norm(np.maximum(abs(x.lo[:3]-6),abs(x.hi[:3]-6))))
   sweep_box=I(np.min([v.lo for v in sweeps],axis=0),np.max([v.hi for v in sweeps],axis=0))
   sweep_radius=float(np.linalg.norm(np.maximum(abs(sweep_box.lo-6),abs(sweep_box.hi-6))))
   row={'time_s':(step+1)*dt,'position_radius_m':radius,'position_halfwidths_m':x.radius[:3].tolist(),'velocity_halfwidths_mps':x.radius[7:10].tolist(),'quaternion_halfwidths':x.radius[3:7].tolist(),'bodyrate_halfwidths_radps':x.radius[10:13].tolist(),'motor_thrust_interval_N':t.out(),'input':u.out(),'room_ball2m_fits_body':radius+.17<2,'swept_position_box':sweep_box.out(),'swept_center_radius_m':sweep_radius,'swept_body_fits_ball2m':sweep_radius+.17<2}
   rows.append(row);states.append({'state':x.out(),'motor':t.out()})
   if not np.isfinite(x.lo).all() or radius>2 or max(x.radius[10:13])>10:
    status='wrapping_limit';break
 except(ArithmeticError,FloatingPointError,OverflowError) as error:status=str(error)
 return {'label':label,'mode':mode,'unit_shell':unit_shell,'velocity_center':list(velocity_center),'quaternion_xyz_center':list(quaternion_xyz_center),'rounding_pad':rounding,'uncertainty_component_bounds':{'p_m':ep,'v_mps':ev,'q_xyz':eq,'w_radps':ew},'status':status,'elapsed_s':time.perf_counter()-start,'steps':rows,'states':states,'last':rows[-1] if rows else None,'final_state':x.out()}

if __name__=='__main__':
 result={'scope':'Conditional fixed-expression IEEE32 interval feasibility; NOT a compiled CUDA numerical proof or safety contract. No ARES package edits.','float_contract':{'host_endpoint_rounding':'binary64 nextafter after each primitive/reduction','target_primitives':'2u32 outward each arithmetic primitive; 4u32 sqrt;32u32 exp/atan placeholders; reduction gamma_n sumabs','unverified':['exp/atan error envelopes for actual eager/compiled target','vector_norm reduction/sqrt/rsqrt implementation','fast-math reassociation beyond preserved expression DAG','TF32/reduced precision matmul codegen','normalization unit-shell bound for actual compiler'],'fused_math':'Fixed-DAG FMA contraction enclosed by exact subexpression intervals and conservative separate rounding; arbitrary reassociation not certified.'},'parameter_sets':{k:v.out() for k,v in PARAM.items()},'runs':[]}
 for uncertainty,label in [((.005,.01,.001,.01),'small_public_set'),((.03,.05,.01,.05),'larger_public_set')]:
  for mode in ['fixed_hover','pd_backup']:
   for shell in [False,True]:
    row=run(label,uncertainty,mode,shell)
    result['runs'].append(row);print(label,mode,shell,row['status'],row['last'],flush=True)
 for mode in ['fixed_hover','pd_backup']:
  result['runs'].append(run('small_public_set_no_rounding',(.005,.01,.001,.01),mode,True,rounding=False))
 result['runs'].append(run('small_public_set_moving_tilted',(.005,.01,.001,.01),'pd_backup',True,velocity_center=(.1,.1,.05),quaternion_xyz_center=(math.sin(.005),-math.sin(.005),0.)))
 result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 def finite(v):
  if isinstance(v,dict):return {k:finite(x) for k,x in v.items()}
  if isinstance(v,list):return [finite(x) for x in v]
  if isinstance(v,float) and not math.isfinite(v):return str(v)
  return v
 Path(__file__).with_name('feasibility.json').write_text(json.dumps(finite(result),indent=2,allow_nan=False)+'\n')
