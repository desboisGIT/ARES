"""Joint state/new-command REAL RK4 Hessian; controller kept outside the flow.

Input slots0..20 are centered state,21..24 public SI new command deviations;
25..26 unused. Motor/queue are affine for each fixed admissible parameter.
No floating-execution error budget or nonlinear invariance claim.
"""
from fractions import Fraction as F
import math,numpy as np
from interval_taylor import (I,D,DomainFailure,params,frac,exp_endpoint,strict_domain,
                            qchart,ad,dot,cross,rotate,qmul,scale,add,summation,sqrt_rational)

def flow_map(state_radius,command_radius,branch,point=None,command_point=None,parameter_values=None):
 p=params()
 if parameter_values is not None:
  for key,value in parameter_values.items():
   if key!='delay_fraction':p[key]=[I(x) for x in value] if isinstance(value,list) else I(value)
  arg=-I.rational(F(1,500))/p['tau']
  p['alpha']=I(exp_endpoint(float(arg.lo),False),exp_endpoint(float(arg.hi),True))
 z=[D.variable(I(-r,r) if point is None else I(point[i]),i) for i,r in enumerate(state_radius)]
 action=[D.variable(I(-r,r) if command_point is None else I(command_point[i]),21+i) for i,r in enumerate(command_radius)]
 checks={};mg=p['m']*p['g'];teq=mg/p['battery']
 strict_domain((action[0]+teq).v,0,22.,'new_thrust_policy_and_plant_unclamped',checks)
 for i in range(1,4):strict_domain(action[i].v,-6,6,f'new_body_rate_{i}_unclamped',checks)
 for slot in (13,17):
  strict_domain((z[slot]+teq).v,0,22.,f'queue_thrust_{slot}_unclamped',checks)
  for i in range(1,4):strict_domain(z[slot+i].v,-6,6,f'queue_rate_{slot}_{i}_unclamped',checks)
 strict_domain((mg+z[12]).v,0,22.,'initial_motor_positive_bounded',checks)
 q=qchart(z[6:9]);strict_domain(q[3].v,0,math.inf,'positive_initial_qw',checks)
 a,b=(action,z[13:17]) if branch==0 else (z[13:17],z[17:21])
 fraction=I.rational(F(1,2),1) if branch==0 else I.rational(0,1)
 if parameter_values is not None and 'delay_fraction' in parameter_values:fraction=I(parameter_values['delay_fraction'])
 delayed=[(1-fraction)*x+fraction*y for x,y in zip(a,b)]
 u=delayed[1:];motor=z[12];physical=z[:3]+q+z[3:6]+z[9:12];h=F(1,500);maxgyro=0.
 def dynamics(x,motor):
  nonlocal maxgyro
  qq=x[3:7];v=x[7:10];w=x[10:13]
  nominal_cross=cross(w,[p['jn'][i]*w[i] for i in range(3)])
  cross2=sum((frac(float(x.v.absmax()))**2 for x in nominal_cross),F(0))
  if cross2>=10000:raise DomainFailure('nominal compensation saturation not excluded')
  maxgyro=max(maxgyro,sqrt_rational(cross2,True))
  mismatch=cross(w,[(p['jn'][i]-p['j'][i])*w[i] for i in range(3)])
  wdot=[30*p['jn'][i]/p['j'][i]*(u[i]-w[i])+mismatch[i]/p['j'][i] for i in range(3)]
  bz=rotate(qq,[ad(0),ad(0),ad(1)])
  norm4=summation([q.square() for q in qq]).square()
  acc=[p['g']*(bz[i]-(1 if i==2 else 0))+(motor/p['m'])*bz[i]-(p['drag']/p['m'])*norm4*v[i] for i in range(3)]
  return v+scale(qmul(qq,w+[ad(0)]),F(1,2))+acc+wdot
 for step in range(5):
  motor=p['alpha']*motor+(1-p['alpha'])*p['battery']*delayed[0]
  # This recurrence is exactly affine in state/new-command for fixed theta;
  # parameter intervals do not introduce state Hessians. Remove only roundoff
  # artifacts of the interval evaluator's zero operations, by this identity.
  motor=D(motor.v,motor.g,I(np.zeros((27,27))))
  k1=scale(dynamics(physical,motor),h);k2=scale(dynamics(add(physical,scale(k1,F(1,2))),motor),h)
  k3=scale(dynamics(add(physical,scale(k2,F(1,2))),motor),h);k4=scale(dynamics(add(physical,k3),motor),h)
  nxt=add(physical,scale(add(add(k1,scale(k2,2)),add(scale(k3,2),k4)),F(1,6)))
  raw=nxt[3:7];strict_domain(raw[3].v,0,math.inf,f'positive_raw_qw_{step}',checks)
  norm=summation([x.square() for x in raw]).sqrt();strict_domain(norm.v,1e-12,math.inf,f'normalization_norm_{step}',checks)
  xyz=scale(raw[:3],norm.reciprocal());physical=nxt[:3]+qchart(xyz)+nxt[7:]
 checks['max_nominal_compensation_norm_upper']=maxgyro
 result=physical[:3]+physical[7:10]+physical[3:6]+physical[10:13]+[motor]+action+z[13:17]
 return result,checks,p
