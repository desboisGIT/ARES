"""Finite implementation audit, explicitly not a nonlinear set-containment proof."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import interval_taylor as t
import torch,numpy as np,json,math,time,hashlib
from ares.flight_control import ControllerConfig,acceleration_to_action
from ares.vendor.diffaero.quadrotor import QuadrotorModel

torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
OUT=Path(__file__).resolve().parent
report=json.loads((OUT/'taylor_bounds.json').read_text());radius=np.asarray(report['state_component_radii_binary64_outward'])
Jn=[float(np.float32(x)) for x in [.002,.002,.003]]

def torch_map(x,p,branch):
 z=x[:21];noise=x[21:];m=p['m'];g=p['g'];b=p['battery'];mg=m*g;teq=mg/b
 eq=(9.80665-2*teq)/1.5
 anchor=torch.tensor([0.,0.,eq]);position=z[:3]+anchor;velocity=z[3:6]
 xyz=z[6:9];q=torch.cat((xyz,(1-xyz.square().sum()).sqrt().reshape(1)))
 w=z[9:12];motor=z[12]+mg
 measured=position+noise[:3];measured_v=velocity+noise[3:]
 obs={'p':measured[None],'v':measured_v[None],'q':q[None],'gaze':measured[None]+torch.tensor([[1.,0.,0.]])}
 desired=-1.5*measured-2.5*measured_v
 action=acceleration_to_action(obs,desired[None],ControllerConfig())[0]
 eqaction=torch.tensor([teq,0.,0.,0.]);q1=z[13:17]+eqaction;q2=z[17:21]+eqaction
 a,bb=(action,q1) if branch==0 else (q1,q2);f=p['delay_fraction']
 delayed=(1-f)*a+f*bb
 state=torch.cat((position,q,velocity,w))[None]
 model=QuadrotorModel(torch.tensor([m]),torch.tensor([p['j']]),torch.full((1,3),p['drag']),torch.tensor([g]),torch.full((1,3),30.),torch.tensor([p['jn']]))
 decay=math.exp(-.002/p['tau'])
 for _ in range(5):
  motor=b*delayed[0]+(motor-b*delayed[0])*decay
  u=torch.cat(((motor/mg).reshape(1),delayed[1:]))[None]
  state=model.step(state,u,.002)
 return torch.cat((state[0,:3]-anchor,state[0,7:10],state[0,3:6],state[0,10:13],(motor-mg).reshape(1),action-eqaction,z[13:17]))

def excess(a,lo,hi):return float(np.maximum(np.maximum(lo-a,a-hi),0).max())
rng=np.random.default_rng(81011);rows=[];start=time.monotonic()
for case in range(6):
 branch=case%2
 p={'m':float(np.float32(.45 if case<2 else .55 if case<4 else .5)),
    'g':float(np.float32(9.78 if case%2==0 else 9.83)),
    'drag':float(np.float32(.05 if case%2==0 else .2)),
    'tau':float(np.float32(.015 if case%2==0 else .04)),
    'j':list(map(lambda x:float(np.float32(x)),[.0018,.0022,.0027] if case%2==0 else [.0022,.0018,.0033])),
    'jn':Jn,'battery':.95 if case<3 else 1.,'delay_fraction':.75 if branch==0 else .5}
 point=rng.uniform(-1,1,21)*radius;noise=rng.uniform(-1,1,6)*np.array([.005]*3+[.01]*3)
 x=torch.tensor(np.r_[point,noise],requires_grad=True)
 y=torch_map(x,p,branch).detach().numpy()
 jac=torch.autograd.functional.jacobian(lambda xx:torch_map(xx,p,branch),x,vectorize=True).detach().numpy()
 output,_,_=t.backup_map(radius,.005,.01,branch,point=point,noise_point=noise,parameter_values=p)
 vlow=np.array([o.v.lo for o in output]);vhigh=np.array([o.v.hi for o in output]);jlow=np.array([o.g.lo for o in output]);jhigh=np.array([o.g.hi for o in output])
 selected=[3,9,13,14];hessian_excess=0.;global_excess=0.
 big=[r for r in report['runs'] if r['label']=='declared_service_bounds' and r['branch']==branch][0]
 globalH=np.asarray(big['hessian_absolute_upper'])
 for index in selected:
  h=torch.autograd.functional.hessian(lambda xx:torch_map(xx,p,branch)[index],x,vectorize=True).detach().numpy()
  hessian_excess=max(hessian_excess,excess(h,output[index].h.lo,output[index].h.hi))
  global_excess=max(global_excess,float(np.maximum(abs(h)-globalH[index],0).max()))
 zero=torch.zeros(27,requires_grad=True)
 origin=torch_map(zero,p,branch).detach().numpy()
 A=torch.autograd.functional.jacobian(lambda xx:torch_map(xx,p,branch),zero,vectorize=True).detach().numpy()[:,:21]
 remainder=y-origin-A@point
 remainder_bound=np.asarray(big['epsilon_state_nonlinear'])+np.asarray(big['epsilon_measurement_noise'])
 remainder_excess=float(np.maximum(abs(remainder)-remainder_bound,0).max())
 row={'equilibrium_float64_residual_max':float(abs(origin).max()),
      'nonlinear_plus_noise_remainder_max':float(abs(remainder).max()),
      'remainder_excess_over_interval_bound':remainder_excess,'case':case,'branch':branch,'parameters':p,'point_value_float64_excess':excess(y,vlow,vhigh),
      'point_jacobian_float64_excess':excess(jac,jlow,jhigh),
      'point_selected_hessian_float64_excess':hessian_excess,
      'full_domain_selected_hessian_float64_excess':global_excess,'hessian_outputs_checked':selected}
 rows.append(row);print(row,flush=True)
assert max(r[k] for r in rows for k in ('point_value_float64_excess','point_jacobian_float64_excess','point_selected_hessian_float64_excess','full_domain_selected_hessian_float64_excess','remainder_excess_over_interval_bound'))<1e-9
sources=['flight_control.py','vendor/diffaero/quadrotor.py','vendor/diffaero/controller.py','vendor/diffaero/math.py']
result={'scope':'Finite float64 autograd audit against existing vendor/backup implementation; tolerance1e-9 for different floating evaluation graphs, NOT a proof of all-state/parameter/rounding containment.',
        'all_six_audits_within_tolerance':True,'cases':rows,'elapsed_s':time.monotonic()-start,
        'prototype_source_sha256':hashlib.sha256((OUT/'interval_taylor.py').read_bytes()).hexdigest(),
        'runtime_source_sha256':{name:hashlib.sha256((ROOT/'ares'/name).read_bytes()).hexdigest() for name in sources}}
(OUT/'autograd_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
