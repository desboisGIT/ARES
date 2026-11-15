"""Finite actual-vendor sanity audit of independent-command real flow bounds."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
import numpy as np,torch,json,math,time,hashlib
from ares.vendor.diffaero.quadrotor import QuadrotorModel
from interval_flow_increment import flow_map

torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
report=json.loads((OUT/'increment_inflated_flow_bounds.json').read_text());r=report['state_radii'];ur=report['command_deviation_radii_binary64_outward']
Jn=[float(np.float32(x)) for x in [.002,.002,.003]]
def model(x,p,branch):
 z=x[:21];u=x[21:25];m=p['m'];g=p['g'];b=p['battery'];mg=m*g;teq=mg/b
 eq=(9.80665-2*teq)/1.5;anchor=torch.tensor([0.,0.,eq]);position=z[:3]+anchor
 xyz=z[6:9];q=torch.cat((xyz,(1-xyz.square().sum()).sqrt().reshape(1)));motor=z[12]+mg
 state=torch.cat((position,q,z[3:6],z[9:12]))[None];eqcommand=torch.tensor([teq,0.,0.,0.])
 new=u+eqcommand;one=z[13:17]+eqcommand;two=z[17:21]+eqcommand
 a,bb=(new,one) if branch==0 else (one,two);f=p['delay_fraction'];applied=(1-f)*a+f*bb
 plant=QuadrotorModel(torch.tensor([m]),torch.tensor([p['j']]),torch.full((1,3),p['drag']),torch.tensor([g]),torch.full((1,3),30.),torch.tensor([p['jn']]))
 decay=math.exp(-.002/p['tau'])
 for _ in range(5):
  motor=b*applied[0]+(motor-b*applied[0])*decay
  action=torch.cat(((motor/mg).reshape(1),applied[1:]))[None]
  state=plant.step(state,action,.002)
 return torch.cat((state[0,:3]-anchor,state[0,7:10],state[0,3:6],state[0,10:13],(motor-mg).reshape(1),u,z[13:17]))
def excess(a,lo,hi):return float(np.maximum(np.maximum(lo-a,a-hi),0).max())
rng=np.random.default_rng(81012);records=[];start=time.monotonic()
for case in range(6):
 branch=case%2
 p={'m':float(np.float32(.45 if case<3 else .55)),'g':float(np.float32(9.78 if case%2==0 else 9.83)),
    'drag':float(np.float32(.05 if case%2==0 else .2)),'tau':float(np.float32(.015 if case%2==0 else .04)),
    'j':[float(np.float32(v)) for v in ([.0018,.0022,.0027] if case%2==0 else [.0022,.0018,.0033])],
    'jn':Jn,'battery':.95 if case<3 else 1.,'delay_fraction':.75 if branch==0 else .5}
 z=rng.uniform(-1,1,21)*r;u=rng.uniform(-1,1,4)*ur
 x=torch.tensor(np.r_[z,u,0.,0.],requires_grad=True)
 point,_,_=flow_map(r,ur,branch,point=z,command_point=u,parameter_values=p)
 y=model(x,p,branch).detach().numpy();J=torch.autograd.functional.jacobian(lambda xx:model(xx,p,branch),x,vectorize=True).detach().numpy()
 vexc=excess(y,np.array([a.v.lo for a in point]),np.array([a.v.hi for a in point]));jexc=excess(J,np.array([a.g.lo for a in point]),np.array([a.g.hi for a in point]))
 hexcess=0.
 for k in [3,6,9,12,13]:
  H=torch.autograd.functional.hessian(lambda xx:model(xx,p,branch)[k],x,vectorize=True).detach().numpy()
  hexcess=max(hexcess,excess(H,point[k].h.lo,point[k].h.hi))
 zero=torch.zeros(27,requires_grad=True);origin=model(zero,p,branch).detach().numpy()
 local=torch.autograd.functional.jacobian(lambda xx:model(xx,p,branch),zero,vectorize=True).detach().numpy()
 A=local[:,:21];Gamma=local[:,21:25]
 bounds=report['runs'][branch];gexc=excess(Gamma,np.asarray(bounds['local_Gamma_parameter_interval_lower']),np.asarray(bounds['local_Gamma_parameter_interval_upper']))
 rem=y-origin-A@z-Gamma@u
 rexc=float(np.maximum(abs(rem)-np.asarray(bounds['epsilon_joint_state_command']),0).max())
 row={'case':case,'branch':branch,'parameters':p,'point_value_excess':vexc,'point_J_excess':jexc,'point_selected_H_excess':hexcess,
      'local_Gamma_global_interval_excess':gexc,'joint_remainder_excess':rexc,'joint_remainder_abs_max':float(abs(rem).max()),
      'motor_queue_H_exact_zero_in_prototype':all(np.all(v.h.lo==0) and np.all(v.h.hi==0) for v in point[12:])}
 records.append(row);print(row,flush=True)
assert max(x[k] for x in records for k in ['point_value_excess','point_J_excess','point_selected_H_excess','local_Gamma_global_interval_excess','joint_remainder_excess'])<1e-9
assert all(x['motor_queue_H_exact_zero_in_prototype'] for x in records)
result={'scope':'Finite float64 vendor implementation audit; tolerance1e-9 for different floating algebra, not a proof of state-set or machine error enclosure.',
        'all_six_cases_within_tolerance':True,'cases':records,'elapsed_s':time.monotonic()-start,
        'source_sha256':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['interval_flow_increment.py','interval_taylor.py','increment_inflated_flow_bounds.json']}}
(OUT/'increment_inflated_flow_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
