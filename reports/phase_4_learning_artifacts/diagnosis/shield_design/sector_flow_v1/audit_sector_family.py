"""Finite actual-vendor membership + sector-family derivative coding audit."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent;BASE=ROOT/'reports/phase_4_gate_artifacts/shield_research/taylor_remainder_v1';sys.path.insert(0,str(BASE));sys.path.insert(0,str(OUT))
import numpy as np,torch,json,math,time,hashlib
from sector_flow import flow_map,GCENTER,GDELTA
from ares.vendor.diffaero.quadrotor import QuadrotorModel as OriginalPlant
from ares.vendor.diffaero.math import quat_rotate,quat_inv
actual={"__file__":str(BASE/'audit_increment_flow.py')};exec((BASE/'audit_increment_flow.py').read_text().split('rng=')[0],actual);torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
report=json.loads((OUT/'sector_quantized_flow_bounds.json').read_text());r=report['state_radii'];ur=report['command_deviation_radii_binary64_outward'];rng=np.random.default_rng(19903);rows=[];start=time.monotonic()
def excess(x,lo,hi):return float(np.maximum(np.maximum(lo-x,x-hi),0).max())
for case in range(6):
 branch=case%2;p={'m':float(np.float32(.45 if case<3 else .55)),'g':float(np.float32(9.78 if case%2==0 else 9.83)),'drag':float(np.float32(.05 if case%2==0 else .2)),'tau':float(np.float32(.015 if case%2==0 else .04)),'j':[float(np.float32(v)) for v in ([.0018,.0022,.0027] if case%2==0 else [.0022,.0018,.0033])],'jn':[float(np.float32(x)) for x in [.002,.002,.003]],'battery':.95 if case<3 else 1.,'delay_fraction':.75 if branch==0 else .5};z=rng.uniform(-1,1,21)*r;u=rng.uniform(-1,1,4)*ur;x=torch.tensor(np.r_[z,u,0.,0.],requires_grad=True);a,b=(u,z[13:17]) if branch==0 else (z[13:17],z[17:21]);delayed=(1-p['delay_fraction'])*a+p['delay_fraction']*b;motor=z[12];alpha=math.exp(-.002/p['tau']);geff=[]
 for j in range(5):motor=alpha*motor+(1-alpha)*p['battery']*delayed[0];geff.append(p['g']+motor/p['m'])
 assert all(abs(g-float(GCENTER))<float(GDELTA) for g in geff)
 actual['QuadrotorModel']=OriginalPlant;yactual=actual['model'](x,p,branch).detach().numpy()
 class SectorPlant(OriginalPlant):
  def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.index=0
  def step(self,state,action,dt):
   self.geff=geff[self.index];self.index+=1
   return super().step(state,action,dt)
  def dynamics(self,state,action):
   original=super().dynamics(state,action);q=state[:,3:7];v=state[:,7:10];up=torch.tensor([[0.,0.,1.]]);bz=quat_rotate(q,up);drag=quat_rotate(q,(self._D@quat_rotate(quat_inv(q),v).unsqueeze(-1)).squeeze(-1));delta=action[:,0]*self.gravity*self.mass-self.gravity*self.mass if hasattr(self,'gravity') else (action[:,0]-1)*(-self._G_vec[:,2])*self.mass
   acc=self.geff*(bz-up)+(delta/self.mass)[:,None]*up-drag/self.mass[:,None]
   return torch.cat((original[:,:7],acc,original[:,10:]),-1)
 actual['QuadrotorModel']=SectorPlant
 def model(xx):return actual['model'](xx,p,branch)
 y=model(x).detach().numpy();psector={**p,'geff':geff};point,_,_=flow_map(r,ur,branch,point=z,command_point=u,parameter_values=psector);J=torch.autograd.functional.jacobian(model,x,vectorize=True).detach().numpy();jex=excess(J,np.array([v.g.lo for v in point]),np.array([v.g.hi for v in point]));hexcess=0.
 for output in [3,6,9,12,13]:
  H=torch.autograd.functional.hessian(lambda xx:model(xx)[output],x,vectorize=True).detach().numpy();hexcess=max(hexcess,excess(H,point[output].h.lo,point[output].h.hi))
 zero=torch.zeros(27,requires_grad=True);origin=model(zero).detach().numpy();linear=torch.autograd.functional.jacobian(model,zero,vectorize=True).detach().numpy();rem=y-origin-linear@x.detach().numpy();remex=float(np.maximum(abs(rem)-np.array(report['runs'][branch]['epsilon_joint_state_command']),0).max());row={'case':case,'branch':branch,'stage_geff':geff,'actual_vendor_value_vs_containing_sector_family_max_error':float(abs(yactual-y).max()),'sector_point_interval_value_excess':excess(y,np.array([v.v.lo for v in point]),np.array([v.v.hi for v in point])),'sector_point_J_excess':jex,'sector_point_selected_H_excess':hexcess,'sector_global_box_remainder_excess':remex};rows.append(row);print(row,flush=True)
assert max(v for r in rows for k,v in r.items() if k.endswith('error') or k.endswith('excess'))<1e-9
result={'status':'FINITE_CODING_AUDIT_ONLY','actual_membership_value_checked':True,'sector_derivatives_hold_stage_geff_constant':'This family derivative intentionally differs from the actual state-dependent plant Jacobian; the proof is set containment, not derivative equality.','cases':rows,'all_six_within_1e_minus9':True,'elapsed_s':time.monotonic()-start};(OUT/'sector_family_coding_audit.json').write_text(json.dumps(result,indent=2)+'\n')
