"""Explicit norm3/4 + verified sqrt, original eager CPU13D/RK4 equations.

Research numerical backend only. Original package51 and old backend unchanged.
No opaque torch.norm/linalg/vector_norm or library sqrt call is used in steps.
"""
from pathlib import Path
import sys,hashlib,struct,importlib.util,types,ctypes,json
from fractions import Fraction as F
import torch
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(BASE/'roundoff_enclosure_v1'))
from verified_sqrt import sqrt_word,bits
from conditional_cpu_backend import ConditionalCPUBackend,GuardFailure,rational

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class NormWitness:
 def __init__(self):self.sqrt_points=0;self.norm3_calls=0;self.norm4_calls=0;self.digest=hashlib.sha256()
 def snapshot(self):return dict(sqrt_points=self.sqrt_points,norm3_calls=self.norm3_calls,norm4_calls=self.norm4_calls,verified_input_output_words_sha256=self.digest.hexdigest(),every_sqrt_point_exact_bin_verified=True)

def verified_sqrt_tensor(value,audit):
 if type(value) is not torch.Tensor or value.device.type!='cpu' or value.dtype!=torch.float32 or value.requires_grad or not bool(torch.isfinite(value).all()) or bool((value<0).any()):raise ValueError('plain finite nonnegative CPU Float32 sqrt words required')
 result=[]
 for x in value.reshape(-1).tolist():
  y,witness=sqrt_word(x)
  if not witness['exact_squared_bin_pass']:raise ArithmeticError('sqrt witness failed')
  audit.sqrt_points+=1;audit.digest.update(struct.pack('>II',bits(x),bits(y)));result.append(y)
 return torch.tensor(result,dtype=torch.float32,device='cpu').reshape(value.shape)

def explicit_norm(value,width,audit):
 if type(value) is not torch.Tensor or value.device.type!='cpu' or value.dtype!=torch.float32 or value.requires_grad or value.shape[-1]!=width or width not in (3,4) or not bool(torch.isfinite(value).all()):raise ValueError('plain finite CPU Float32 norm3/4 words required')
 # Separate eager product and addition operators; N products and N-1 sums.
 total=value[...,0]*value[...,0]
 for j in range(1,width):
  product=value[...,j]*value[...,j];total=total+product
 if width==3:audit.norm3_calls+=1
 else:audit.norm4_calls+=1
 return verified_sqrt_tensor(total,audit).unsqueeze(-1)

def make_adapter_class():
 from ares.vendor.diffaero.controller import RateController
 from ares.vendor.diffaero.math import rk4
 VerifiedDecay=load('verified_norm_decay_parent',BASE/'verified_decay_v1.py').make_adapter_class()
 class VerifiedNormRateController(RateController):
  def __init__(self,*args,norm_witness,**kwargs):super().__init__(*args,**kwargs);self.norm_witness=norm_witness
  def __call__(self,q_xyzw,w,action):
   angvel_err=action[:,1:]-w
   cross=torch.cross(w,(self.inertia@w.unsqueeze(-1)).squeeze(-1),dim=1)
   cross=cross/torch.maximum(explicit_norm(cross,3,self.norm_witness)/100,torch.ones_like(cross[:,:1])).detach()
   angacc=self.K_angvel*angvel_err
   torque=(self.inertia@angacc.unsqueeze(-1)).squeeze(-1)+cross
   thrust=action[:,0]*self.gravity*self.mass
   return thrust,torque
 def verified_model_step(model,state,action,dt):
  new=rk4(model.dynamics,state,action,dt=dt,M=1)
  quat=new[...,3:7];quat=quat/explicit_norm(quat,4,model.norm_witness).clamp_min(1e-12)
  return torch.cat((new[...,:3],quat,new[...,7:]),-1)
 class VerifiedNormCPUAdapter(VerifiedDecay):
  def __init__(self,*args,**kwargs):
   self.norm_witness=NormWitness();super().__init__(*args,**kwargs)
   old=self.model.controller
   self.model.controller=VerifiedNormRateController(old.mass,old.inertia,old.gravity,old.K_angvel,norm_witness=self.norm_witness)
   self.model.norm_witness=self.norm_witness;self.model.step=types.MethodType(verified_model_step,self.model)
   self.verified_model_step_function=verified_model_step;self.verified_controller_class=VerifiedNormRateController
  def reset(self,position=None,velocity=None,quaternion=None,angular_velocity=None):
   zero=torch.zeros(self.batch_size,3,device='cpu',dtype=torch.float32)
   p=zero if position is None else self._tensor(position,3);v=zero if velocity is None else self._tensor(velocity,3);w=zero if angular_velocity is None else self._tensor(angular_velocity,3)
   quat=self._tensor([0.,0.,0.,1.] if quaternion is None else quaternion,4);norm=explicit_norm(quat,4,self.norm_witness)
   if bool((norm<1e-8).any()):raise ValueError('quaternion must have nonzero norm')
   self.state=torch.cat((p,quat/norm,v,w),-1);self.applied_thrust=self.mass*self.gravity;self.last_action=torch.cat((self.applied_thrust[:,None],zero),-1);return self.state
 return VerifiedNormCPUAdapter

class ConditionalVerifiedNormCPUBackend(ConditionalCPUBackend):
 """New guarded variant; old conditional wrapper is retained unchanged."""
 def __init__(self,profile,*,batch_size=1,delay_s=.0125):
  # Instantiate only the explicit-norm adapter: no discarded native-norm
  # initialization is needed in this new variant. The historical parent stays.
  self.guard_module=load('verified_norm_cpu_guard',BASE/'cpu_arithmetic_guard_v1.py')
  self.precondition_report=self.guard_module.probe();torch.backends.mkldnn.enabled=False
  if not self.precondition_report['cpu_preconditions_met']:raise GuardFailure('CPU arithmetic preconditions failed')
  frozen=BASE.parent/'frozen_flight_runtime'/'source_manifest.json'
  self.frozen_source_manifest=json.loads(frozen.read_text());package=HERE.parents[3]/'ares'
  if any(not (package/n).is_file() or hashlib.sha256((package/n).read_bytes()).hexdigest()!=h for n,h in self.frozen_source_manifest.items()):raise GuardFailure('frozen original package source mismatch')
  self.helper=ctypes.CDLL(self.precondition_report['helper']['library_path'])
  self.helper.guard_read_mxcsr.restype=ctypes.c_uint;self.helper.guard_read_round.restype=ctypes.c_int;self.helper.guard_nearest_constant.restype=ctypes.c_int
  Verified=make_adapter_class();self.plant=Verified(profile,batch_size=batch_size,device='cpu',dt=.01,integration_dt=.002,compile_dynamics=False);batch_size=self.plant.batch_size
  self._step_implementation=self.plant.step.__func__;self._norm_ready=True;self._original_dynamics=self.plant.model.dynamics.__func__
  self._static_words={n:v.detach().clone() for n,v in self.plant.parameters.items()};model=self.plant.model
  self._model_static=[v.detach().clone() for v in [model._J,model._J_inv,model._D,model._G_vec,model.controller.inertia,model.controller.K_angvel,model.controller.mass,model.controller.gravity,model.mass,self.plant.verified_decay]]
  self.budget=json.loads((BASE/'roundoff_enclosure_v1/conditional_projective21_budget.json').read_text());allbounds=json.loads((BASE/'roundoff_enclosure_v1/conditional_budgets.json').read_text())
  row=next(x for x in allbounds['rows'] if x['status']=='conditional_forward_budget_available' and x['verified_decay'] and x['current_command_is_stored_binary32_word'] and x['error_shell_refinement'] and x['range_unit_refinement'] and x['world_position_abs_bound']==16.)
  self.domains=row['checks']['entry_domains'];self._blocked=False
  values=[delay_s]*batch_size if isinstance(delay_s,(int,float)) else list(delay_s)
  if len(values)!=batch_size or any(not F(1,200)<=rational(x)<=F(1,50) for x in values):raise GuardFailure('declared delay not in[.005,.02]s')
  self.delay=torch.tensor(values,dtype=torch.float32);self.history=[torch.tensor([[4.903325,0.,0.,0.]],dtype=torch.float32).expand(batch_size,-1).clone() for _ in range(2)]
  paths=[HERE/'verified_sqrt.py',Path(__file__),BASE/'verified_decay_v1.py',BASE/'cpu_arithmetic_guard_v1.py',BASE/'roundoff_enclosure_v1/conditional_projective21_budget.json']
  self.source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};self.new_source_sha256=self.source_hashes
  self._guard_platform();self._guard_state()
 def _guard_platform(self,advanced=False):
  super()._guard_platform(advanced)
  if self._norm_ready:
   m=self.plant.model
   if (m.step.__func__ is not self.plant.verified_model_step_function or m.dynamics.__func__ is not self._original_dynamics or type(m.controller) is not self.plant.verified_controller_class):self._fail('explicit verified norm program path changed',advanced)
 def step(self,command,*,battery_authority=1.):
  try:result=super().step(command,battery_authority=battery_authority)
  except GuardFailure:raise
  except Exception as error:
   # A motor/substep may already have advanced before a point witness fails.
   self._fail('verified primitive/program rejected: '+type(error).__name__+': '+str(error),True)
  result.update(norm_sqrt_runtime_bin_verified=True,opaque_norm_or_sqrt_library_contract_required=False,norm_witness=self.plant.norm_witness.snapshot(),remaining_arithmetic_assumptions=['fullprecision IEEE binary32 RNE basic operations and scalar conversions','matched original source and tiny matvec dispatch with oneDNN disabled','no concurrent arithmetic mode or code mutation'],flight_certificate=False)
  return result
