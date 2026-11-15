"""Minimal guarded eager-CPU numerical backend prototype; NEVER flight certified.

Original full13D plant and normalizedquaternion update, exact-verified motor exp,
public SI queue words. Numeric-domain guards are not a true-state/map supplier or
an RPI admission proof. Primitive/library contracts remain explicit assumptions.
"""
from pathlib import Path
from fractions import Fraction as F
import importlib.util,ctypes,json,hashlib,math
import torch

OUT=Path(__file__).resolve().parent
PARENT=OUT.parent
class GuardFailure(RuntimeError):
 def __init__(self,reason,*,state_advanced=False):
  super().__init__(reason);self.reason=reason;self.state_advanced=state_advanced;self.flight_certificate=False

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def rational(x):
 value=float(x)
 if not math.isfinite(value):raise GuardFailure('nonfinite numeric input')
 return F.from_float(value)

class ConditionalCPUBackend:
 def __init__(self,profile,*,batch_size=1,delay_s=.0125):
  guard=load('ares_cpu_guard',PARENT/'cpu_arithmetic_guard_v1.py')
  self.guard_module=guard;self.precondition_report=guard.probe()
  # Dedicated-worker setting: exclude the oneDNN branch preceding tiny bmm.
  torch.backends.mkldnn.enabled=False
  frozen=PARENT.parent/'frozen_flight_runtime'/'source_manifest.json'
  self.frozen_source_manifest=json.loads(frozen.read_text())
  package=OUT.parents[3]/'ares'
  if any(not (package/name).is_file() or hashlib.sha256((package/name).read_bytes()).hexdigest()!=digest for name,digest in self.frozen_source_manifest.items()):
   raise GuardFailure('frozen original package source mismatch')
  if not self.precondition_report['cpu_preconditions_met']:raise GuardFailure('CPU arithmetic preconditions failed')
  self.helper=ctypes.CDLL(self.precondition_report['helper']['library_path'])
  self.helper.guard_read_mxcsr.restype=ctypes.c_uint
  self.helper.guard_read_round.restype=ctypes.c_int
  self.helper.guard_nearest_constant.restype=ctypes.c_int
  Verified=load('ares_verified_decay',PARENT/'verified_decay_v1.py').make_adapter_class()
  self.plant=Verified(profile,batch_size=batch_size,device='cpu',dt=.01,integration_dt=.002,compile_dynamics=False)
  batch_size=self.plant.batch_size
  self._step_implementation=self.plant.step.__func__
  self._static_words={name:value.detach().clone() for name,value in self.plant.parameters.items()}
  model=self.plant.model
  self._model_static=[value.detach().clone() for value in [model._J,model._J_inv,model._D,model._G_vec,model.controller.inertia,model.controller.K_angvel,model.controller.mass,model.controller.gravity,model.mass,self.plant.verified_decay]]
  self.budget=json.loads((OUT/'conditional_projective21_budget.json').read_text())
  allbounds=json.loads((OUT/'conditional_budgets.json').read_text())
  row=next(x for x in allbounds['rows'] if x['status']=='conditional_forward_budget_available' and x['verified_decay'] and x['current_command_is_stored_binary32_word'] and x['error_shell_refinement'] and x['range_unit_refinement'] and x['world_position_abs_bound']==16.)
  self.domains=row['checks']['entry_domains'];self._blocked=False
  values=[delay_s]*batch_size if isinstance(delay_s,(int,float)) else list(delay_s)
  if len(values)!=batch_size or any(not F(1,200)<=rational(x)<=F(1,50) for x in values):raise GuardFailure('declared delay not in[.005,.02]s')
  self.delay=torch.tensor(values,dtype=torch.float32)
  self.history=[torch.tensor([[4.903325,0.,0.,0.]],dtype=torch.float32).expand(batch_size,-1).clone() for _ in range(2)]
  self.source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [OUT/'conditional_projective21_budget.json',PARENT/'verified_decay_v1.py',PARENT/'cpu_arithmetic_guard_v1.py']}
  self._guard_platform();self._guard_state()

 def _fail(self,reason,advanced=False):
  self._blocked=True;raise GuardFailure(reason,state_advanced=advanced)
 def _guard_platform(self,advanced=False):
  c=self.guard_module.check_control_values(self.helper.guard_read_mxcsr(),self.helper.guard_read_round(),self.helper.guard_nearest_constant())
  try:autocast=torch.is_autocast_enabled('cpu')
  except TypeError:autocast=torch.is_autocast_cpu_enabled()
  compiling=getattr(torch.compiler,'is_compiling',lambda:False)()
  if (not c['passed'] or torch.get_num_threads()!=1 or torch.get_default_dtype()!=torch.float32
      or torch.get_float32_matmul_precision()!='highest' or autocast or compiling
      or torch.jit.is_tracing() or torch.jit.is_scripting() or torch.backends.mkldnn.enabled):self._fail('CPU dtype/precision/rounding/eager precondition lost',advanced)
  if self.plant.compile_dynamics or self.plant.step.__func__ is not self._step_implementation:
   self._fail('verified eager program path changed',advanced)
 def _tensor_words(self,value,width,name,advanced=False):
  if (type(value) is not torch.Tensor or value.device.type!='cpu' or value.dtype!=torch.float32
      or value.shape!=(self.plant.batch_size,width) or value.requires_grad or not bool(torch.isfinite(value).all())):
   self._fail(name+' must be finite plain CPU Float32 stored words',advanced)
  return value.detach().tolist()
 def _in(self,value,interval):return rational(interval['lo'])<=rational(value)<=rational(interval['hi'])
 def _guard_words(self,value,intervals,name,advanced=False):
  rows=self._tensor_words(value,4,name,advanced)
  if any(not self._in(x,intervals[i]) for row in rows for i,x in enumerate(row)):self._fail(name+' domain lost',advanced)
 def _guard_parameters(self,advanced=False):
  if any(not torch.equal(value,self._static_words[name]) for name,value in self.plant.parameters.items()):self._fail('static profile words changed',advanced)
  model=self.plant.model
  current=[model._J,model._J_inv,model._D,model._G_vec,model.controller.inertia,model.controller.K_angvel,model.controller.mass,model.controller.gravity,model.mass,self.plant.verified_decay]
  if any(not torch.equal(value,old) for value,old in zip(current,self._model_static)):self._fail('static physical/controller/verified-decay words changed',advanced)
  names={'mass_kg':'m','gravity_mps2':'g','linear_drag':'drag','motor_lag_s':'tau','inertia_kg_m2':'j'}
  domains=self.domains['physical_parameter_real_ranges']
  for original,key in names.items():
   data=self.plant.parameters[original]
   if data.device.type!='cpu' or data.dtype!=torch.float32 or not bool(torch.isfinite(data).all()):self._fail('physical parameter dtype/nonfinite',advanced)
   values=data.detach().tolist()
   for row in values:
    xs=row if isinstance(row,list) else [row];intervals=domains[key] if isinstance(domains[key],list) else [domains[key]]
    if any(not self._in(x,d) for x,d in zip(xs,intervals)):self._fail('physical parameter range lost:'+original,advanced)
  if (self.plant.dt!=.01 or self.plant.substeps!=5 or float(self.plant.integration_dt)!=.002
      or self.plant.rate_gain!=30. or self.plant.max_body_rate!=6.
      or self.plant.controller_inertia!=(.002,.002,.003)
      or not bool(((self.plant.max_thrust>=22)&(self.plant.max_thrust<=27)).all())):
   self._fail('plant interval/rate/inertia/authority configuration changed',advanced)
 def _guard_state(self,advanced=False):
  self._guard_parameters(advanced)
  rows=self._tensor_words(self.plant.state,13,'state',advanced)
  qrad=list(map(rational,self.domains['projective_qxyz_radii']));eps=F(8,2**24)
  for row in rows:
   for i in [0,1,2,7,8,9,10,11,12]:
    if not self._in(row[i],self.domains['raw13_real_ranges'][i]):self._fail('position/velocity/angular state domain lost',advanced)
   q=list(map(rational,row[3:7]));n2=sum((x*x for x in q),F(0))
   if q[3]<=0 or not (1-eps)**2<=n2<=(1+eps)**2:self._fail('quaternion positive norm shell lost',advanced)
   if any(q[i]*q[i]>qrad[i]*qrad[i]*n2 for i in range(3)):self._fail('projective quaternion domain lost',advanced)
  motor=self.plant.applied_thrust
  if (type(motor) is not torch.Tensor or motor.device.type!='cpu' or motor.dtype!=torch.float32
      or motor.shape!=(self.plant.batch_size,) or motor.requires_grad or not bool(torch.isfinite(motor).all())):self._fail('motor word invalid',advanced)
  if any(not self._in(x,self.domains['motor_real_range']) for x in motor.detach().tolist()):self._fail('motor domain lost',advanced)
  if len(self.history)!=2:self._fail('two-slot history domain lost',advanced)
  for i,word in enumerate(self.history):self._guard_words(word,self.domains['history_word_real_ranges'][i],f'history{i}',advanced)

 def reset(self,*args,**kwargs):
  self._guard_platform();self.plant.reset(*args,**kwargs)
  self.history=[torch.tensor([[4.903325,0.,0.,0.]],dtype=torch.float32).expand(self.plant.batch_size,-1).clone() for _ in range(2)]
  self._blocked=False;self._guard_state();return self.plant.state.detach().clone()

 def step(self,command,*,battery_authority=1.):
  if self._blocked:raise GuardFailure('backend latched blocked; explicit reset and valid domains required')
  self._guard_platform();self._guard_state()
  self._guard_words(command,self.domains['current_word_real_ranges'],'newcommand')
  battery=torch.as_tensor(battery_authority,dtype=torch.float32,device='cpu').expand(self.plant.batch_size)
  if not bool(torch.isfinite(battery).all()) or any(not F(19,20)<=rational(x)<=1 for x in battery.tolist()):self._fail('represented battery authority not in[.95,1]')
  frames=self.delay/.01
  if not bool(((frames>=.5)&(frames<=2)).all()):self._fail('causal delay queue domain lost')
  low=frames.floor().long();high=frames.ceil().long();f=(frames-low)[:,None]
  history=torch.stack([command]+self.history);env=torch.arange(self.plant.batch_size)
  applied=history[low,env]*(1-f)+history[high,env]*f
  applied=torch.cat((applied[:,:1]*battery[:,None],applied[:,1:]),-1)
  with torch.no_grad():self.plant.step(applied)
  self.history=[command.detach().clone(),self.history[0]]
  self._guard_platform(True);self._guard_state(True)
  return {'state':self.plant.state.detach().clone(),'conditional_numeric_budget_applicable':True,
          'numeric_component_error_upper_rational':self.budget['component_error_upper_rational'],
          'input_current_word_quantization_external':True,'RPI_admission_proved_here':False,
          'platform_primitive_contract_audited':False,'flight_certificate':False,
          'cpu_preconditions_met':True,'elapsed_simulation_H_s':.01}
