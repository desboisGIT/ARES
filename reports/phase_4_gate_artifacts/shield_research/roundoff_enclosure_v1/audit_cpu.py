"""Finite original-Float32/Float64 frame comparison; not a primitive proof."""
import sys,importlib.util,math,json,hashlib
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent;PARENT=OUT.parent
sys.path.insert(0,str(ROOT))
import numpy as np,torch
from ares.vendor.diffaero.quadrotor import QuadrotorModel

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
control=load('guard',PARENT/'cpu_arithmetic_guard_v1.py').probe()
if not control['cpu_preconditions_met']:raise RuntimeError('CPU guard failed: '+str(control))
Verified=load('verified',PARENT/'verified_decay_v1.py').make_adapter_class()
source=json.loads((PARENT/'taylor_remainder_v1/increment_inflated_flow_bounds.json').read_text())
r=np.array(source['state_radii']);ur=np.array([.116,.224,.224,.0243]);B=128;rng=np.random.default_rng(2251)
base=json.loads((ROOT/'ares/data/configs/drone_sim_v2.json').read_text());profiles=[]
for i in range(B):
 p={k:v['nominal'] for k,v in base['parameters'].items()}
 for name in ['mass_kg','gravity_mps2','linear_drag','motor_lag_s','max_thrust_n','compute_latency_s']:
  lo,hi=base['parameters'][name]['range'];p[name]=lo if i==0 else hi if i==1 else rng.uniform(lo,hi)
 p['inertia_kg_m2']=[lo if i==0 else hi if i==1 else rng.uniform(lo,hi) for lo,hi in base['parameters']['inertia_kg_m2']['range']]
 profiles.append(p)
plant=Verified(profiles,device='cpu',dt=.01)
position=torch.tensor(rng.uniform(-10,10,(B,3)),dtype=torch.float32)
velocity=torch.tensor(rng.uniform(-.95,.95,(B,3))*r[3:6],dtype=torch.float32)
xyz=torch.tensor(rng.uniform(-.95,.95,(B,3))*r[6:9],dtype=torch.float32)
q=torch.cat([xyz,(1-xyz.square().sum(-1,keepdim=True)).sqrt()],-1)
w=torch.tensor(rng.uniform(-.95,.95,(B,3))*r[9:12],dtype=torch.float32)
plant.reset(position,velocity,q,w)
battery=torch.tensor(rng.uniform(.9505,1,B),dtype=torch.float32);battery[0]=1
m=plant.mass.double();g=plant.gravity.double();mg=m*g;teq=mg/battery.double()
plant.applied_thrust=(mg+torch.tensor(rng.uniform(-.95,.95,B)*r[12],dtype=torch.float64)).float()
history=[]
for slot in (-1,13,17):
 radii=ur if slot==-1 else r[slot:slot+4]
 values=torch.tensor(rng.uniform(-.95,.95,(B,4))*radii,dtype=torch.float64)
 values[:,0]+=teq
 history.append(values.float())
latency=plant.parameters['compute_latency_s'];delay=latency/.01;low=delay.floor().long();high=delay.ceil().long();fraction=(delay-low)[:,None]
h=torch.stack(history);index=torch.arange(B)
applied=h[low,index]*(1-fraction)+h[high,index]*fraction
applied=torch.cat([applied[:,:1]*battery[:,None],applied[:,1:]],-1)
# Reference uses the same represented physical parameters. Nominal firmware J
# uses its exact decimal ideal; its binary32 constructor cast is in the budget.
reference=QuadrotorModel(m,plant.parameters['inertia_kg_m2'].double(),plant.parameters['linear_drag'].double()[:,None].expand(-1,3),g,
                         torch.full((B,3),30.,dtype=torch.float64),torch.tensor([[.002,.002,.003]],dtype=torch.float64).expand(B,-1))
state=plant.state.double().clone();state[:,3:7]/=state[:,3:7].norm(dim=-1,keepdim=True)
motor=plant.applied_thrust.double().clone()
d64=torch.tensor([p['compute_latency_s']/.01 for p in profiles],dtype=torch.float64);lo=d64.floor().long();hi=d64.ceil().long();f=(d64-lo)[:,None]
h64=h.double();u64=h64[lo,index]*(1-f)+h64[hi,index]*f
u64=torch.cat([u64[:,:1]*battery.double()[:,None],u64[:,1:]],-1)
qstart=plant.quaternion.double().tolist()
def shell_ok(values):
 n2=sum((F.from_float(float(x))**2 for x in values),F(0));e=F(8,2**24)
 return (1-e)**2<=n2<=(1+e)**2
assert all(shell_ok(q) for q in qstart)
with torch.no_grad():
 plant.step(applied)
 decay=torch.exp(-.002/plant.motor_lag.double())
 # Vendorquat_axisallocatesitsbasisusingdefaultdtype. OnlytheFloat64
 # referencephaseusesDoubledefault; theactualverifiedCPUstepaboveusedFloat32.
 torch.set_default_dtype(torch.float64)
 for _ in range(5):
  motor=u64[:,0]+(motor-u64[:,0])*decay
  action=torch.cat([(motor/mg)[:,None],u64[:,1:]],-1)
  state=reference.step(state,action,.002)
 torch.set_default_dtype(torch.float32)
raw=(plant.state.double()-state).abs();projected=plant.quaternion.double();projected/=projected.norm(dim=-1,keepdim=True)
qerror=(projected-state[:,3:7]).abs()[:,:3];motorerror=(plant.applied_thrust.double()-motor).abs()
aug=torch.cat([raw[:,:3],raw[:,7:10],qerror,raw[:,10:13],motorerror[:,None],torch.zeros(B,8,dtype=torch.float64)],-1)
record=json.loads((OUT/'conditional_projective21_budget.json').read_text());bounds=torch.tensor([float(F(x)) for x in record['component_error_upper_rational']],dtype=torch.float64)
excess=(aug-bounds).clamp_min(0)
assert bool((excess==0).all())
assert all(shell_ok(q) for q in plant.quaternion.double().tolist())
report={'scope':'128finite CPU eager fullbody original-Float32 vsFloat64 one-H implementation comparisons, verified-decay backend. No samples define or fit analytic error budgets; no global binary audit.',
 'samples':B,'full_body_steps':B,'H_s':.01,'all_21_error_components_below_conditional_ceiling':True,
 'maximum_observed_21_error':aug.amax(0).tolist(),'ceiling_21':bounds.tolist(),'max_positive_excess':float(excess.max()),
 'all_input_output_quaternion_shells_passed_exact':True,'cpu_guard':control,'platform_primitive_contract_audited':False,'flight_certificate':False,
 'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'conditional_projective21_budget.json',PARENT/'verified_decay_v1.py',PARENT/'cpu_arithmetic_guard_v1.py',ROOT/'ares/vendor/diffaero/quadrotor.py',ROOT/'ares/vendor/diffaero/math.py']}}
(OUT/'finite_cpu_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['cpu_guard','source_sha256']},indent=2))
