import sys
from pathlib import Path
sys.path.insert(0,'/home/ares-user/Documents/projects/ARES')
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hashlib,json,math,time
from pathlib import Path
import numpy as np, torch
import prototype as interval
from ares.diffaero_adapter import DiffAeroAdapter
from ares.flight_control import ControllerConfig,acceleration_to_action

torch.set_num_threads(2)
root=Path('/home/ares-user/Documents/projects/ARES')
names=['diffaero_adapter.py','flight_control.py','vendor/diffaero/quadrotor.py','vendor/diffaero/controller.py','vendor/diffaero/math.py']
hashes={name:hashlib.sha256((root/'ares'/name).read_bytes()).hexdigest() for name in names}
profile=json.loads((root/'ares/data/configs/drone_sim_v2.json').read_text())
rng=np.random.default_rng(1171);n=256
profiles=[]
for index in range(n):
 p={name:value['nominal'] for name,value in profile['parameters'].items()}
 for name in ('mass_kg','gravity_mps2','linear_drag','motor_lag_s','max_thrust_n'):
  lo,hi=profile['parameters'][name]['range'];p[name]=lo if index==0 else hi if index==1 else rng.uniform(lo,hi)
 p['inertia_kg_m2']=[lo if index==0 else hi if index==1 else rng.uniform(lo,hi) for lo,hi in profile['parameters']['inertia_kg_m2']['range']]
 profiles.append(p)

results=[]
for mode in ('fixed_hover','pd_backup'):
 for large in (False,True):
  uncertainty=(.03,.05,.01,.05) if large else (.005,.01,.001,.01)
  label='larger_public_set' if large else 'small_public_set'
  bound=interval.run(label,uncertainty,mode,True)
  steps=min(10,len(bound['states']));ep,ev,eq,ew=uncertainty
  plant=DiffAeroAdapter(profiles,dt=.01)
  p=6+torch.from_numpy(rng.uniform(-ep,ep,(n,3))).float()
  v=torch.from_numpy(rng.uniform(-ev,ev,(n,3))).float()
  xyz=torch.from_numpy(rng.uniform(-eq,eq,(n,3))).float();q=torch.cat([xyz,(1-xyz.square().sum(-1,keepdim=True)).sqrt()],-1)
  w=torch.from_numpy(rng.uniform(-ew,ew,(n,3))).float()
  plant.reset(position=p,velocity=v,quaternion=q,angular_velocity=w)
  queue=[torch.tensor([4.903325,0.,0.,0.]).expand(n,-1) for _ in range(3)]
  delay=torch.from_numpy(rng.uniform(.005,.02,n)).float()/.01
  low=delay.floor().long();high=delay.ceil().long();fraction=(delay-low)[:,None]
  misses=[];max_excess=0.;initial=interval.initial(*uncertainty)
  initial_state=plant.state.numpy()
  if not np.all((initial_state>=initial.lo)&(initial_state<=initial.hi)):raise AssertionError('Initial sample outside declared set')
  with torch.no_grad():
   for step in range(steps):
    if mode=='fixed_hover':action=torch.tensor([4.903325,0.,0.,0.]).expand(n,-1)
    else:
     obs={'p':plant.position+torch.from_numpy(rng.uniform(-ep,ep,(n,3))).float(),
          'v':plant.velocity+torch.from_numpy(rng.uniform(-ev,ev,(n,3))).float(),'q':plant.quaternion}
     obs['gaze']=obs['p']+torch.tensor([1.,0.,0.])
     desired=-1.5*(obs['p']-6)-2.5*obs['v']
     action=acceleration_to_action(obs,desired,ControllerConfig())
    queue.insert(0,action);del queue[3:]
    history=torch.stack(queue);applied=history[low,torch.arange(n)]*(1-fraction)+history[high,torch.arange(n)]*fraction
    plant.step(applied)
    state=plant.state.numpy();motor=plant.applied_thrust.numpy()
    b=bound['states'][step];lo=np.asarray(b['state']['lo']);hi=np.asarray(b['state']['hi'])
    excess=np.maximum(lo-state,state-hi);max_excess=max(max_excess,float(excess.max()))
    bad=np.any(excess>0,axis=1)|(motor<b['motor']['lo'])|(motor>b['motor']['hi'])
    if np.any(bad):misses.append({'step':step+1,'bad_samples':np.flatnonzero(bad).tolist(),'max_state_excess':float(excess.max())})
  results.append({'mode':mode,'public_set':label,'samples':n,'steps':steps,'physical_transitions':n*steps,'all_points_contained':len(misses)==0,'max_positive_state_excess':max_excess,'misses':misses,'limits':'Finite points are a sanity test only; no proof of set or compiled-target enclosure.'})
assert hashes=={name:hashlib.sha256((root/'ares'/name).read_bytes()).hexdigest() for name in names}
result={'scope':'CPU float32 actual DiffAero trajectory containment sanity checks; no package edits or target arithmetic proof.','source_sha256':hashes,'seed':1171,'results':results}
Path(__file__).with_name('containment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
