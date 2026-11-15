from pathlib import Path
import json,torch,time,hashlib
from ares.flight_env import FlightEnv
from ares.flight_training import load_checkpoint
from ares.phase2 import default_profile
from ares.worldgen import generate_scene
root=Path.cwd();out=root/'reports/phase_4_learning_artifacts/diagnosis';torch.set_num_threads(1);path=root/'reports/phase_4_learning_validation_artifacts/runs/561f1356c8ff4465b8c51e7f5ea2c477/bptt.pt';policy,_=load_checkpoint(path,'cuda');results=[]
for scene,batch,lane in [(45001,1,1),(46001,1,41),(46002,2,12)]:
 env=FlightEnv(generate_scene(scene,'apartment','validation'),default_profile(),batch_size=64,device='cuda',seed=900001+batch*7919,horizon=600,randomization='high' if batch==1 else 'random');handoff=None;maxlat=0.;begin=time.monotonic()
 with torch.no_grad():
  for t in range(600):
   o=env.observation();ref=policy.reference_acceleration(o);res=policy.desired_acceleration(o)-ref;anchor=torch.where((env.waypoint_index==0)[:,None],env.starts,env.targets[:,0]);target=torch.where((env.waypoint_index==0)[:,None],env.targets[:,0],env.targets[:,1]);axis=target-anchor;axis=axis/axis.norm(dim=-1,keepdim=True);res=axis*(res*axis).sum(-1)[:,None];delta=env.adapter.position-anchor;lat=delta-axis*(delta*axis).sum(-1)[:,None];maxlat=max(maxlat,float(lat[lane].norm()));env.step(policy.action_from_acceleration(o,ref+res))
   if env.waypoint_index[lane] and handoff is None:handoff=(t+1)/60
   if env.done[lane]:break
 r={'scene_seed':scene,'lane':lane,'mode':'diagnostic_true_axis_tangential_residual_only','success':bool(env.success[lane]),'collision':bool(env.collided[lane]),'handoff_s':handoff,'min_clearance_m':float(env.minimum_clearance[lane]),'max_lateral_error_m':maxlat,'time_s':float(env.episode_time_s[lane])};results.append(r);print(r,'walltime',time.monotonic()-begin,flush=True)
artifact={'checkpoint':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'warning':'Privileged evaluator true leg axis used solely to isolate lateral residual effect; this is not a deployable controller or an independent validation gate.','results':results};p=out/'contact_lateral_ablation.json';p.write_text(json.dumps(artifact,indent=2)+'\n')
