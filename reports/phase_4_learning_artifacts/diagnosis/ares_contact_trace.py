from pathlib import Path
import json,torch,time,hashlib
import numpy as np
from ares.flight_env import FlightEnv
from ares.flight_training import load_checkpoint
from ares.phase2 import default_profile
from ares.worldgen import generate_scene
from ares.flight_guidance import guidance_velocity
root=Path.cwd();out=root/'reports/phase_4_learning_artifacts/diagnosis';out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
learned_path=root/'reports/phase_4_learning_validation_artifacts/runs/561f1356c8ff4465b8c51e7f5ea2c477/bptt.pt';reference_path=root/'reports/phase_4_learning_artifacts/reference.pt';policy,payload=load_checkpoint(learned_path,'cuda');reference,rpayload=load_checkpoint(reference_path,'cuda');allresults=[]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for scene_seed,batch,lane in [(45001,1,1),(46001,1,41),(46002,2,12)]:
 for label,pol in [('learned',policy),('reference',reference)]:
  env=FlightEnv(generate_scene(scene_seed,'apartment','validation'),default_profile(),batch_size=64,device='cuda',seed=900001+batch*7919,horizon=600,randomization='high' if batch==1 else 'random');rows=[];handoff=None;start=time.monotonic();task=env.tasks[lane];profile=env.sampled_profiles[lane]
  with torch.no_grad():
   for t in range(600):
    o=env.observation();ref=pol.reference_acceleration(o);desired=pol.desired_acceleration(o);residual=desired-ref;action=pol.action_from_acceleration(o,desired);oldp=env.adapter.position[lane].clone();oldv=env.adapter.velocity[lane].clone();q=env.adapter.quaternion[lane].clone();leg=int(env.waypoint_index[lane]);anchor=env.starts[lane] if leg==0 else env.targets[lane,0];end=env.targets[lane,leg];unit=(end-anchor)/(end-anchor).norm();delta=oldp-anchor;lateral=delta-unit*(delta*unit).sum();vp=(oldv*unit).sum();vg=guidance_velocity(o['p'],o['wp1'],o['wp2'],o['speedlimit']);row={'time_s':t/60,'leg':leg,'position':oldp.tolist(),'velocity':oldv.tolist(),'quaternion':q.tolist(),'target_distance':float((end-oldp).norm()),'lateral_error':lateral.tolist(),'lateral_norm':float(lateral.norm()),'forward_velocity':float(vp),'guidance_velocity':vg[lane].tolist(),'reference_acceleration':ref[lane].tolist(),'residual':residual[lane].tolist(),'desired_acceleration':desired[lane].tolist(),'action':action[lane].tolist(),'controller_bias':o['controller_bias'][lane].tolist(),'observed_position':o['p'][lane].tolist(),'observed_velocity':o['v'][lane].tolist(),'instant_clearance':float(env.clearance(env.adapter.position)[lane])}
    env.step(action);row.update(next_position=env.adapter.position[lane].tolist(),realized_acceleration=((env.adapter.velocity[lane]-oldv)/env.dt).tolist(),applied_thrust=float(env.adapter.applied_thrust[lane]),cumulative_clearance=float(env.minimum_clearance[lane]));rows.append(row)
    if int(env.waypoint_index[lane])==1 and handoff is None:handoff=(t+1)/60
    if bool(env.done[lane]):break
  result={'scene_seed':scene_seed,'batch':batch,'lane':lane,'policy':label,'success':bool(env.success[lane]),'collision':bool(env.collided[lane]),'handoff_s':handoff,'task':task,'sampled_profile':profile,'route_clearance_m':float(env.task_route_clearance_m[lane]),'loss_margin_m':float(env.obstacle_margin_m[lane]),'minimum_clearance_m':float(env.minimum_clearance[lane]),'max_lateral_error_m':max(r['lateral_norm'] for r in rows),'rows':rows};allresults.append(result)
  print('CONTACTTRACE',scene_seed,lane,label,'success',result['success'],'collision',result['collision'],'handoff',handoff,'routeclear',result['route_clearance_m'],'minclear',result['minimum_clearance_m'],'maxlat',result['max_lateral_error_m'],'end',rows[-1],'seconds',round(time.monotonic()-start,1),flush=True)
artifact={'schema_version':'ares.learning_contact_trace.v1','checkpoints':{label:{'path':str(path.relative_to(root)),'sha256':sha(path)} for label,path in [('learned',learned_path),('reference',reference_path)]},'runtime_source_sha256':{name:sha(root/'ares'/name) for name in payload['metadata']['runtime_source_sha256']},'checkpoint_training_source_sha256':payload['metadata']['runtime_source_sha256'],'results':allresults};dest=out/'contact_traces_seed1741.json';dest.write_text(json.dumps(artifact,indent=2)+'\n');print('SAVED',dest,sha(dest),flush=True)
