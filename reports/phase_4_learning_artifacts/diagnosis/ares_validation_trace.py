import json,torch,numpy as np,copy,time
from ares.flight_env import FlightEnv
from ares.flight_training import load_checkpoint
from ares.flight_guidance import guidance_velocity
from ares.phase4 import digest
from ares.worldgen import generate_scene
p=json.load(open('reports/phase_4_repair_final_validation.json'));torch.set_num_threads(1)
policy,_=load_checkpoint(p['checkpoints']['bptt']['path'],device='cuda');reference=copy.deepcopy(policy)
with torch.no_grad(): reference.actor.weight.zero_();reference.actor.bias.zero_()
for batch in range(2):
 expected=[r for r in p['evaluation']['bptt']['records'] if r['scene_seed']==45000+batch]
 failed=[r['task_index'] for r in expected if not r['success']]
 for label,pol in [('learned',policy),('reference',reference)]:
  env=FlightEnv(generate_scene(45000+batch,'apartment','validation'),p['profile'],batch_size=64,device='cuda',seed=900001+batch*7919,horizon=600,randomization='low' if batch==0 else 'high')
  assert digest(env.scene)==expected[0]['scene_sha256']
  records={i:[] for i in failed};handoff={i:None for i in failed};closest={i:[999.,999.] for i in failed};started=time.monotonic()
  with torch.no_grad():
   for t in range(600):
    obs=env.observation();ref=pol.reference_acceleration(obs);desired=pol.desired_acceleration(obs);res=desired-ref;action=pol.action_from_acceleration(obs,desired)
    td=obs['wp1']-obs['p'];direction=td/td.norm(dim=-1,keepdim=True).clamp_min(1e-6);vd=guidance_velocity(obs['p'],obs['wp1'],obs['wp2'],obs['speedlimit']);vp=(obs['v']*direction).sum(-1)
    terms=env.differentiable_loss(action,env.previous_action)
    for i in failed:
     if env.done[i]:continue
     leg=int(env.waypoint_index[i]);d=float((env.adapter.position[i]-env.targets[i,leg]).norm());closest[i][leg]=min(closest[i][leg],d)
     if leg==1 and handoff[i] is None:handoff[i]=t/60
     records[i].append([t/60,leg,d,float(td[i].norm()),float(env.adapter.velocity[i].norm()),float(vp[i]),float(vd[i].norm()),float((ref[i]*direction[i]).sum()),float((res[i]*direction[i]).sum()),float(terms['obstacle'][i]),float(terms['gaze'][i]),float(terms['target_velocity'][i])])
    env.step(action)
    if bool(env.done.all()):break
  print('COHORT',batch,label,'success',int(env.success.sum()),'collision',int(env.collided.sum()),'seconds',round(time.monotonic()-started,1),flush=True)
  for i,r in records.items():
   a=np.array(r);tail=a[-60:];print('LANE',batch,label,i,'success',bool(env.success[i]),'collision',bool(env.collided[i]),'handoff',handoff[i],'closest',closest[i],'final',np.round(a[-1],5).tolist(),'tailmean',np.round(tail.mean(0),5).tolist(),flush=True)
