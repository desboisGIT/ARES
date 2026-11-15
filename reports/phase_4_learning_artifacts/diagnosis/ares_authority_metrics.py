import json,torch,time
from ares.flight_env import FlightEnv
from ares.flight_training import load_checkpoint
from ares.flight_guidance import guidance_velocity
from ares.worldgen import generate_scene
p=json.load(open('reports/phase_4_repair_final_validation.json'));torch.set_num_threads(1);policy,_=load_checkpoint(p['checkpoints']['bptt']['path'],'cuda')
for mode in ['learned','projected','scaled025']:
 for batch in range(2):
  env=FlightEnv(generate_scene(45000+batch,'apartment','validation'),p['profile'],batch_size=64,device='cuda',seed=900001+batch*7919,horizon=600,randomization='low' if batch==0 else 'high');start=time.monotonic();lateral=0.;latmax=0.;obstacle=0.;count=0
  with torch.no_grad():
   for t in range(600):
    o=env.observation();reference=policy.reference_acceleration(o);desired=policy.desired_acceleration(o);residual=desired-reference
    if mode=='scaled025': desired=reference+.25*residual
    if mode=='projected':
     vg=guidance_velocity(o['p'],o['wp1'],o['wp2'],o['speedlimit']);speed=vg.norm(dim=-1);direction=vg/speed[:,None].clamp_min(1e-6);along=(residual*direction).sum(-1);lower=-.5*policy.config.velocity_gain*speed;desired=desired+direction*(lower-along).clamp_min(0)[:,None]
    live=~env.done;anchor=torch.where((env.waypoint_index==0)[:,None],env.starts,env.targets[:,0]);end=torch.where((env.waypoint_index==0)[:,None],env.targets[:,0],env.targets[:,1]);line=end-anchor;delta=env.adapter.position-anchor;unit=line/line.norm(dim=-1,keepdim=True);error=delta-unit*(delta*unit).sum(-1)[:,None];le=error.norm(dim=-1)
    lateral+=float(error[live].square().sum());latmax=max(latmax,float(le[live].max()));count+=int(live.sum());action=policy.action_from_acceleration(o,desired);obstacle+=float(env.differentiable_loss(action,env.previous_action)['obstacle'][live].sum());env.step(action)
    if env.done.all():break
  print('METRICS',mode,batch,'success',int(env.success.sum()),'collision',int(env.collided.sum()),'minclearance',float(env.minimum_clearance.min()),'peak',float(env.maximum_speed.max()),'lateralRMS',(lateral/count)**.5,'lateralMax',latmax,'obstaclemean',obstacle/count,'seconds',round(time.monotonic()-start,1),'failureidx',((~env.success).nonzero().flatten().tolist()),flush=True)
