import json,torch,time
from ares.flight_env import FlightEnv
from ares.flight_training import load_checkpoint
from ares.flight_guidance import guidance_velocity
from ares.worldgen import generate_scene
p=json.load(open('reports/phase_4_repair_final_validation.json'));torch.set_num_threads(1);policy,_=load_checkpoint(p['checkpoints']['bptt']['path'],'cuda')
for batch in range(2):
 env=FlightEnv(generate_scene(45000+batch,'apartment','validation'),p['profile'],batch_size=64,device='cuda',seed=900001+batch*7919,horizon=600,randomization='low' if batch==0 else 'high');handoff=torch.full((64,),-1.,device='cuda');n=0;affected=torch.zeros(64,dtype=torch.bool,device='cuda');start=time.monotonic()
 with torch.no_grad():
  for t in range(600):
   o=env.observation();reference=policy.reference_acceleration(o);desired=policy.desired_acceleration(o);residual=desired-reference;vg=guidance_velocity(o['p'],o['wp1'],o['wp2'],o['speedlimit']);speed=vg.norm(dim=-1);direction=vg/speed[:,None].clamp_min(1e-6);along=(residual*direction).sum(-1);lower=-.5*policy.config.velocity_gain*speed
   correction=(lower-along).clamp_min(0);affected|=(correction>0)&~env.done;n+=int(((correction>0)&~env.done).sum());desired=desired+direction*correction[:,None]
   env.step(policy.action_from_acceleration(o,desired));handoff=torch.where((handoff<0)&(env.waypoint_index==1),torch.full_like(handoff,(t+1)/60),handoff)
   if env.done.all():break
 print('PROJECTED',batch,'success',int(env.success.sum()),'collision',int(env.collided.sum()),'affected',int(affected.sum()),'steps',n,'time',round(time.monotonic()-start,2),'failureidx',((~env.success).nonzero().flatten().tolist()),'handoff_failureoriginal',[(i,round(float(handoff[i]),3)) for i,r in enumerate(p['evaluation']['bptt']['records'][batch*64:(batch+1)*64]) if not r['success']],flush=True)
