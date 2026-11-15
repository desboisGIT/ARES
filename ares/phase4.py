"""Train, freeze and evaluate flight policies; hardware gates stay separate."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import uuid

import numpy as np
import torch

from .diffaero_adapter import DIFFAERO_ADAPTER_VERSION, UPSTREAM_SHA
from .flight_env import FlightEnv, FLIGHT_ENV_VERSION
from .flight_policy import FlightPolicy
from .flight_training import TrainingConfig, train_policy, save_checkpoint, load_checkpoint, seed_training
from .flight_randomization import randomization_manifest
from .flight_control import stop_acceleration
from .phase2 import default_profile
from .safety import SafetyShield, ShieldConfig, SAFETY_VERSION
from .worldgen import generate_scene, GENERATOR_VERSION


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def runtime_source_hashes():
    """Freeze all package Python sources, including orchestration and rendering."""
    root=Path(__file__).parent
    return {path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*.py'))}


def require_runtime_unchanged(expected):
    if runtime_source_hashes()!=expected:
        raise ValueError('flight runtime changed during training or evaluation')


class CurriculumFlightEnv:
    """Cycle only training geometry at episode boundaries, with one runtime API."""
    def __init__(self, profile, *, device, batch_size, seed, horizon, scene_count=10,stage_updates=5,guidance_speed_fraction=1.):
        self.scenes=[generate_scene(100+i,'apartment','train') for i in range(scene_count)]
        self.profile,self.device,self.batch_size,self.seed,self.horizon=profile,device,batch_size,seed,horizon
        self.guidance_speed_fraction=guidance_speed_fraction
        self.count=0
        self.sampled=[]
        self.env=None
        self.stage_updates=max(1,int(stage_updates))
        self.stage_history=[]

    def reset(self):
        scene=self.scenes[self.count%len(self.scenes)]
        self.env=FlightEnv(scene,self.profile,batch_size=self.batch_size,device=self.device,
                           seed=self.seed+self.count*10007,horizon=self.horizon,
                           guidance_speed_fraction=self.guidance_speed_fraction)
        self.sampled.extend(self.env.sampled_profiles)
        self.count+=1
        return self.env.observation()

    def start_update(self,update):
        if update>0 and update%self.stage_updates==0:
            observation=self.reset()
            self.stage_history.append({'update':update,'scene_id':self.env.scene['scene_id'],
                                       'reason':'declared curriculum boundary; optimizer retained'})
            return observation
        return None

    def reset_idx(self,indices):
        observation=self.env.reset_idx(indices)
        # Retain sampled records for the audit; survivor samples are not replaced.
        mask=torch.as_tensor(indices,device=self.env.device)
        selected=torch.nonzero(mask,as_tuple=False).flatten().tolist() if mask.dtype==torch.bool else mask.flatten().tolist()
        self.sampled.extend(self.env.sampled_profiles[i] for i in selected)
        return observation

    def __getattr__(self,name):
        if self.env is None:
            raise AttributeError(name)
        return getattr(self.env,name)


def inference_benchmark(policy,env,repeats=100):
    observation={key:value[:1].detach() for key,value in env.observation().items()}
    policy.eval()
    with torch.no_grad():
        for _ in range(10): policy(observation)
        if env.device.type=='cuda':torch.cuda.synchronize()
        durations=[]
        for _ in range(repeats):
            start=time.perf_counter(); policy(observation)
            if env.device.type=='cuda':torch.cuda.synchronize()
            durations.append(time.perf_counter()-start)
    return {'host':'current_desktop_not_onboard','device':str(env.device),'samples':repeats,
            'median_ms':float(np.median(durations)*1000),'p90_ms':float(np.percentile(durations,90)*1000),
            'p99_ms':float(np.percentile(durations,99)*1000),'onboard_gate_passed':False,
            'scope':'Single sample; depth encoder, state network and nominal geometric action conversion; sensor, estimator, shield and runtime scheduling excluded.'}


def summarize_flights(records):
    passing=[r['time_s'] for r in records if r['success']]
    return {'runs':len(records),'successes':sum(r['success'] for r in records),
        'success_rate':sum(r['success'] for r in records)/len(records) if records else None,
        'median_time_s':float(np.median(passing)) if passing else None,
        'p90_time_s':float(np.percentile(passing,90)) if passing else None,
        'time_population':'successful runs only; failures stay in success-rate denominator',
        'collisions':sum(r['collision'] for r in records),
        'shield_interventions':sum(r['shield_interventions'] for r in records),
        'certified_commands':sum(r['certified_commands'] for r in records),
        'shield_checks':sum(r['shield_checks'] for r in records),
        'max_speed_mps':max((r['max_speed_mps'] for r in records),default=None),
        'median_peak_speed_mps':float(np.median([r['max_speed_mps'] for r in records])) if records else None,
        'speed_limit_exceeding_runs':sum(r['max_speed_mps']>r.get('speed_limit_mps',float('inf'))+1e-5 for r in records),
        'minimum_clearance_m':min((r['minimum_clearance_m'] for r in records),default=None),
        'clearance_note':'Continuous sphere/voxel sweeps; positive distances outside the local query cap are flagged lower bounds per run.'}


def evaluate_flight_policy(policy,profile,*,device='cpu',episodes=2,batch_size=16,max_steps=600,
                           start_seed=70000,split='test',shield=False,progress=None):
    if type(episodes) is not int or episodes<1 or type(batch_size) is not int or batch_size<1:
        raise ValueError('positive episode and batch counts required')
    policy=policy.to(device).eval()
    records,profiles,scene_hashes=[],[],[]
    batches=math.ceil(episodes/batch_size)
    for batch in range(batches):
        count=min(batch_size,episodes-len(records))
        scene_seed=start_seed+batch
        scene=generate_scene(scene_seed,'apartment',split)
        # Deterministic endpoint probes precede interior random samples. These
        # are not an exhaustive proof over all joint parameter combinations.
        mode='low' if batch==0 else 'high' if batch==1 else 'random'
        control_dt=.01 if shield else 1/60
        evaluation_steps=math.ceil(max_steps*(1/60)/control_dt)
        env=FlightEnv(scene,profile,batch_size=count,device=device,seed=900001+batch*7919,
                      horizon=evaluation_steps,randomization=mode,dt=control_dt,
                      guidance_speed_fraction=getattr(getattr(policy,'config',None),'guidance_speed_fraction',1.))
        scene_hash=digest(scene);scene_hashes.append(scene_hash)
        profiles.extend(env.sampled_profiles)
        interventions=np.zeros(count,dtype=int);certified=interventions.copy();checks=interventions.copy()
        shields=[]
        if shield:
            for i in range(count):
                params=env.sampled_profiles[i]['parameters']
                shields.append(SafetyShield(ShieldConfig(body_radius_m=params['body_radius_m']['nominal'],
                    control_interval_s=env.dt,latency_s=.2+params['compute_latency_s']['nominal']+params['motor_lag_s']['nominal'],
                    max_speed_mps=env.speed_limit,terminal_speed_mps=.4,terminal_stop_radius_m=.2)))
        started=time.perf_counter()
        next_policy_time=0.; acceleration=None
        with torch.no_grad():
            for _ in range(evaluation_steps):
                observation=env.observation()
                if shield:
                    if acceleration is None or env.time_s+1e-8>=next_policy_time:
                        acceleration=policy.desired_acceleration(observation)
                        next_policy_time+=1/60
                    safe=acceleration.clone()
                    emergency=torch.zeros(count,device=device,dtype=torch.bool)
                    stop_command=stop_acceleration(observation,max_acceleration=1.)
                    for i in range(count):
                        if env.done[i]: continue
                        velocity=observation['v']
                        decision=shields[i].filter(env.known_grid(i),observation['p'][i].cpu().numpy(),
                            velocity[i].cpu().numpy(),acceleration[i].cpu().numpy(),
                            origin=env.origin_np,voxel_size=env.resolution,assumptions_valid=False)
                        emergency[i]=decision.emergency
                        safe[i]=stop_command[i] if decision.emergency else torch.as_tensor(decision.safe_acceleration,device=device,dtype=safe.dtype)
                        interventions[i]+=not decision.accepted
                        certified[i]+=decision.certified;checks[i]+=1
                    action=policy.action_from_acceleration(observation,safe)
                else:
                    action=policy(observation)
                if shield:env.step(action,stop_requested=emergency)
                else:env.step(action)
                if bool(env.done.all()):break
        for i in range(count):
            records.append({'scene_seed':scene_seed,'scene_sha256':scene_hash,'split':split,
                'task_seed':env.seed,'task_index':i,'task':env.tasks[i],'profile_mode':mode,
                'profile_sha256':digest(env.sampled_profiles[i]),
                'success':bool(env.success[i]),'collision':bool(env.collided[i]),
                'time_s':int(env.episode_steps[i])*env.dt,
                'max_speed_mps':float(env.maximum_speed[i]),
                'speed_limit_mps':env.speed_limit,'speed_limit_exceeding_steps':int(env.speed_limit_steps[i]),
                'minimum_clearance_m':float(env.minimum_clearance[i]),
                'minimum_clearance_is_lower_bound':bool(env.minimum_clearance_is_lower_bound[i]),
                'shield_interventions':int(interventions[i]),'certified_commands':int(certified[i]),
                'shield_checks':int(checks[i])})
        if progress:progress({'evaluated':len(records),'total':episodes,'shield':shield,
                               'batch_successes':int(env.success.sum()),'wall_s':time.perf_counter()-started})
    cohort=[{key:r[key] for key in ('scene_sha256','task_seed','task_index','task','profile_sha256','profile_mode')}
            for r in records]
    return {'summary':summarize_flights(records),'records':records,
        'profile_mode_summary':{mode:{'runs':sum(r['profile_mode']==mode for r in records),
            'successes':sum(r['profile_mode']==mode and r['success'] for r in records)} for mode in ('low','high','random')},
        'randomization':randomization_manifest(profiles),'scene_sha256':scene_hashes,
        'dataset_sha256':digest(cohort),'split':split,'shield_enabled':shield,
        'shield_assumptions_validated':False,'policy_hz':60,'shield_hz':100 if shield else None,
        'randomization_coverage_note':'All-low, all-high and random interior batches; finite tests do not exhaust joint parameter combinations.'}


def run_phase4(*,device='cpu',train_updates=1,batch_size=2,rollout_steps=4,episodes=2,
               max_steps=10,output_dir=Path('reports/phase_4_artifacts'),seed=1741,progress=None,
               shield_episodes=4,eval_split='test',eval_start_seed=70000,algorithms=('bptt','ppo'),
               residual_acceleration=None,residual_penalty=0.,guidance_speed_fraction=1.):
    if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
        raise ValueError('requested device must be available; no silent fallback')
    if residual_acceleration is not None and (not math.isfinite(residual_acceleration) or residual_acceleration <= 0):
        raise ValueError('residual_acceleration must be finite and positive')
    if not math.isfinite(residual_penalty) or residual_penalty < 0:
        raise ValueError('residual_penalty must be finite and nonnegative')
    if not math.isfinite(guidance_speed_fraction) or not 0 < guidance_speed_fraction <= 1:
        raise ValueError('guidance_speed_fraction must be finite and in (0, 1]')
    torch.set_num_threads(1)
    algorithms=tuple(algorithms)
    if not algorithms or len(set(algorithms))!=len(algorithms) or any(name not in ('bptt','ppo') for name in algorithms) or 'bptt' not in algorithms:
        raise ValueError('include the primary bptt policy and optional ppo comparison once each')
    profile=default_profile()
    source_hashes=runtime_source_hashes()
    root=Path(output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    directory=root/'runs'/uuid.uuid4().hex;directory.mkdir(parents=True)
    policies,training,checkpoints={},{},{}
    for algorithm in algorithms:
        env=CurriculumFlightEnv(profile,device=device,batch_size=batch_size,seed=seed,
                                horizon=max_steps,scene_count=2 if train_updates<5 else 10,
                                stage_updates=max(1,train_updates//(2 if train_updates<5 else 10)),
                                guidance_speed_fraction=guidance_speed_fraction)
        config=TrainingConfig(steps=train_updates,rollout_steps=rollout_steps,seed=seed,
                               minibatch_size=min(256,batch_size*rollout_steps),
                               residual_penalty=residual_penalty)
        callback=(lambda row,algorithm=algorithm:progress({'algorithm':algorithm,'training':row})) if progress else None
        if residual_acceleration is None and guidance_speed_fraction == 1:
            policy,result=train_policy(env,algorithm=algorithm,config=config,progress=callback)
        else:
            seed_training(seed)
            policy_config={'guidance_speed_fraction':guidance_speed_fraction}
            if residual_acceleration is not None:
                policy_config['residual_acceleration']=residual_acceleration
            policy=FlightPolicy(policy_config).to(device)
            policy,result=train_policy(env,algorithm=algorithm,config=config,policy=policy,progress=callback)
        checkpoint=directory/f'{algorithm}.pt'
        save_checkpoint(checkpoint,policy,algorithm=algorithm,seed=seed,training_steps=result['training_steps'],
            training_curve=result['training_curve'],metadata={'profile':profile,'adapter_version':DIFFAERO_ADAPTER_VERSION,
                'upstream_sha':UPSTREAM_SHA,'train_scene_seeds':list(range(100,100+min(getattr(env,'count',len(env.scenes)),len(env.scenes)))),
                'training_config':asdict(config),'runtime_source_sha256':source_hashes,
                'environment':{'policy_hz':60,'rate_loop_minimum_hz':500,'max_steps':max_steps,'speed_limit_mps':1.5},
                'controller':policy.architecture,'controller_version':policy.config.controller_version,
                'guidance_version':policy.config.guidance_version})
        policies[algorithm]=policy;training[algorithm]=result
        checkpoints[algorithm]={'path':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        result['curriculum_stage_updates']=getattr(env,'stage_updates',None)
        result['curriculum_stage_history']=getattr(env,'stage_history',[])
        if getattr(env,'sampled',None):result['randomization']=randomization_manifest(env.sampled)
        (directory/f'{algorithm}_training.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    require_runtime_unchanged(source_hashes)
    # Freeze complete artifacts before generating any held-out tasks.
    frozen={name:dict(value) for name,value in checkpoints.items()}
    evaluation={};inference={}
    for algorithm,policy in policies.items():
        policy,payload=load_checkpoint(checkpoints[algorithm]['path'],device=device)
        evaluation[algorithm]=evaluate_flight_policy(policy,profile,device=device,episodes=episodes,
            batch_size=batch_size,max_steps=max_steps,progress=progress,split=eval_split,start_seed=eval_start_seed)
        if shield_episodes:
            evaluation[algorithm]['shielded_probe']=evaluate_flight_policy(policy,profile,device=device,
                episodes=min(episodes,shield_episodes),batch_size=min(batch_size,shield_episodes),
                max_steps=max_steps,shield=True,progress=progress,split=eval_split,start_seed=eval_start_seed)
        timing_env=FlightEnv(generate_scene(100,'apartment','train'),profile,batch_size=1,
                            device=device,seed=seed,horizon=max_steps,
                            guidance_speed_fraction=policy.config.guidance_speed_fraction)
        inference[algorithm]=inference_benchmark(policy,timing_env)
        if hashlib.sha256(Path(frozen[algorithm]['path']).read_bytes()).hexdigest()!=frozen[algorithm]['sha256']:
            raise ValueError('checkpoint changed during held-out evaluation')
        require_runtime_unchanged(source_hashes)
    if len({result['dataset_sha256'] for result in evaluation.values()})!=1:
        raise ValueError('flight policy comparison cohorts differ')
    software=all(result['optimizer_updates']>0 and all(math.isfinite(row['loss']) for row in result['training_curve'])
                 for result in training.values())
    success_gate=eval_split=='test' and episodes>=1000 and evaluation['bptt']['summary']['success_rate']>=.99
    report={'schema_version':'ares.phase4.v2','env_version':FLIGHT_ENV_VERSION,'adapter_version':DIFFAERO_ADAPTER_VERSION,
        'upstream_sha':UPSTREAM_SHA,'generator_version':GENERATOR_VERSION,'profile':profile,'seed':seed,
        'device':device,'python':platform.python_version(),'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
        'execution':{'cuda_dynamics_compiled':device=='cuda','policy_hz':60,'shield_hz':100,
            'rate_loop_minimum_hz':500,'controller_inertia_kg_m2':[.002,.002,.003],
            'controller_inertia_is_nominal_not_sampled':True},
        'controller_version':'adaptive_v2','guidance_version':'braking_v2','safety_version':SAFETY_VERSION,
        'algorithms':list(algorithms),
        'policy_configuration':{name:asdict(policy.config) for name,policy in policies.items()},
        'training':training,'checkpoints':checkpoints,'runtime_source_sha256':source_hashes,
        'evaluation':evaluation,'desktop_inference':inference,
        'software_preparation_passed':software,'waypoint_reliability_gate':{'passed':success_gate,'primary_policy':'bptt','required_success_rate':.99,
            'required_held_out_runs':1000,'scope':'Finite two-leg randomized tasks; not every possible joint parameter combination.'},
        'shield_gate':{'passed':False,'status':
            'coupled_quadrotor_probe_failed; tracking_bounds_unvalidated' if any(
                value.get('shielded_probe',{}).get('summary',{}).get('collisions',0)>0 for value in evaluation.values()) else
            'conditional_geometry_tested; quadrotor_tracking_bounds_unvalidated',
            'probe_collisions':{name:value['shielded_probe']['summary']['collisions'] for name,value in evaluation.items() if 'shielded_probe' in value},
            'uncertified_closed_loop_checks_are_not_collision_prevention_certificates':True},
        'onboard_inference_gate':{'passed':False,'status':'pending_actual_onboard_computer','required_ms':5},
        'phase_passed':False,'artifacts':str(directory),
        'limitations':['Nominal geometric feedback plus learned residual, not a purely learned controller.',
            'Randomized estimator and battery/power approximations; no physical rate-loop/estimator calibration.',
            'Depth capture cadence is approximated by an age model; actuator and sensor delivery queues are explicit.',
            'Shield is conditional on acceleration/tracking/map bounds not established for this aircraft.',
            'Phase 3 human timings and Phase 2 real recording/onboard gates remain pending.',
            'Phase 4b maneuver skills are outside this phase.']}
    (directory/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
