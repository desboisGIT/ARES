"""Independent, seeded closed-loop development regression; no learned policy.

All outcomes, including acquisition failures and invalid initial conditions, are
retained. Finite empirical tests do not establish universal shield assumptions.
"""
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .contracts import load_json, validate_profile
from .diffaero_adapter import DiffAeroAdapter
from .flight_collision import FlightCollisionChecker
from .flight_randomization import sample_flight_profile, sampled_profile_values, randomization_manifest
from .geometry import voxel_mesh
from .safety import SafetyShield, ShieldConfig

DT = .01
FAMILIES = ('hover', 'low_speed_hold', 'vertical_up', 'vertical_down',
            'horizontal_brake', 'three_dimensional_brake', 'occupied_wall', 'unknown_slab')


def development_room(*, unknown_slab=False):
    """Analytic 12 m room, independent of generated train/test scene IDs."""
    n = 60
    occupied = {(x,y,z) for x in range(n) for y in range(n) for z in range(n)
                if 0 in (x,y,z) or n-1 in (x,y,z)}
    cells = [list(c) for c in sorted(occupied)]
    observed = np.ones((n,n,n), dtype=np.uint8)
    for c in occupied:
        observed[c] = 2
    if unknown_slab:
        observed[45,:,:] = 0  # Observed unknown; physical room remains empty here.
    scene = {'schema_version':'ares.scene.v1', 'scene_id':'control_development_analytic_room',
             'seed':0, 'units':'m', 'frame':'right_handed_z_up',
             'license':{'spdx':'CC0-1.0','commercial_ok':True},
             'grid':{'resolution':.2,'origin':[0.,0.,0.],'shape':[n,n,n],'occupied':cells},
             'collision_mesh':voxel_mesh(occupied,resolution=.2,origin=(0.,0.,0.)),
             'rooms':[{'room_id':'room','label':'Analytic development room',
                       'polygon':[[.2,.2],[11.8,.2],[11.8,11.8],[.2,11.8]]}],
             'doors':[], 'surfaces':[{'material':'normal','voxel_indices':cells}],
             'lighting':[{'room_id':'room','state':'lit'}], 'spawn':[6.,6.,6.]}
    return scene, observed


def make_cases(profile, count=8, seed=0, *, initial_tilt_deg=0.):
    """Predeclare deterministic strata before running any dynamics."""
    if type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError('positive integer count and integer seed required')
    validate_profile(profile)
    if isinstance(initial_tilt_deg,bool) or not math.isfinite(initial_tilt_deg) or not 0 <= initial_tilt_deg <= 20:
        raise ValueError('initial_tilt_deg must be finite in [0,20] degrees')
    rng = random.Random(seed)
    cases = []
    velocities = ([0,0,0],[.15,-.1,.05],[0,0,1.],[0,0,-1.],
                  [1.,0,0],[.7,.5,-.5],[1.,0,0],[1.,0,0])
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        # Each family receives both joint endpoints before random profiles.
        mode = ('low','high')[index // len(FAMILIES)] if index < 16 else ('low' if index < 20 else 'random')
        sample_seed = seed * 1000003 + index
        sampled = sample_flight_profile(profile, sample_seed, mode)
        p = [6.,6.,6.]
        if family == 'occupied_wall': p = [10.2,6.,6.]
        if family == 'unknown_slab': p = [7.4,6.,6.]
        v = list(velocities[index % len(FAMILIES)])
        if index >= 16:
            p = [p[k] + rng.uniform(-.15,.15) for k in range(3)]
            v = [value * rng.uniform(.75,1.25) for value in v]
        counterexample_family = None
        if 16 <= index < 20:
            # Four independent development placements: mechanism probes only.
            p = ([4.2,3.2,.9],[2.3,4.1,1.1],[.8,1.3,1.4],[4.4,3.5,.8])[index-16]
            v = [0.,0.,0.]; family = 'hover'
            counterexample_family = f'low_mass_low_altitude_{index-16}'
        # Explicit, pre-run invalid strata are never removed from the report.
        invalid = None
        if index >= 24 and index % 29 == 0:
            p = [11.7,6.,6.]; invalid = 'invalid_initial_clearance'
        elif index >= 24 and index % 29 == 1:
            v = [4.5,0.,0.]; invalid = 'initial_overspeed'
        elif index >= 24 and index % 29 == 2:
            p = [11.3,6.,6.]; v = [2.,0.,0.]; invalid = 'braking_out_of_space'
        # Independent stream never changes profile/position/velocity draws.
        tilt_rng = random.Random(f'ares.control_validation.tilt:{seed}:{index}')
        roll, pitch = [math.radians(initial_tilt_deg)/math.sqrt(2)*tilt_rng.choice((-1,1)) for _ in range(2)]
        sr,cr,sp,cp = math.sin(roll/2),math.cos(roll/2),math.sin(pitch/2),math.cos(pitch/2)
        imposed_q = [sr*cp,cr*sp,-sr*sp,cr*cp] if initial_tilt_deg > 0 else None
        actual_tilt = math.degrees(math.acos(max(-1.,min(1.,math.cos(roll)*math.cos(pitch)))))
        cases.append({'case_id':index,'family':family,'seed':sample_seed,'profile_mode':mode,
                      'initial_tilt_deg':actual_tilt,'initial_roll_pitch_deg':[math.degrees(roll),math.degrees(pitch)],
                      'imposed_quaternion_xyzw':imposed_q,'counterexample_family':counterexample_family,'profile':sampled,'position':p,'velocity':v,'declared_invalid':invalid})
    return cases


def _nominal_action(obs, acceleration):
    # Frozen pre-repair mapper: nominal mass times requested specific thrust.
    from .flight_policy import quaternion_rotate, quaternion_inverse_rotate
    q = obs['q']; z = torch.zeros_like(acceleration); z[:,2] = 1
    body_z = quaternion_rotate(q,z)
    force = acceleration + 9.80665*z
    desired_z = force / force.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    rates = 6. * quaternion_inverse_rotate(q,torch.linalg.cross(body_z,desired_z,dim=-1))
    thrust = (.5*(force*body_z).sum(-1)).clamp(0,24.5)
    return torch.cat((thrust[:,None],rates.clamp(-6.,6.)),-1)


def _observation(plant, generator, bounds):
    obs = {'p':plant.position.clone(), 'v':plant.velocity.clone(), 'q':plant.quaternion.clone()}
    for key, name in (('p','position_m'),('v','velocity_mps')):
        noise = torch.rand(obs[key].shape,generator=generator)*2-1
        obs[key] += noise.to(plant.device)*bounds[name]
    obs['gaze'] = obs['p'] + torch.tensor([1.,0.,0.],device=plant.device)
    return obs


def _hashes():
    folder = Path(__file__).parent
    names = ('control_validation.py','flight_control.py','diffaero_adapter.py','flight_collision.py',
             'safety.py','flight_policy.py','flight_randomization.py','geometry.py','contracts.py',
             'vendor/diffaero/quadrotor.py','vendor/diffaero/controller.py','vendor/diffaero/math.py')
    return {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in names}


def _terminal_metrics(series, *, terminal_speed=.4, fixed_latency_s=.4):
    """Finite observed tail/active-braking evidence, with fixed public cutoffs."""
    speed = series[:,3:6].norm(dim=-1)
    tracking = (series[:,6:9]-series[:,9:12]).norm(dim=-1)
    times = (torch.arange(len(series))+1)*DT
    terminal = torch.nonzero(speed <= terminal_speed).flatten()
    entry = int(terminal[0]) if len(terminal) else None
    tail_radius = (float((series[entry:,:3]-series[entry,:3]).norm(dim=-1).max())
                   if entry is not None else None)
    after_latency = torch.arange(len(series)) >= max(0, math.ceil(fixed_latency_s/DT)-1)
    active = after_latency & (speed > terminal_speed)
    braking = -(series[:,9:12]*series[:,3:6]).sum(-1)/speed.clamp_min(1e-8)
    return {'terminal_entry_time_s':float(times[entry]) if entry is not None else None,
            'terminal_tail_radius_m':tail_radius,
            'terminal_tail_sample_count':len(series)-entry if entry is not None else 0,
            'fixed_latency_s':fixed_latency_s,
            'fixed_latency_active_speed_floor_mps':terminal_speed,
            'fixed_latency_min_active_braking_mps2':float(braking[active].min()) if active.any() else None,
            'fixed_latency_active_braking_sample_count':int(active.sum()),
            'fixed_latency_max_tracking_error_mps2':float(tracking[after_latency].max()) if after_latency.any() else None,
            'fixed_latency_tracking_sample_count':int(after_latency.sum())}


def run_control_validation(profile, device='cpu', count=8, seed=0, output_dir=None, *,
                           duration_s=4., settle_s=2., batch_size=64,
                           compile_dynamics=False, noise_bounds=None, command_delay_s=0.,
                           shield_interval_s=.1, include_nominal=True, initial_tilt_deg=0.):
    """Run batched rigid-body stop/hold cases; return a JSON-serializable report.

    Bounds are uniform componentwise, independently drawn with recorded seeds.
    Transport delay includes sampled profile compute latency, rounded UP to DT.
    Observer receives public noisy state and issued commands, never plant truth.
    """
    from .flight_control import (ControllerConfig, initialize_memory, update_memory,
                                acceleration_to_action, stop_acceleration)
    if isinstance(profile,(str,Path)): profile = load_json(profile)
    validate_profile(profile)
    if (type(batch_size) is not int or batch_size < 1 or duration_s <= 0 or settle_s < 0
            or command_delay_s < 0 or shield_interval_s < DT
            or not all(math.isfinite(x) for x in (duration_s,settle_s,command_delay_s,shield_interval_s))):
        raise ValueError('invalid intervals or batch size')
    bounds = {'position_m':.005,'velocity_mps':.01}
    bounds.update(noise_bounds or {})
    if set(bounds) != {'position_m','velocity_mps'} or any(not math.isfinite(x) or x < 0 for x in bounds.values()):
        raise ValueError('noise bounds require finite nonnegative position_m and velocity_mps')
    source_hashes = _hashes()
    cases = make_cases(profile,count,seed,initial_tilt_deg=initial_tilt_deg)
    scene, observed = development_room()
    unknown_observed = observed.copy(); unknown_observed[45,:,:] = 0
    checker = FlightCollisionChecker(scene,device)
    config = ControllerConfig()
    shield = SafetyShield(ShieldConfig(control_interval_s=DT,latency_s=.1,
                                        body_radius_m=profile['parameters']['body_radius_m']['range'][1],
                                        max_speed_mps=4.,acceleration_error_mps2=0.,
                                        terminal_speed_mps=.4,terminal_stop_radius_m=.2))
    if (math.sqrt(3)*bounds['position_m'] > shield.config.position_uncertainty_m
            or math.sqrt(3)*bounds['velocity_mps'] > shield.config.velocity_uncertainty_mps):
        raise ValueError('noise vector norm bounds exceed configured shield estimation uncertainty')
    results = []; traces = {}; started = time.perf_counter()
    for variant in (('repaired','nominal') if include_nominal else ('repaired',)):
        for first in range(0,count,batch_size):
            group = cases[first:first+batch_size]; b = len(group)
            plant = DiffAeroAdapter([c['profile'] for c in group],device=device,dt=DT,
                                    integration_dt=.002,compile_dynamics=compile_dynamics)
            radii = torch.tensor([c['profile']['parameters']['body_radius_m']['nominal'] for c in group],device=device)
            initial = torch.tensor([c['position'] for c in group],device=device)
            plant.reset(position=initial)
            generator = torch.Generator().manual_seed(seed+first)
            obs = _observation(plant,generator,bounds); memory = initialize_memory(obs,config)
            stop = torch.ones(b,dtype=torch.bool,device=device)
            delay = torch.tensor([math.ceil((c['profile']['parameters']['compute_latency_s']['nominal']+
                                             command_delay_s)/DT) for c in group],device=device)
            max_delay = int(delay.max().item())
            queue = [plant.last_action.clone() for _ in range(max_delay+1)]
            history = []; acquisition_history = []; settle_error = torch.zeros(b,device=device)
            min_clearance = torch.full((b,),float('inf'),device=device)
            collision = torch.zeros(b,dtype=torch.bool,device=device)
            left_bounds = collision.clone(); unknown_crossing = collision.clone()
            initial_class = []; initial_cert = []
            for c in group:
                grid = unknown_observed if c['family']=='unknown_slab' else observed
                decision = shield.filter(grid,c['position'],c['velocity'],[0.,0.,0.],voxel_size=.2,
                                         assumptions_valid=False)
                initial_cert.append(asdict(decision) | {'safe_acceleration':decision.safe_acceleration.tolist()})
                initial_class.append(c['declared_invalid'] or ('valid_brakable' if decision.state_viable else 'braking_out_of_space'))
            settle_steps = round(settle_s/DT); run_steps = math.ceil(duration_s/DT)
            with torch.no_grad():
                for step in range(settle_steps+run_steps):
                    if step == settle_steps:
                        plant.state[:,7:10] = torch.tensor([c['velocity'] for c in group],device=device)
                        if initial_tilt_deg > 0:
                            plant.state[:,3:7] = torch.tensor([c['imposed_quaternion_xyzw'] for c in group],device=device)
                            plant.state[:,10:13] = 0
                        # The externally imposed test initial condition is public.
                        obs = _observation(plant,generator,bounds)
                        memory = replace(memory,velocity=obs['v'].detach().clone(),
                                         holding=torch.zeros_like(memory.holding))
                        if initial_tilt_deg > 0:
                            memory = replace(memory,previous_quaternion=obs['q'].detach().clone())
                        start_position = plant.position.clone()
                    obs.update(memory.observation_state())
                    requested = stop_acceleration(obs,config=config)
                    if step >= settle_steps and (step-settle_steps) % max(1,round(shield_interval_s/DT)) == 0:
                        requests = []; emergencies = []
                        for j,c in enumerate(group):
                            grid = unknown_observed if c['family']=='unknown_slab' else observed
                            decision = shield.filter(grid,obs['p'][j].cpu().tolist(),obs['v'][j].cpu().tolist(),
                                                     requested[j].cpu().tolist(),voxel_size=.2,assumptions_valid=False)
                            requests.append(decision.safe_acceleration)
                            emergencies.append(decision.emergency)
                        # Shield remains explicitly uncertified. Its actual stop command drives the plant.
                        shield_request = torch.tensor(np.asarray(requests),device=device,dtype=torch.float32)
                        shield_emergency = torch.tensor(emergencies,device=device)
                    if step >= settle_steps:
                        coupled = torch.tensor([c['family'] in ('occupied_wall','unknown_slab') for c in group],device=device)
                        smooth_stop = stop_acceleration(obs,max_acceleration=shield.config.min_braking_acceleration_mps2,config=config)
                        shield_request_now = torch.where(shield_emergency[:,None],smooth_stop,shield_request)
                        requested = torch.where(coupled[:,None],shield_request_now,requested)
                    action = (acceleration_to_action(obs,requested,config) if variant=='repaired'
                              else _nominal_action(obs,requested))
                    queue.append(action); queue.pop(0)
                    queued = torch.stack(queue,0)
                    applied = queued[max_delay-delay,torch.arange(b,device=device)]
                    p0 = plant.position.clone(); v0 = plant.velocity.clone()
                    plant.step(applied)
                    check = checker.check(p0,plant.position,radii)
                    collision |= check['collision']; left_bounds |= check['left_bounds']
                    min_clearance = torch.minimum(min_clearance,check['clearance'])
                    if step < settle_steps:
                        settle_error = torch.maximum(settle_error,(plant.position-initial).norm(dim=-1))
                        acquisition_history.append(torch.cat((plant.position,plant.velocity,requested,(plant.velocity-v0)/DT,
                                                              action,plant.quaternion,check['clearance'][:,None]),-1).cpu())
                    else:
                        for j,c in enumerate(group):
                            if c['family']=='unknown_slab':
                                unknown_crossing[j] |= bool(max(float(p0[j,0]),float(plant.position[j,0]))+float(radii[j]) >= 9.)
                        history.append(torch.cat((plant.position,plant.velocity,requested,(plant.velocity-v0)/DT,
                                                  action,plant.quaternion,check['clearance'][:,None]),-1).cpu())
                    next_obs = _observation(plant,generator,bounds)
                    memory = update_memory(memory,next_obs,action,DT,config,stop_requested=stop)
                    obs = next_obs
            series = torch.stack(history,0)
            start_cpu = start_position.cpu()
            for j,c in enumerate(group):
                s = series[:,j]; speed = s[:,3:6].norm(dim=-1)
                displacement = s[:,:3]-start_cpu[j]; v = torch.tensor(c['velocity'],dtype=torch.float32)
                direction = v/v.norm().clamp_min(1e-8)
                along = displacement@direction
                tilt = torch.acos((1-2*(s[:,16]**2+s[:,17]**2)).clamp(-1,1))
                tracking = (s[:,6:9]-s[:,9:12]).norm(dim=-1)
                reaction_s = float(delay[j])*DT + 3*float(plant.motor_lag[j])
                eligible = (torch.arange(len(s))*DT >= reaction_s) & (speed > .25)
                braking = -(s[:,9:12]*s[:,3:6]).sum(-1)/speed.clamp_min(1e-8)
                min_braking = float(braking[eligible].min()) if eligible.any() else None
                stopped = torch.nonzero(speed < .1).flatten()
                anchor_error = float((plant.position[j]-memory.hold_anchor[j]).norm())
                terminal_metrics = _terminal_metrics(s)
                terminal_contract_failed = (terminal_metrics['terminal_tail_radius_m'] is not None
                                            and terminal_metrics['terminal_tail_radius_m'] > shield.config.terminal_stop_radius_m)
                acquisition_failed = float(settle_error[j]) > .3
                failed = bool(collision[j] or left_bounds[j] or unknown_crossing[j] or acquisition_failed or terminal_contract_failed or speed[-1]>.25)
                record = {'case_id':c['case_id'],'variant':variant,'family':c['family'],'seed':c['seed'],
                          'profile_mode':c['profile_mode'],'counterexample_family':c['counterexample_family'],'sampled_parameters':sampled_profile_values(c['profile']),
                          'initial_position_m':c['position'],'initial_velocity_mps':c['velocity'],
                          'initial_tilt_deg':c['initial_tilt_deg'],'initial_roll_pitch_deg':c['initial_roll_pitch_deg'],
                          'imposed_quaternion_xyzw':c['imposed_quaternion_xyzw'],
                          'initial_class':initial_class[j],'initial_shield_state_viable':initial_cert[j]['state_viable'],
                          'shield_certified':False,'emergency_mapping':'terminal_stop_controller','coupled_shield':c['family'] in ('occupied_wall','unknown_slab'),'collision':bool(collision[j]),'left_bounds':bool(left_bounds[j]),
                          'unknown_crossing':bool(unknown_crossing[j]),'failed':failed,
                          'terminal_contract_failed':terminal_contract_failed,
                          'settle_max_position_error_m':float(settle_error[j]),'acquisition_failed':acquisition_failed,
                          'setup':('imposed public velocity/quaternion perturbation and zero body rates after hover acquisition'
                                   if initial_tilt_deg > 0 else 'imposed public velocity perturbation after hover acquisition'),
                          'metrics':{'stopping_distance_m':float(along.max().clamp_min(0)),
                                     'time_to_first_speed_below_point1_s':float(stopped[0])*DT if len(stopped) else None,
                                     'measured_min_braking_after_reaction_mps2':min_braking,
                                     'braking_measurement_speed_floor_mps':.25,
                                     'reaction_exclusion_s':reaction_s,
                                     'reaction_metric_scope':'privileged sampled actuator-time-constant diagnostic; not runtime bound',
                                     'final_hold_anchor_error_m':anchor_error,
                                     'drift_m':float(displacement[-1].norm()),'overshoot_m':float((-along.min()).clamp_min(0)),
                                     'max_speed_mps':float(speed.max()),'final_speed_mps':float(speed[-1]),
                                     'max_tilt_rad':float(tilt.max()),'min_clearance_lower_bound_m':float(min_clearance[j]),
                                     'acceleration_tracking_rmse_mps2':float(tracking.square().mean().sqrt()),
                                     'max_acceleration_error_mps2':float(tracking.max()),
                                     'command_age_s':float(delay[j])*DT,'command_delay_s':float(delay[j])*DT,
                                     'effective_delay_including_motor_time_constant_s':float(delay[j])*DT+float(plant.motor_lag[j]),
                                     'final_position_error_m':float(displacement[-1].norm())}}
                record['metrics'].update(terminal_metrics)
                results.append(record)
                if failed or c['case_id'] < 8:
                    traces[f'{variant}_{c["case_id"]}'] = s.tolist()
                    traces[f'{variant}_{c["case_id"]}_acquisition'] = (torch.stack(acquisition_history)[:,j].tolist() if acquisition_history else [])
    summary = {}
    for variant in sorted({r['variant'] for r in results}):
        selected = [r for r in results if r['variant']==variant]
        valid = [r for r in selected if r['initial_class']=='valid_brakable']
        summary[variant] = {'count':len(selected),'valid_count':len(valid),
                            'valid_gate_passed':bool(valid) and not any(r['failed'] for r in valid),
                            'invalid_count':len(selected)-len(valid),
                            'terminal_contract_failures':sum(r['terminal_contract_failed'] for r in selected),'failures':sum(r['failed'] for r in selected),
                            'collisions':sum(r['collision'] for r in selected),
                            'initial_classes':dict(Counter(r['initial_class'] for r in selected)),
                            'valid_brakable_failures':sum(r['failed'] and r['initial_class']=='valid_brakable' for r in selected)}
    valid_repaired = [r for r in results if r['variant']=='repaired' and r['initial_class']=='valid_brakable']
    def empirical_metric(name, reduction=max):
        values = [r['metrics'][name] for r in valid_repaired if r['metrics'][name] is not None]
        return {'value':reduction(values) if values else None,'case_count':len(values)}
    empirical_bounds = {
        'finite_empirical':True,'universal_certification':False,'population':'all valid repaired cases, including failures',
        'valid_case_count':len(valid_repaired),
        'max_acquisition_drift_m':{'value':max((r['settle_max_position_error_m'] for r in valid_repaired),default=None),
                                   'case_count':len(valid_repaired)},
        'max_terminal_tail_radius_m':empirical_metric('terminal_tail_radius_m'),
        'max_post_latency_tracking_error_mps2':empirical_metric('fixed_latency_max_tracking_error_mps2'),
        'min_measured_active_braking_mps2':empirical_metric('fixed_latency_min_active_braking_mps2',min),
        'active_braking_sample_count':sum(r['metrics']['fixed_latency_active_braking_sample_count'] for r in valid_repaired),
        'tracking_sample_count':sum(r['metrics']['fixed_latency_tracking_sample_count'] for r in valid_repaired),
        'terminal_tail_sample_count':sum(r['metrics']['terminal_tail_sample_count'] for r in valid_repaired)}
    if _hashes() != source_hashes:
        raise RuntimeError('validation source files changed during run; refusing mismatched provenance')
    report = {'schema_version':'ares.control_validation.v1','seed':seed,'device':str(device),
              'source_hashes':source_hashes,'source_hashes_stable':True,'scene_id':scene['scene_id'],
              'profile_sha256':hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest(),
              'configuration':{'dt_s':DT,'rate_loop_hz':500,'duration_s':duration_s,'settle_s':settle_s,
                               'batch_size':batch_size,'compile_dynamics':compile_dynamics,'noise_bounds':bounds,
                               'added_command_delay_s':command_delay_s,'shield_interval_s':shield_interval_s,
                               'initial_tilt_deg':initial_tilt_deg,
                               'controller':asdict(config),'shield':asdict(shield.config)},
              'limitations':['Finite development regression is not universal controller or shield certification.',
                             'Shield assumptions_valid=False throughout; every shield decision is conditional and uncertified.',
                             'Braking velocity is imposed after settling; observer trim is retained and velocity synchronized from public state.',
                             'Unknown slab is observed-map geometry, separate from physical occupied collision.',
                             'Counterexample families reproduce mechanisms on development geometry, not held-out scene 60000 probes.'],
              'nominal_vertical_mass_counterexample':{
                  'requested_net_acceleration_mps2':-1.,
                  'low_endpoint_mass_kg':profile['parameters']['mass_kg']['range'][0],
                  'low_endpoint_gravity_mps2':profile['parameters']['gravity_mps2']['range'][0],
                  'old_nominal_steady_level_zero_speed_acceleration_mps2':
                      .5*(9.80665-1.)/profile['parameters']['mass_kg']['range'][0]-profile['parameters']['gravity_mps2']['range'][0],
                  'scope':'Algebraic counterexample at level, zero speed, settled actuator; actual rollout regressions above measure repaired/nominal hover and braking.'},
              'elapsed_s':time.perf_counter()-started,'randomization':randomization_manifest([c['profile'] for c in cases]),
              'summary':summary,'empirical_bounds':empirical_bounds,'cases':results,'timeseries_columns':['px','py','pz','vx','vy','vz',
              'requested_ax','requested_ay','requested_az','achieved_ax','achieved_ay','achieved_az',
              'command_thrust','command_wx','command_wy','command_wz','qx','qy','qz','qw','clearance_lower_bound'],
              'timeseries_dt_s':DT,'timeseries':traces}
    if output_dir is not None:
        folder = Path(output_dir); folder.mkdir(parents=True,exist_ok=True)
        (folder/'control_validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
