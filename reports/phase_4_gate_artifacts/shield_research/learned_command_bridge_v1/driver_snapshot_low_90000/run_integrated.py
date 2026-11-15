"""Frozen BPTT requested-command consumer with independent certified CPU shield.

Prospective development SIL, not a Phase4 acceptance result. No NN retuning.
Original Gaussian FlightEnv observations and an additional hard-bound monitor
remain separate. Physical truth is restricted to sensors/oracle and restoration.
"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parent;ROOT=RESEARCH.parents[2]
for folder in [ROOT,HERE,RESEARCH,RESEARCH/'coupled_reanchoring_v1',RESEARCH/'bounded_service_v1']:
    sys.path.insert(0,str(folder))
import argparse,hashlib,json,math,time
from fractions import Fraction as F
from collections import Counter
import torch
from ares.contracts import load_json
from ares.flight_training import load_checkpoint
from ares.flight_randomization import sample_flight_profile
from ares.flight_collision import FlightCollisionChecker
from bounded_service import BoundedSupplier,preflight_scan
from sensor_renderer import TruthState,TestOracle
from reanchoring_shield_v1 import ProofBundle,ReanchoringShield,InitialAdmission
from run_coupled import PlantAndOracle,public_initial_packet,ATTESTATION
from scene import make_scene
from gaussian_observer import GaussianObserver
from adapter import propose_anchor,word_intervention


def source_hashes():
    paths=list((ROOT/'ares').rglob('*.py'))+list(HERE.glob('*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(case=0,steps=100,task_seed=90000,output='smoke.json'):
    if not 0<=case<8 or not 1<=steps<=1000 or task_seed<90000:
        raise ValueError('case0..7, steps1..1000 and prospective seed>=90000 required')
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_flush_denormal(False)
    before_hash=source_hashes();started=time.perf_counter()
    gate=load_json(ROOT/'reports/phase_4_gate.json');checkpoint=gate['checkpoints']['bptt']
    checkpoint_path=Path(checkpoint['path']);actual=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if actual!=checkpoint['sha256']:raise RuntimeError('frozen checkpoint changed')
    policy,payload=load_checkpoint(checkpoint_path,'cpu')
    proof=ProofBundle(ROOT/'reports/phase_4_shield_artifacts/cpu_rpi_certificate_v1')
    capture=load_json(RESEARCH/'public_trim_capture_v1.json');mode,seed=capture['profile_modes_and_seeds'][case]
    profile=load_json(ROOT/'ares/data/configs/drone_sim_v2.json');sampled=sample_flight_profile(profile,seed,mode)
    scene,renderer=make_scene(task_seed)
    # Fresh ray witnesses are rendered against this very same furniture world.
    map_truth=TruthState((6.125,6.125,2.125),(0.,0.,0.))
    supplier=BoundedSupplier(renderer,map_truth,seed=task_seed)
    known=preflight_scan(supplier,supplier.capture(capture_time_s=0.,expires_s=30.),time_s=0.)
    if not TestOracle.free_map_valid(renderer,known.free):raise RuntimeError('invalid ray map')
    admission_proof=InitialAdmission.load(proof,RESEARCH/'public_trim_history_v1.json',case)
    admission=admission_proof.row
    private=PlantAndOracle(sampled,capture['simulation_truth_for_replay_only'],case,renderer,known)
    guard=ReanchoringShield(proof,known,hold_thrust=admission['hold_thrust'],
        trim_force_interval=admission['trim_force_interval'],motor_residual_interval=admission['motor_residual_interval'],
        issued_queue=admission['issued_queue'],deadline_tick=1000,service_lease_expires_tick=1000,
        initial_energy_upper_J=4338,arithmetic_backend_attestation=ATTESTATION,initial_admission=admission_proof)
    anchor=list(admission['anchor']);start=private.backend.plant.position[0].tolist()
    waypoints=[[anchor[0]+.6,anchor[1]+.2,anchor[2]],
               [anchor[0]+1.2,anchor[1]+1.1,anchor[2]+.1]]
    observer=GaussianObserver(scene,profile,sampled,private.backend.plant,waypoints,task_seed,policy)
    collision=FlightCollisionChecker(scene,device='cpu')
    requested=None;trace=[];policy_trace=[];reasons=Counter();error=None
    sensor_state=private.backend.plant.state.clone();sensor_epoch=F(0)
    for clock in range(steps*3):
        now=F(clock,300)
        if clock%3==0:
            sensor_state=private.backend.plant.state.detach().clone();sensor_epoch=now
        if clock%5==0:
            with torch.no_grad():
                observation=observer.observe(clock//5,float(now),sensor_state)
                requested=policy(observation)[0].tolist()
            policy_trace.append({'frame':clock//5,'nominal_time_s':float(now),'physical_capture_epoch_s':float(sensor_epoch),
                'capture_age_s':float(now-sensor_epoch),'requested_command':requested,
                'estimated_p':observation['p'][0].tolist(),'estimated_v':observation['v'][0].tolist(),
                'q':observation['q'][0].tolist(),'wp1':observation['wp1'][0].tolist(),
                'depth_shape':list(observation['depth'].shape),'depth_valid_pixels':int(observation['depth'][:,1].sum()),
                'feature_shape':list(observation['features'].shape)})
        if clock%3:continue
        tick=clock//3
        try:
            packet=public_initial_packet(admission) if tick==0 else private.packet(tick,case)
            t=time.perf_counter();proposal=propose_anchor(guard,packet,requested,anchor)
            decision=guard.decide(packet,tick=tick,proposed_anchor=proposal['candidate_anchor'])
            decide_elapsed=time.perf_counter()-t
            reasons[decision['reason']]+=1
            if decision['command'] is None:raise RuntimeError('no certified command: '+decision['reason'])
            command=decision['command'];before_position=private.backend.plant.position.clone()
            before,numeric,issued=private.step(command)
            t=time.perf_counter();guard.acknowledge_issued(issued);ack_elapsed=time.perf_counter()-t
            private.update_energy();observer.after_physical_step(issued,before_position)
            oracle=private.check(before,guard,proof)
            sweep=collision.check(torch.tensor([before]),private.backend.plant.position,
                                  private.backend.plant.parameters['body_radius_m'])
            oracle['physical_collision']=bool(sweep['collision'][0] or sweep['left_bounds'][0])
            oracle['physical_clearance_m']=float(sweep['clearance'][0])
            row={'tick':tick,'time_s':float(now),'requested_command':requested,'issued_command':issued,
                'physical_word_intervention':word_intervention(requested,issued),'proposal':proposal,
                'proposal_accepted':decision['proposal_accepted'],'cached_backup_used':decision.get('cached_backup_used',False),
                'conditional_motion_certificate':decision['conditional_motion_certificate'],
                'guard_plus_adapter_s':decide_elapsed+ack_elapsed,'anchor':decision['anchor'],
                'numeric_budget_applicable':numeric['conditional_numeric_budget_applicable'],
                'waypoint_index':int(observer.env.waypoint_index[0]),'mission_reached':observer.reached(),**oracle}
            trace.append(row)
            if not oracle['actual_C_included'] or oracle['physical_collision'] or oracle['unknown_entry'] or not oracle['queue_matches_public']:
                raise RuntimeError('functional oracle failed')
            if (tick+1)%100==0:print('PROGRESS',tick+1,'distance',math.dist(oracle['position'],waypoints[0]),flush=True)
        except Exception as exc:
            error={'tick':tick,'type':type(exc).__name__,'message':str(exc)};break
    end=private.backend.plant.position[0].tolist();after_hash=source_hashes()
    report={'schema_version':'ares.learned_command_bridge_SIL.v1','task_seed':task_seed,'case':case,
      'scene_id':scene['scene_id'],'scene_sha256':hashlib.sha256(json.dumps(scene,sort_keys=True).encode()).hexdigest(),
      'known_free_cells':len(known.free),'map_ray_supplier_seed':task_seed,'full_randomized_profile':sampled,
      'checkpoint_sha256':actual,'checkpoint_config':payload['policy_config'],'requested_steps':steps,'completed_steps':len(trace),
      'physical_rate_hz':100,'shield_rate_hz':100,'policy_rate_hz':60,'controller_observer_rate_hz':100,
      'policy_capture_semantics':'latest completed100Hz endpoint; age<10ms recorded; no future state or interpolation',
      'NN_observation_path':'Unmodified frozen FlightEnv._observe + FlightSensors Gaussian/noisy depth/ring + observed controller memory; bounded monitor is separate',
      'waypoints':waypoints,'leg_lengths_m':[math.dist(start,waypoints[0]),math.dist(*waypoints)],
      'shared_reach_thresholds_m':{'first':observer.env.guidance_config.waypoint_radius,'second':.35},
      'mission_success':observer.reached(),'waypoint_index':int(observer.env.waypoint_index[0]),
      'first_waypoint_distance_progress_m':math.dist(start,waypoints[0])-math.dist(end,waypoints[0]),
      'final_first_waypoint_distance_m':math.dist(end,waypoints[0]),
      'physical_word_interventions':sum(x['physical_word_intervention'] for x in trace),
      'positive_conditional_certificates':sum(x['conditional_motion_certificate'] for x in trace),
      'anchor_accepts':sum(x['proposal_accepted'] for x in trace),'decision_reasons':dict(reasons),
      'physical_collisions':sum(x['physical_collision'] for x in trace),'unknown_entries':sum(x['unknown_entry'] for x in trace),
      'actual_C_misses':sum(not x['actual_C_included'] for x in trace),'queue_mismatches':sum(not x['queue_matches_public'] for x in trace),
      'max_speed_mps':max((x['speed_mps'] for x in trace),default=0),
      'minimum_clearance_m':min((x['physical_clearance_m'] for x in trace),default=None),
      'policy_frames':len(policy_trace),'guard_plus_adapter_max_s':max((x['guard_plus_adapter_s'] for x in trace),default=None),
      'source_hashes_stable':before_hash==after_hash,'source_hashes_before':before_hash,
      'proof_sha256':proof.source_sha256,'error':error,'all_cases_retained':True,'phase4_gate_passed':False,
      'scope':'Prospective integration development smoke; no NN tuning, no default-Gaussian hard certificate, no hardware claim',
      'runtime_elapsed_s':time.perf_counter()-started,'trace':trace,'policy_trace':policy_trace}
    path=HERE/output;path.write_text(json.dumps(report,indent=2,allow_nan=False,default=str)+'\n')
    print('SAVED',path,json.dumps({k:report[k] for k in ['completed_steps','mission_success','first_waypoint_distance_progress_m','physical_word_interventions','source_hashes_stable','error']}),flush=True)
    return report
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=int,default=0);parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--task-seed',type=int,default=90000);parser.add_argument('--output',default='smoke.json');a=parser.parse_args()
    run(case=a.case,steps=a.steps,task_seed=a.task_seed,output=a.output)
