"""Prospective full-mission development with unchanged NN and renewable backup.

Global60Hz Gaussian NN/sensor queues never reset. Only verified guard epochs
renew after a new public constant-command history admits a new full-state family.
Truth remains confined to restore, sensors, and separate functional oracles.
"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[2]
for folder in [ROOT,HERE,BASE,BASE/'learned_command_bridge_v1',BASE/'coupled_reanchoring_v1',BASE/'bounded_service_v1',BASE/'renewable_trim_v1']:
    sys.path.insert(0,str(folder))
import argparse,json,hashlib,math,time
from fractions import Fraction as F
from dataclasses import replace
from collections import Counter
import torch
from ares.contracts import load_json
from ares.flight_training import load_checkpoint
from ares.flight_randomization import sample_flight_profile
from ares.flight_collision import FlightCollisionChecker
from bounded_service import BoundedSupplier,preflight_scan
from sensor_renderer import TruthState,TestOracle
from reanchoring_shield_v1 import ProofBundle,ReanchoringShield,InitialAdmission
from run_coupled import PlantAndOracle,public_initial_packet,ATTESTATION,hashes as base_hashes
from gaussian_observer import GaussianObserver
from adapter import word_intervention
from adapter_v2 import propose_anchor
from scene_v2 import make_scene
from mapper import ForwardMapper,extend_guard_free_snapshot
from producerwide_v2 import PublicTrimWindow,recalibration_candidate,target_from_public_commands
from trim_target import choose_trim_word
from energy import bootstrap_bound,PublicEnergyLedger
from epochs import prepare_epoch_switch,rebase_packet,encode


def hashes():
    paths=list(HERE.glob('*.py'))+[BASE/'renewable_trim_v1'/n for n in ['producer.py','producerwide_v2.py','energy.py','epochs.py']]
    d=base_hashes();d.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths});return d


def sensor(private,renderer,*,tick,case,lease_s):
    row=private.backend.plant.state[0].tolist()
    truth=TruthState(tuple(row[:3]),tuple(row[7:10]),tuple(row[3:7]),tuple(row[10:13]))
    supplier=BoundedSupplier(renderer,truth,seed=f'fullmission:{case}:{tick}')
    packet=supplier.capture(capture_time_s=float(F(tick,100)),expires_s=lease_s)
    if not TestOracle.state_error_valid(truth,packet):raise RuntimeError('bounded supplier error check failed')
    return supplier,packet


def run(case=0,steps=4500,task_seed=90100,output='development_low.json',calibration_remaining=430):
    if not 0<=case<8 or not 1<=steps<=4500 or task_seed<90100:raise ValueError('case0..7, steps1..4500, newseed>=90100 required')
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_flush_denormal(False)
    started=time.perf_counter();before_hash=hashes();lease_tick=5500;lease_s=55.
    checkpoint=load_json(ROOT/'reports/phase_4_gate.json')['checkpoints']['bptt'];checkpoint_path=Path(checkpoint['path'])
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()!=checkpoint['sha256']:raise RuntimeError('checkpoint changed')
    policy,payload=load_checkpoint(checkpoint_path,'cpu');proof=ProofBundle(ROOT/'reports/phase_4_shield_artifacts/cpu_rpi_certificate_v1')
    capture=load_json(BASE/'public_trim_capture_v1.json');public_history=load_json(BASE/'public_trim_history_v1.json')
    mode,seed=capture['profile_modes_and_seeds'][case];profile=load_json(ROOT/'ares/data/configs/drone_sim_v2.json');sampled=sample_flight_profile(profile,seed,mode)
    scene,renderer=make_scene(task_seed)
    preflight=BoundedSupplier(renderer,TruthState((6.125,6.125,2.125),(0.,0.,0.)),seed=task_seed)
    known=preflight_scan(preflight,preflight.capture(capture_time_s=0.,expires_s=lease_s),time_s=0.)
    if not TestOracle.free_map_valid(renderer,known.free):raise RuntimeError('initial sound-ray map false label')
    admission_proof=InitialAdmission.load(proof,BASE/'public_trim_history_v1.json',case);a=admission_proof.row
    private=PlantAndOracle(sampled,capture['simulation_truth_for_replay_only'],case,renderer,known)
    guard=ReanchoringShield(proof,known,hold_thrust=a['hold_thrust'],trim_force_interval=a['trim_force_interval'],
        motor_residual_interval=a['motor_residual_interval'],issued_queue=a['issued_queue'],deadline_tick=1000,
        service_lease_expires_tick=lease_tick,initial_energy_upper_J=4338,arithmetic_backend_attestation=ATTESTATION,initial_admission=admission_proof)
    boot=bootstrap_bound(maximum_issued_thrust=public_history['maximum_issued_thrust_N'],steps=public_history['full_rigid_body_steps_per_case'],mass_gravity_upper=proof.mg_upper)
    ledger=PublicEnergyLedger(boot['energy_upper'],boot['motor_upper'])
    initial_anchor=list(a['anchor']);start=private.backend.plant.position[0].tolist()
    waypoints=[[initial_anchor[0]+.6,initial_anchor[1]+.2,initial_anchor[2]],
               [initial_anchor[0]+1.2,initial_anchor[1]+1.1,initial_anchor[2]+.1]]
    observer=GaussianObserver(scene,profile,sampled,private.backend.plant,waypoints,task_seed,policy)
    collision=FlightCollisionChecker(scene,device='cpu')
    param=sampled['parameters'];mapper=ForwardMapper(known,horizontal_fov=param['depth_horizontal_fov_rad']['nominal'],
      vertical_fov=param['depth_vertical_fov_rad']['nominal'],max_range=param['depth_max_range_m']['nominal'])
    trace=[];policy_trace=[];renewals=[];receipts=[];mapper_trace=[];reasons=Counter();error=None
    epoch=0;requested=None;window=None;calibrating=False;recent=[];target=None;success=False;empty_streak=0;calibration_start=-1
    sensor_state=private.backend.plant.state.clone();sensor_epoch=F(0)
    for clock in range(steps*3):
        now=F(clock,300)
        if clock%3==0:sensor_state=private.backend.plant.state.detach().clone();sensor_epoch=now
        if clock%5==0:
            with torch.no_grad():
                obs=observer.observe(clock//5,float(now),sensor_state);requested=policy(obs)[0].tolist()
            policy_trace.append({'frame':clock//5,'time_s':float(now),'sensor_epoch_s':float(sensor_epoch),'capture_age_s':float(now-sensor_epoch),
              'requested_command':requested,'estimated_p':obs['p'][0].tolist(),'estimated_v':obs['v'][0].tolist(),
              'depth_shape':list(obs['depth'].shape),'features_shape':list(obs['features'].shape),'waypoint1':obs['wp1'][0].tolist()})
        if clock%3:continue
        tick=clock//3
        try:
            supplier,absolute=sensor(private,renderer,tick=tick,case=case,lease_s=lease_s)
            mapping_packet=absolute
            if tick==0:
                first=public_initial_packet(a);absolute=replace(first,contract=replace(first.contract,expires_s=lease_s))
            mapping={'added':[],'attempted':0}
            if tick%5==0:
                t=time.perf_counter();mapping=mapper.observe(supplier,mapping_packet,time_s=float(F(tick,100)))
                mapper_trace.append({'global_tick':tick,'elapsed_s':time.perf_counter()-t,'known_free_count':len(known.free),
                    'added_cells':mapping['added'],'attempted':mapping['attempted'],'supplier_capture_s':absolute.contract.capture_time_s,
                    'supplier_expiry_s':absolute.contract.expires_s})
                extend_guard_free_snapshot(guard,known,global_tick=tick,epoch_tick=epoch)
            switching=None
            if calibrating and len(window.records)>=270:
                window.note_final_packet(tick=tick,packet=absolute)
                try:
                    switching=prepare_epoch_switch(guard,window,proof=proof,known_map=known,public_energy_upper=ledger.energy,
                        service_lease_expires_tick=lease_tick,current_absolute_packet=absolute,old_epoch_global_tick=epoch)
                    receipts.append({'producer_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'renewable_trim_v1'/n for n in ['producer.py','producerwide_v2.py','energy.py','epochs.py']]},
                        'records':list(window.records),'final':window.final,'new_admission':switching['new_admission'].row,
                        'public_history_sha256':switching['new_admission'].public_history_sha256})
                except Exception as exc:
                    renewals.append({'global_tick':tick,'admitted':False,'type':type(exc).__name__,'message':str(exc),
                        'window_count':len(window.records),'old_epoch_global_tick':epoch})
            if calibrating and empty_streak>=30 and tick-calibration_start>=100 and guard.deadline_tick-guard.tick>300:
                target_proposal=choose_trim_word(trace)
                if abs(target_proposal['target_word']-float(target))>1e-4:
                    target=target_proposal['target_word'];window=PublicTrimWindow(target,proof.source_sha256)
                    calibration_start=tick;empty_streak=0
                    renewals.append({'global_tick':tick,'calibration_target_proposal':target_proposal,'admitted':False,'event':'untrusted_target_retry'})
            if not calibrating and guard.deadline_tick-guard.tick<=calibration_remaining:
                target_proposal=choose_trim_word(trace)
                target=target_proposal['target_word']
                renewals.append({'global_tick':tick,'calibration_target_proposal':target_proposal,'admitted':False,'event':'untrusted_target_selection'})
                window=PublicTrimWindow(target,proof.source_sha256);calibrating=True;calibration_start=tick;empty_streak=0
            before_epoch=epoch
            t=time.perf_counter()
            if switching is not None:
                selected_guard=switching['new_guard'];decision=switching['first_decision'];next_epoch=switching['epoch_global_tick'];proposal={'renewal_first_backup':True}
            else:
                selected_guard=guard;next_epoch=epoch;packet=rebase_packet(absolute,global_tick=tick,epoch_global_tick=epoch)
                if calibrating:
                    proposal=recalibration_candidate(packet,guard.hold,target)
                    decision=guard.decide(packet,tick=tick-epoch,proposed_anchor=proposal['anchor'])
                else:
                    proposal=propose_anchor(guard,packet,requested,initial_anchor)
                    decision=guard.decide(packet,tick=tick-epoch,proposed_anchor=proposal['candidate_anchor'])
            decide_s=time.perf_counter()-t
            if decision['command'] is None:raise RuntimeError('no certified command: '+decision['reason'])
            old_queue=list(selected_guard.queue);before_position=private.backend.plant.position.clone()
            before,numeric,issued=private.step(decision['command']);selected_guard.acknowledge_issued(issued);private.update_energy()
            ledger.advance(issued,old_queue);observer.after_physical_step(issued,before_position)
            if switching is not None:
                guard=selected_guard;epoch=next_epoch;calibrating=False
                renewals.append({'global_tick':tick,'admitted':True,'old_epoch_global_tick':before_epoch,'new_epoch_global_tick':epoch,
                    'new_global_backup_deadline_tick':epoch+1000,'initial_admission_sha256':switching['new_admission'].public_history_sha256,
                    'represented_global_epoch_word_exact':str(F.from_float(float(F(epoch,100)))),
                    'epoch_transform':'Exact global macro tick origin; map queries bracket epoch/100+localword down and up; state expiry subtracts exact epoch/100 and rounds down',
                    'public_energy_upper':ledger.energy,'hold_thrust':guard.hold})
                window=None
            elif calibrating:
                window.note_issued(tick=tick,packet=absolute,decision=decision,acknowledged_command=issued,decision_epoch_global_tick=epoch)
                empty_streak=empty_streak+1 if not window.records else 0
            recent.append(issued);recent=recent[-100:]
            oracle=private.check(before,guard,proof)
            sweep=collision.check(torch.tensor([before]),private.backend.plant.position,private.backend.plant.parameters['body_radius_m'])
            oracle['physical_collision']=bool(sweep['collision'][0] or sweep['left_bounds'][0]);oracle['physical_clearance_m']=float(sweep['clearance'][0])
            oracle['physical_clearance_is_lower_bound']=bool(sweep['clearance_is_lower_bound'][0])
            row={'global_tick':tick,'epoch_tick':epoch,'local_guard_tick_executed':tick-epoch,'requested_command':requested,'issued_command':issued,
                'physical_word_intervention':word_intervention(requested,issued),'proposal':proposal,'proposal_accepted':decision['proposal_accepted'],
                'cached_backup_used':decision.get('cached_backup_used',False),'calibration_mode':calibrating,'proposed_membership_maxima':decision.get('membership_maxima'),
                'public_packet':{n:list(getattr(absolute,n)) for n in ['p','v','q','w']},
                'calibration_window_count':len(window.records) if window else 0,'conditional_motion_certificate':decision['conditional_motion_certificate'],
                'guard_plus_adapter_s':decide_s,'anchor':decision['anchor'],'public_energy_upper':ledger.energy,
                'waypoint_index':int(observer.env.waypoint_index[0]),'mission_reached':observer.reached(),**oracle}
            trace.append(row);reasons[decision['reason']]+=1
            if not oracle['actual_C_included'] or oracle['physical_collision'] or oracle['unknown_entry'] or not oracle['queue_matches_public']:
                raise RuntimeError('functional oracle failed')
            if observer.reached():success=True;break
            if (tick+1)%100==0:print('PROGRESS',tick+1,'wp',int(observer.env.waypoint_index[0]),'dist',math.dist(oracle['position'],waypoints[int(observer.env.waypoint_index[0])]),'epoch',epoch,'calibration',calibrating,'window',len(window.records) if window else 0,flush=True)
        except Exception as exc:
            error={'global_tick':tick,'epoch_tick':epoch,'type':type(exc).__name__,'message':str(exc)};break
    after_hash=hashes();end=private.backend.plant.position[0].tolist()
    report={'schema_version':'ares.renewable_learned_mission_SIL.v2','task_seed':task_seed,'case':case,'profile_mode':mode,'sampled_profile':sampled,
      'scene_id':scene['scene_id'],'scene_sha256':hashlib.sha256(json.dumps(scene,sort_keys=True).encode()).hexdigest(),
      'requested_steps':steps,'completed_steps':len(trace),'mission_success':success,'waypoint_index':int(observer.env.waypoint_index[0]),
      'waypoints':waypoints,'leg_lengths_m':[math.dist(start,waypoints[0]),math.dist(*waypoints)],'reach_thresholds_m':[.1,.35],
      'first_waypoint_progress_m':math.dist(start,waypoints[0])-math.dist(end,waypoints[0]),
      'remaining_current_waypoint_distance_m':math.dist(end,waypoints[int(observer.env.waypoint_index[0])]),
      'physical_word_interventions':sum(x['physical_word_intervention'] for x in trace),'positive_certificates':sum(x['conditional_motion_certificate'] for x in trace),
      'actual_C_misses':sum(not x['actual_C_included'] for x in trace),'collisions':sum(x['physical_collision'] for x in trace),
      'unknown_entries':sum(x['unknown_entry'] for x in trace),'queue_mismatches':sum(not x['queue_matches_public'] for x in trace),
      'max_speed_mps':max((x['speed_mps'] for x in trace),default=0),'minimum_clearance_lower_bound_m':min((x['physical_clearance_m'] for x in trace),default=None),
      'policy_frames':len(policy_trace),'policy_rate_hz':60,'shield_rate_hz':100,'mapper_rate_hz':20,'global_service_lease_s':lease_s,
      'successful_renewals':sum(x['admitted'] for x in renewals),'renewals':renewals,'renewal_receipt_bundles':receipts,
      'public_energy_bootstrap':boot,'public_energy_final':ledger.energy,'known_free_final_count':len(known.free),
      'new_forward_observation_cells':mapper.admitted,'bounded_mapper_rays_attempted':mapper.attempted,'bounded_mapper_range_rejects':mapper.range_rejected,
      'checkpoint_sha256':checkpoint['sha256'],'NN_unchanged_Gaussian_interface':True,'global_NN_clock_queues_memory_never_reset':True,
      'source_hashes_before':before_hash,'source_hashes_after':after_hash,'sources_stable':before_hash==after_hash,
      'producer_driver_sources':{str(p.relative_to(BASE)):p.read_text() for p in [*HERE.glob('*.py'),*[BASE/'renewable_trim_v1'/n for n in ['producer.py','producerwide_v2.py','energy.py','epochs.py']]]},
      'error':error,'all_attempts_retained':True,'phase4_gate_passed':False,'conditional_SIL_only':True,'trace':trace,'policy_trace':policy_trace,'mapper_trace':mapper_trace,
      'runtime_elapsed_s':time.perf_counter()-started}
    path=HERE/output;path.write_text(json.dumps(encode(report),indent=2,allow_nan=False)+'\n')
    print('SAVED',path,json.dumps({k:report[k] for k in ['completed_steps','mission_success','successful_renewals','sources_stable','error']}),flush=True);return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,default=0);p.add_argument('--steps',type=int,default=4500);p.add_argument('--task-seed',type=int,default=90100);p.add_argument('--output',default='development_low.json');p.add_argument('--calibration-remaining',type=int,default=430);a=p.parse_args()
    run(case=a.case,steps=a.steps,task_seed=a.task_seed,output=a.output,calibration_remaining=a.calibration_remaining)
