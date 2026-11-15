"""Coupled conditional-CPU SIL; truth restricted to restore/sensor/oracle code.

No hardware/Gaussian certificate and no overall ARES phase gate is set here.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import gc
import json
import math
import os
import signal
import struct
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parent
ROOT=RESEARCH.parents[2]
for folder in (RESEARCH,RESEARCH/'bounded_service_v1',RESEARCH/'verified_norm_backend_v1',RESEARCH/'terminal_reservation_v1'):
    sys.path.insert(0,str(folder))
import torch
from ares.contracts import load_json
from ares.flight_randomization import sample_flight_profile
from ares.safety_contract import MeasurementContract
from bounded_service import (BoundedSupplier,StatePacket,WitnessMap,POSITION_ERROR_M,VELOCITY_ERROR_MPS)
from sensor_renderer import AnalyticRenderer,TruthState
from backend import ConditionalVerifiedNormCPUBackend
import verified_battery_v1 as battery
from reanchoring_shield_v1 import ProofBundle,ReanchoringShield,InitialAdmission
from current_terminal_membership_v1 import current_terminal_membership
from reservation import free_grid_reserve

STOP_REQUESTED=False
def request_stop(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True
signal.signal(signal.SIGTERM,request_stop)

ATTESTATION={key:True for key in ('explicit_norm_sqrt_runtime_verified','verified_motor_decay',
    'verified_power_and_battery','guarded_IEEE_basic_arithmetic','same_full_rigid_body_RK4_model',
    'no_GPU_or_compiled_path','exact_current_orientation_feedback')}


def hashes():
    paths=list((ROOT/'ares').rglob('*.py'))+[RESEARCH/n for n in (
      'reanchoring_shield_v1.py','current_terminal_membership_v1.py','verified_battery_v1.py',
      'verified_backup_point_v1.py','verified_backup_point_v2.py','verified_backup_point_v2_compatibility.json',
      'assemble_initial_contract_v1.py','initial_terminal_membership_v1.py',
      'trim_history_v2.py','zero_rate_history_v1.py','public_trim_capture_v1.py','public_trim_history_v1.json',
      'public_trim_capture_v1.json','bounded_service_v1/known_free_map.json',
      'verified_decay_v1.py','cpu_arithmetic_guard_v1.py','terminal_reservation_v1/reservation.py',
      'verified_norm_backend_v1/backend.py','verified_norm_backend_v1/verified_sqrt.py','verified_norm_backend_v1/manifest.json',
      'roundoff_enclosure_v1/conditional_projective21_budget.json','roundoff_enclosure_v1/conditional_budgets.json',
      'roundoff_enclosure_v1/conditional_cpu_backend.py','bounded_service_v1/bounded_service.py',
      'bounded_service_v1/sensor_renderer.py')]+[Path(__file__),
        ROOT/'reports/phase_4_shield_artifacts/cpu_rpi_certificate_v1/certificate.json',
        ROOT/'reports/phase_4_shield_artifacts/cpu_rpi_certificate_v1/bundle_sha256.json',
        RESEARCH.parent/'frozen_flight_runtime/source_manifest.json',HERE/'run_sweep.py']
    bundle=ROOT/'reports/phase_4_shield_artifacts/cpu_rpi_certificate_v1'
    paths.extend(bundle/name for name in json.loads((bundle/'bundle_sha256.json').read_text()))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def gc_snapshot():
    return tuple((row['collections'],row['collected'],row['uncollectable']) for row in gc.get_stats())

def gc_delta(after,before):
    return [[a-b for a,b in zip(x,y)] for x,y in zip(after,before)]

def public_initial_packet(admission):
    c=MeasurementContract(position_error_m=POSITION_ERROR_M,velocity_error_mps=VELOCITY_ERROR_MPS,
        world_ray_error=.001,depth_error_m=.01,capture_time_s=0.,expires_s=10.,
        supplier='bounded_service_v1',provenance='SHA-bound public trim capture; rebased mission tick zero',
        source_kind='hard_external',bound_basis='deterministic_guarantee',
        voxel_model='homogeneous_static_until_expiry',depth_model='first_hit_range')
    row=admission['public_final_packet']
    return StatePacket(p=tuple(row['p']),v=tuple(row['v']),q=tuple(row['q']),w=tuple(row['w']),contract=c)


class PlantAndOracle:
    """Private test boundary: never passed into ReanchoringShield."""
    def __init__(self,profile,private,index,renderer,known_map):
        params=profile['parameters']
        raw_delay=params['compute_latency_s']['nominal'];bounded_delay=raw_delay
        while F.from_float(bounded_delay)>F(1,50):bounded_delay=math.nextafter(bounded_delay,0.)
        if F.from_float(bounded_delay)<F(1,200):raise ValueError('constructor delay binding below proof lower endpoint')
        if struct.pack('>f',raw_delay)!=struct.pack('>f',bounded_delay):
            raise ValueError('constructor binding changed physical Float32 delay word')
        self.backend=ConditionalVerifiedNormCPUBackend(profile,delay_s=bounded_delay)
        self.delay_binding={'raw_profile_delay_s':raw_delay,'constructor_bound_delay_s':bounded_delay,
            'represented_float32_delay_s':float(self.backend.delay[0]),
            'normalized_float32_queue_frames':float((self.backend.delay/.01)[0]),
            'physical_float32_word_unchanged':True,'high_boundary_only':bounded_delay!=raw_delay}
        plant=self.backend.plant
        plant.state=torch.tensor([private['final_state'][index]],dtype=torch.float32)
        plant.applied_thrust=torch.tensor([private['final_motor_thrust'][index]],dtype=torch.float32)
        self.backend.history=[torch.tensor([slot[index]],dtype=torch.float32) for slot in private['last_two_issued_commands']]
        plant.last_action=self.backend.history[0].clone()
        self.energy=private['energy_words_J'][index]
        self.capacity=float(plant.parameters['battery_energy_j'][0])
        self.mass_gravity=F.from_float(float(plant.mass[0]))*F.from_float(float(plant.gravity[0]))
        self.radius=F.from_float(float(plant.parameters['body_radius_m'][0]))
        self.renderer,self.known_map=renderer,known_map
        self.physical_free=tuple((x,y,z) for x in range(48) for y in range(48) for z in range(16)
                                 if not renderer.occupied((x,y,z)))
        self.backend._guard_platform();self.backend._guard_state()

    def packet(self,tick,index):
        row=self.backend.plant.state[0].tolist()
        truth=TruthState(tuple(row[:3]),tuple(row[7:10]),tuple(row[3:7]),tuple(row[10:13]))
        return BoundedSupplier(self.renderer,truth,seed=f'coupled:{index}:{tick}').capture(
            capture_time_s=float(F(tick,100)),expires_s=10.)

    def step(self,command):
        before=self.backend.plant.position[0].tolist()
        authority=battery.authority32(self.energy,self.capacity)
        issued=torch.tensor([command],dtype=torch.float32)
        result=self.backend.step(issued,battery_authority=authority)
        return before,result,issued[0].tolist()

    def update_energy(self):
        self.energy=battery.energy_step32(self.energy,float(self.backend.plant.applied_thrust[0]))

    def check(self,before,guard,proof):
        plant=self.backend.plant
        row=plant.state[0].tolist();position=row[:3]
        lower=[min(a,b) for a,b in zip(before,position)];upper=[max(a,b) for a,b in zip(before,position)]
        args=dict(origin=self.known_map.origin,shape=self.known_map.shape,resolution=self.known_map.resolution,
                  body_radius=self.radius,interstep_reserve=0)
        physical=free_grid_reserve(lower,upper,free_cells=self.physical_free,**args)
        known=free_grid_reserve(lower,upper,free_cells=self.known_map.free,**args)
        # Exact distance to the full-span planes for this left-of-slab room component.
        distances=[F.from_float(lower[k])-F(1,4) for k in range(3)]
        distances += [F(47,4)-F.from_float(upper[0]),F(47,4)-F.from_float(upper[1]),F(15,4)-F.from_float(upper[2]),F(9)-F.from_float(upper[0])]
        clearance=min(distances)-self.radius
        b=F.from_float(battery.authority32(self.energy,self.capacity))
        d=guard.hold-self.mass_gravity/b
        y=F.from_float(float(plant.applied_thrust[0]))-b*guard.hold
        queue=[tuple(F.from_float(v) for v in word[0].tolist()) for word in self.backend.history]
        membership=current_terminal_membership(p=row[:3],v=row[7:10],q=row[3:7],w=row[10:13],
            anchor=guard.anchor,hold_thrust=guard.hold,issued_queue=queue,trim_force_interval=(d,d),
            motor_residual_interval=(y,y),shapes=proof.shapes,ep=0,ev=0,ew=0,
            q_state_overapprox_error=0,battery_interval=(b,b))
        return {'physical_collision':not physical['reserved'],'unknown_entry':not known['reserved'],
                'physical_clearance_m':float(clearance),'actual_C_included':membership['current_set_included'],
                'actual_C_channel_maxima':{k:float(v['maximum']) for k,v in membership['channels'].items()},
                'queue_matches_public':queue==guard.queue,'position':position,
                'speed_mps':math.sqrt(sum(v*v for v in row[7:10])),'actual_energy_J':self.energy}


def run(*,cases=1,start_case=0,steps=100,output='smoke.json'):
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_flush_denormal(False)
    started=time.perf_counter();before_hashes=hashes()
    initial=json.loads((RESEARCH/'initial_contract_v1.json').read_text())
    capture=json.loads((RESEARCH/'public_trim_capture_v1.json').read_text())
    proof=ProofBundle(ROOT/'reports/phase_4_shield_artifacts/cpu_rpi_certificate_v1')
    known=WitnessMap.from_dict(json.loads((RESEARCH/'bounded_service_v1/known_free_map.json').read_text()),time_s=0.)
    renderer=AnalyticRenderer();profile=load_json(ROOT/'ares/data/configs/drone_sim_v2.json')
    records=[]
    for index in range(start_case,start_case+cases):
        admission_proof=InitialAdmission.load(proof,RESEARCH/'public_trim_history_v1.json',index)
        admission=admission_proof.row;mode,seed=capture['profile_modes_and_seeds'][index]
        sampled=sample_flight_profile(profile,seed,mode)
        private=PlantAndOracle(sampled,capture['simulation_truth_for_replay_only'],index,renderer,known)
        guard=ReanchoringShield(proof,known,hold_thrust=admission['hold_thrust'],
            trim_force_interval=admission['trim_force_interval'],motor_residual_interval=admission['motor_residual_interval'],
            issued_queue=admission['issued_queue'],deadline_tick=1000,service_lease_expires_tick=1000,
            initial_energy_upper_J=4338,arithmetic_backend_attestation=ATTESTATION,initial_admission=admission_proof)
        start_position=private.backend.plant.position[0].tolist()
        initial_anchor=list(admission['anchor']);goal=[a+d for a,d in zip(initial_anchor,(.3,.2,.1))]
        trace=[];latencies=[];ack_latencies=[];reasons=Counter();error=None;positive=accept=reject=fallback=0
        for tick in range(steps):
            if STOP_REQUESTED:
                error={'tick':tick,'type':'SweepCancelled','message':'other worker functional failure or external stop request'};break
            try:
                packet=public_initial_packet(admission) if tick==0 else private.packet(tick,index)
                base=guard.anchor if guard.anchor is not None else initial_anchor
                proposed=[float(a)+delta for a,delta in zip(base,(.0003,.0002,.0001))]
                decide_gc_before=gc_snapshot()
                t=time.perf_counter();decision=guard.decide(packet,tick=tick,proposed_anchor=proposed)
                decide_s=time.perf_counter()-t;decide_gc_after=gc_snapshot()
                latencies.append(decide_s);reasons[decision['reason']]+=1
                positive+=decision['conditional_motion_certificate'];accept+=decision['proposal_accepted'];reject+=not decision['proposal_accepted']
                fallback+=bool(decision.get('cached_backup_used',False))
                if decision['command'] is None:raise RuntimeError('guard returned no executable command: '+decision['reason'])
                command=decision['command']
                before,numeric,issued_words=private.step(command)
                ack_gc_before=gc_snapshot()
                t=time.perf_counter();guard.acknowledge_issued(issued_words)
                ack_s=time.perf_counter()-t;ack_gc_after=gc_snapshot()
                ack_latencies.append(ack_s)
                private.update_energy()
                oracle=private.check(before,guard,proof)
                trace.append({'tick':tick,'proposal':proposed,'accepted':decision['proposal_accepted'],
                    'cached_backup_used':decision.get('cached_backup_used',False),'anchor':decision['anchor'],
                    'command':command,'conditional_motion_certificate':decision['conditional_motion_certificate'],
                    'numeric_budget_applicable':numeric['conditional_numeric_budget_applicable'],
                    'guard_timing':{'decide_s':decide_s,'acknowledge_s':ack_s,'total_s':decide_s+ack_s,
                        'exceeds_10ms':decide_s+ack_s>.01,
                        'decide_gc_stats_delta':gc_delta(decide_gc_after,decide_gc_before),
                        'acknowledge_gc_stats_delta':gc_delta(ack_gc_after,ack_gc_before)},**oracle})
                if not oracle['actual_C_included'] or oracle['physical_collision'] or oracle['unknown_entry'] or not oracle['queue_matches_public']:
                    error={'tick':tick,'type':'FunctionalOracleFailure','message':'actual-C/collision/unknown/queue oracle failed'}
                    print('FUNCTIONAL_FAILURE '+json.dumps(error),flush=True);break
                if (tick+1)%100==0:
                    print(json.dumps({'case':index,'completed':tick+1,'positive':positive,'accepted':accept,
                        'cached':fallback,'actual_C_misses_so_far':sum(not r['actual_C_included'] for r in trace)}),flush=True)
            except Exception as exc:
                error={'tick':tick,'type':type(exc).__name__,'message':str(exc)}
                print('FUNCTIONAL_FAILURE '+json.dumps(error),flush=True);break
        last=private.backend.plant.position[0].tolist()
        membership_misses=[r['tick'] for r in trace if not r['actual_C_included']]
        collisions=[r['tick'] for r in trace if r['physical_collision']]
        unknown=[r['tick'] for r in trace if r['unknown_entry']]
        queue_misses=[r['tick'] for r in trace if not r['queue_matches_public']]
        distance=lambda a,b:math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
        record={'case':index,'profile_mode':mode,'profile_seed':seed,'requested_steps':steps,'completed_steps':len(trace),
            'valid_positive_certificate_count':positive,'anchor_accept_count':accept,'anchor_reject_count':reject,
            'cached_fallback_count':fallback,'decision_reasons':dict(reasons),'error':error,
            'actual_C_membership_misses':membership_misses,'physical_collision_ticks':collisions,
            'unknown_entry_ticks':unknown,'queue_mismatch_ticks':queue_misses,
            'max_speed_mps':max((r['speed_mps'] for r in trace),default=0),
            'min_physical_clearance_m':min((r['physical_clearance_m'] for r in trace),default=None),
            'displacement_xyz_m':[b-a for a,b in zip(start_position,last)],
            'waypoint_distance_progress_m':distance(start_position,goal)-distance(last,goal),
            'guard_decide_latency_s':{'count':len(latencies),'max':max(latencies,default=None),
                                    'mean':sum(latencies)/len(latencies) if latencies else None},
            'guard_total_latency_s':{'count':len(ack_latencies),
                'max':max((d+a for d,a in zip(latencies,ack_latencies)),default=None),
                'mean':sum(d+a for d,a in zip(latencies,ack_latencies))/len(ack_latencies) if ack_latencies else None,
                'exceeds_10ms_count':sum(d+a>.01 for d,a in zip(latencies,ack_latencies))},
            'latency_scope':'decide and acknowledge measured separately around plant step; total excludes supplier, plant, energy update and oracle',
            'delay_representation_binding':private.delay_binding,'backend_source_hashes':private.backend.source_hashes,'norm_witness':private.backend.plant.norm_witness.snapshot(),
            'functional_behavior_passed':not(error or membership_misses or collisions or unknown or queue_misses) and len(trace)==steps,
            'timing_gate_passed':len(ack_latencies)==steps and all(d+a<=.01 for d,a in zip(latencies,ack_latencies)),
            'passed':not(error or membership_misses or collisions or unknown or queue_misses) and len(trace)==steps
                and len(ack_latencies)==steps and all(d+a<=.01 for d,a in zip(latencies,ack_latencies)),
            'trace':trace}
        records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('trace','backend_source_hashes','norm_witness')}),flush=True)
    after_hashes=hashes()
    report={'version':'coupled_reanchoring_v1','conditional_software_SIL_only':True,'flight_certificate':False,
        'hardware_or_Gaussian_runtime_certified':False,'all_cases_retained':True,'cases':records,
        'process_cpu_affinity':sorted(os.sched_getaffinity(0)),
        'gc_measurement':{'enabled':gc.isenabled(),'thresholds':gc.get_threshold(),
            'delta_columns':['collections','collected','uncollectable'],
            'delta_rows':'generations0,1,2; snapshots sampled immediately outside timed call'},
        'source_hashes_stable':before_hashes==after_hashes,'run_valid':before_hashes==after_hashes,'source_sha256':before_hashes,
        'invalid_reason':None if before_hashes==after_hashes else 'executing source/proof/input hashes changed during run',
        'proof_sha256':proof.source_sha256,'elapsed_s':time.perf_counter()-started,
        'all_requested_cases_passed':all(r['passed'] for r in records) and before_hashes==after_hashes}
    target=HERE/output;target.write_text(json.dumps(report,indent=2,allow_nan=False,default=str)+'\n')
    print('SAVED',str(target),'all_cases_passed',report['all_requested_cases_passed'],flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cases',type=int,default=1);parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--output',default='smoke.json');parser.add_argument('--start-case',type=int,default=0);a=parser.parse_args()
    if not 1<=a.cases<=8 or not 0<=a.start_case<8 or a.start_case+a.cases>8 or not 1<=a.steps<=1000:parser.error('cases1..8,steps1..1000 required')
    run(cases=a.cases,start_case=a.start_case,steps=a.steps,output=a.output)
