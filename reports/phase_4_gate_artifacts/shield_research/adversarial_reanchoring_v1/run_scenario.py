"""New-world/unknown-frontier adversaries for the frozen public-only guard."""
import argparse
from collections import Counter
from fractions import Fraction as F
import gc,hashlib,json,math,os,signal,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parent;ROOT=RESEARCH.parents[2]
sys.path.insert(0,str(RESEARCH/'coupled_reanchoring_v1'))
import run_coupled as common
import torch
from bounded_service import WitnessMap,BoundedSupplier,preflight_scan
from sensor_renderer import AnalyticRenderer,TruthState,TestOracle
from reanchoring_shield_v1 import ProofBundle,InitialAdmission,ReanchoringShield

STOP=False
def stop_requested(signum,frame):
    global STOP
    STOP=True
signal.signal(signal.SIGTERM,stop_requested)

class PlaneRenderer(AnalyticRenderer):
    def __init__(self,corner=False):self.corner=corner
    def occupied(self,cell):
        x,y,z=cell
        return (not(0<=x<48 and 0<=y<48 and 0<=z<16) or x in (0,47,26)
                or y in ((0,47,26) if self.corner else (0,47)) or z in (0,15))
    def _boxes(self):
        walls=super()._boxes()[:6]
        return walls+(((6.5,0,0),(6.75,12,4)),)+((((0,6.5,0),(12,6.75,4)),) if self.corner else ())

class ScenarioPlantOracle(common.PlantAndOracle):
    """Only test/renderer boundary sees changed physical geometry or truth."""
    def check(self,before,guard,proof):
        result=super().check(before,guard,proof)
        position=self.backend.plant.position[0].tolist()
        lower=[min(a,b) for a,b in zip(before,position)];upper=[max(a,b) for a,b in zip(before,position)]
        if isinstance(self.renderer,PlaneRenderer):
            clearance=min(F.from_float(result['physical_clearance_m']),F(13,2)-F.from_float(upper[0])-self.radius)
            if self.renderer.corner:clearance=min(clearance,F(13,2)-F.from_float(upper[1])-self.radius)
            result['physical_clearance_m']=float(clearance)
        return result

def hashes():
    result=common.hashes()
    for path in sorted(HERE.glob('*.py')):result[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result

def make_map(scenario):
    if scenario=='unknown_plane':
        renderer=AnalyticRenderer()
        known=WitnessMap.from_dict(json.loads((RESEARCH/'bounded_service_v1/known_free_map.json').read_text()),time_s=0.)
        known.free={cell:entry for cell,entry in known.free.items() if cell[0]<26}
        provenance='existing public ray map with all free cells x>=26 forgotten; physical world unchanged'
    else:
        renderer=PlaneRenderer(corner=scenario=='occupied_corner')
        # Fresh first-hit rays in the changed physical world; no old depth reuse.
        truth=TruthState((6.125,6.125,2.125),(.125,-.25,.0625),angular_velocity=(.1,-.2,.3))
        supplier=BoundedSupplier(renderer,truth,seed=71)
        state=supplier.capture(capture_time_s=0.,expires_s=30.)
        known=preflight_scan(supplier,state,time_s=0.)
        provenance='new bounded first-hit ray scan in changed physical world'
    valid=TestOracle.free_map_valid(renderer,known.free)
    if not valid:raise RuntimeError('independent oracle found occupied voxel labeled free')
    return renderer,known,{'provenance':provenance,'known_free_voxels':len(known.free),
        'oracle_all_free_labels_physically_free':valid,'all_free_labels_have_ray_witness':True,
        'physical_slabs':(['x=9..9.25'] if scenario=='unknown_plane' else ['x=6.5..6.75']+(['y=6.5..6.75'] if scenario=='occupied_corner' else []))}

def make_guard(proof,known,admission,**changes):
    row=admission.row
    fields=dict(hold_thrust=row['hold_thrust'],trim_force_interval=row['trim_force_interval'],
        motor_residual_interval=row['motor_residual_interval'],issued_queue=row['issued_queue'],
        deadline_tick=1000,service_lease_expires_tick=1000,initial_energy_upper_J=4338,
        arithmetic_backend_attestation=common.ATTESTATION,initial_admission=admission)
    fields.update(changes)
    return ReanchoringShield(proof,known,**fields)

def run(scenario,steps=1000,output=None):
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.set_flush_denormal(False)
    before_hashes=hashes();started=time.perf_counter()
    proof=ProofBundle(ROOT/'reports/phase_4_shield_artifacts/cpu_rpi_certificate_v1')
    admission=InitialAdmission.load(proof,RESEARCH/'public_trim_history_v1.json',0)
    renderer,known,map_audit=make_map(scenario)
    (HERE/f'{scenario}_ray_witness_map.json').write_text(json.dumps(known.to_dict(),indent=2)+'\n')
    capture=json.loads((RESEARCH/'public_trim_capture_v1.json').read_text())
    mode,seed=capture['profile_modes_and_seeds'][0]
    profile=common.sample_flight_profile(common.load_json(ROOT/'ares/data/configs/drone_sim_v2.json'),seed,mode)
    private=ScenarioPlantOracle(profile,capture['simulation_truth_for_replay_only'],0,renderer,known)
    guard=make_guard(proof,known,admission)
    initial_anchor=list(admission.row['anchor']);goal=[a+d for a,d in zip(initial_anchor,(.3,.2,.1))]
    start=private.backend.plant.position[0].tolist();trace=[];reasons=Counter();error=None
    for tick in range(steps):
        if STOP:error={'tick':tick,'type':'SweepCancelled'};break
        try:
            packet=common.public_initial_packet(admission.row) if tick==0 else private.packet(tick,0)
            anchor=guard.anchor if guard.anchor is not None else initial_anchor
            proposal=[float(a)+d for a,d in zip(anchor,(.0003,.0002,.0001))]
            gc0=common.gc_snapshot();t=time.perf_counter();decision=guard.decide(packet,tick=tick,proposed_anchor=proposal)
            decide_s=time.perf_counter()-t;gc1=common.gc_snapshot();reasons[decision['reason']]+=1
            if decision['command'] is None:raise RuntimeError('no executable backup: '+decision['reason'])
            p0,numeric,issued=private.step(decision['command'])
            ga=common.gc_snapshot();t=time.perf_counter();guard.acknowledge_issued(issued)
            ack_s=time.perf_counter()-t;gb=common.gc_snapshot()
            private.update_energy();oracle=private.check(p0,guard,proof)
            trace.append({'tick':tick,'proposal':proposal,'reason':decision['reason'],'anchor':decision['anchor'],
                'accepted':decision['proposal_accepted'],'cached_backup_used':decision.get('cached_backup_used',False),
                'conditional_motion_certificate':decision['conditional_motion_certificate'],'command':issued,
                'membership_maxima':decision['membership_maxima'],
                'guard_timing':{'decide_s':decide_s,'acknowledge_s':ack_s,'total_s':decide_s+ack_s,
                    'exceeds_10ms':decide_s+ack_s>.01,'decide_gc_stats_delta':common.gc_delta(gc1,gc0),
                    'acknowledge_gc_stats_delta':common.gc_delta(gb,ga)},**oracle})
            if not oracle['actual_C_included'] or not oracle['queue_matches_public'] or oracle['physical_collision'] or oracle['unknown_entry']:
                raise RuntimeError('independent actual-C/queue/collision/unknown oracle failed')
            if (tick+1)%100==0:print(json.dumps({'scenario':scenario,'steps':tick+1,'reasons':dict(reasons)}),flush=True)
        except Exception as exc:
            error={'tick':tick,'type':type(exc).__name__,'message':str(exc)}
            print('FUNCTIONAL_FAILURE '+json.dumps(error),flush=True);break
    end=private.backend.plant.position[0].tolist();distance=lambda a,b:math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
    timing=[row['guard_timing']['total_s'] for row in trace]
    misses={name:[row['tick'] for row in trace if predicate(row)] for name,predicate in (
        ('actual_C_membership_misses',lambda row:not row['actual_C_included']),
        ('queue_mismatches',lambda row:not row['queue_matches_public']),
        ('physical_collisions',lambda row:row['physical_collision']),('unknown_entries',lambda row:row['unknown_entry']))}
    geometry=[row for row in trace if row['reason']=='proposal_reserve_not_known_free']
    functional=not error and len(trace)==steps and not any(misses.values()) and bool(geometry)
    progress=distance(start,goal)-distance(end,goal)
    after_hashes=hashes();stable=before_hashes==after_hashes
    report={'version':'adversarial_reanchoring_v1','scenario':scenario,'requested_steps':steps,'completed_steps':len(trace),
        'conditional_software_SIL_only':True,'flight_certificate':False,'hardware_or_Gaussian_runtime_certified':False,
        'map_audit':map_audit,'error':error,'decision_reasons':dict(reasons),**misses,
        'positive_conditional_guard_decisions':sum(row['conditional_motion_certificate'] for row in trace),
        'anchor_accepts':sum(row['accepted'] for row in trace),'cached_fallbacks':sum(row['cached_backup_used'] for row in trace),
        'reserve_refusals_with_candidate_C_included':sum(all(value<=1 for value in row['membership_maxima'].values()) for row in geometry),
        'first_reserve_refusal':geometry[0] if geometry else None,'max_speed_mps':max((r['speed_mps'] for r in trace),default=0.),
        'min_physical_clearance_m':min((r['physical_clearance_m'] for r in trace),default=None),
        'displacement_xyz_m':[b-a for a,b in zip(start,end)],'waypoint_distance_progress_m':progress,
        'guard_timing_s':{'count':len(timing),'max':max(timing,default=None),'mean':sum(timing)/len(timing) if timing else None,
                          'exceeds_10ms_count':sum(t>.01 for t in timing)},
        'timing_scope':'decide+ack; supplier, plant, energy update and independent oracle excluded',
        'functional_behavior_passed':functional and progress>0,'timing_gate_passed':len(trace)==steps and all(t<=.01 for t in timing),
        'source_hashes_stable':stable,'source_sha256':before_hashes,'process_cpu_affinity':sorted(os.sched_getaffinity(0)),
        'backend_source_hashes':private.backend.source_hashes,'norm_witness':private.backend.plant.norm_witness.snapshot(),
        'delay_representation_binding':private.delay_binding,'elapsed_s':time.perf_counter()-started,'trace':trace}
    report['passed']=report['functional_behavior_passed'] and report['timing_gate_passed'] and stable
    target=HERE/(output or f'{scenario}.json');target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('trace','source_sha256','backend_source_hashes','first_reserve_refusal')}),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scenario',choices=('unknown_plane','occupied_plane','occupied_corner'),required=True)
    parser.add_argument('--steps',type=int,default=1000);parser.add_argument('--output');a=parser.parse_args()
    if not 1<=a.steps<=1000:parser.error('steps1..1000 required')
    run(a.scenario,a.steps,a.output)
