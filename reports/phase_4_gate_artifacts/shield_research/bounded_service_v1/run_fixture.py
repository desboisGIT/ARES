"""Generate reusable bounded-service packets/map and run local verification.

No ares sources are written. The full source tree is hashed before and after.
"""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import unittest

from ares.contracts import load_json,validate_profile
from bounded_service import BoundedSupplier,preflight_scan
from sensor_renderer import AnalyticRenderer,TruthState,TestOracle

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def source_hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'ares').rglob('*.py'))}


def run(seed=71):
    before=source_hashes()
    fixture_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))}
    profile=load_json(ROOT/'ares/data/configs/drone_sim_v2.json');validate_profile(profile)
    renderer=AnalyticRenderer()
    truth=TruthState((6.125,6.125,2.125),(.125,-.25,.0625),angular_velocity=(.1,-.2,.3))
    supplier=BoundedSupplier(renderer,truth,seed=seed)
    packet=supplier.capture(capture_time_s=0.,expires_s=30.)
    observed=preflight_scan(supplier,packet,time_s=.1)
    reservation=observed.reservation(packet.p,profile['parameters']['body_radius_m']['range'][1]+.15+.005,time_s=.1)
    log=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_bounded_service.py')
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    after=source_hashes()
    if before!=after:raise RuntimeError('ares source bytes changed during fixture generation')
    if not result.wasSuccessful():
        (HERE/'tests.log').write_text(log.getvalue())
        raise RuntimeError('fixture verification failed; see tests.log')
    profile_bytes=(ROOT/'ares/data/configs/drone_sim_v2.json').read_bytes()
    report={'version':'bounded_service_v1','conditional_fixture_only':True,'flight_certificate':False,
            'seed':seed,'capture_time_s':0.,'expires_s':30.,'query_time_s':.1,
            'state_packet':asdict(packet),'state_boxes':asdict(packet.boxes(time_s=.1)),
            'body_rate_box':packet.body_rate_box(time_s=.1),
            'map':{'shape':observed.shape,'resolution':observed.resolution,'known_free_voxel_count':len(observed.free),
                   'rejected_target_count':len(observed.rejections),'unknown_target_slab_x_m':[7.,7.25],
                   'all_free_labels_have_individual_robust_ray_witness':True,
                   'oracle_no_occupied_voxel_labeled_free':TestOracle.free_map_valid(renderer,observed.free)},
            'static_capture_epoch_reservation':reservation,'reservation_radius_m':.17+.15+.005,
            'state_bounds':{'position_euclidean_m':packet.contract.position_error_m,'velocity_euclidean_mps':packet.contract.velocity_error_mps,
                            'angular_velocity_euclidean_radps':packet.angular_velocity_error_radps,'quaternion_measurement':packet.orientation_contract,
                            'orientation_capture_epoch_s':packet.contract.capture_time_s},
            'proof_bound_representation':{'position_exact':'1/200','velocity_exact':'1/100','angular_velocity_exact':'1/1000',
                'advertised_radius_rule':'nextafter(decimal_binary64,0); exact represented value <= rational proof bound',
                'noise_generated_using_advertised_radii':True,'epsilon_relaxation':False},
            'ray_bounds':{'world_direction_euclidean':.001,'first_hit_range_parameter':.01},
            'full_profile_configuration_sha256':hashlib.sha256(profile_bytes).hexdigest(),
            'tests':{'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
                     'state_ray_depth_signed_corner_combinations':1024,'rigid_body_initial_variations':8,
                     'closed_left_face_contact_rejections':['unknown','occupied','outside_support'],
                     'malformed_state_packet_and_grid_metadata_reject':True,
                     'unsupported_orientation_tag_construct_and_reload_reject':True},
            'reservation_cell_enumeration':'lower=ceil((p-r-origin)/resolution)-1; upper=floor((p+r-origin)/resolution); exact closed box distance squared <= radius squared',
            'quaternion_norm_validation':{'unit_roundoff':2**-24,'norm_shell_unit_roundoffs':8,
                                          'positive_scalar_sign_required':False},
            'fixture_source_sha256':fixture_hashes,'ares_source_file_count':len(before),'ares_source_bytes_unchanged':True,'ares_source_sha256':before,
            'limitations':['Synthetic enforced bounds apply only inside this SIL supplier; not real sensor guarantees.',
               'p/v/body-rate state boxes describe capture epoch only, not present-time reachability.',
               'Exact quaternion feedback is separate from any future nonzero quaternion initial-set enclosure.',
               'State packet has no mass/gravity/inertia/motor/energy samples; body-rate bound is an explicit fixture extra.',
               'Static homogeneous voxels and bounded first-hit semantics are enforced by the analytic renderer.',
               'No full 13D shield, verified backup, motor-state uncertainty or terminal invariant certificate is implemented.',
               'Supplier/renderer separation is an API/test boundary, not a malicious-code security sandbox.',
               'The Gaussian benchmark environment/configurations are untouched.']}
    (HERE/'fixture_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (HERE/'known_free_map.json').write_text(json.dumps(observed.to_dict(),indent=2,allow_nan=False)+'\n')
    (HERE/'full_profile_config.json').write_bytes(profile_bytes)
    (HERE/'state_packet.json').write_text(json.dumps(asdict(packet),indent=2,allow_nan=False)+'\n')
    (HERE/'tests.log').write_text(log.getvalue())
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,default=71)
    args=parser.parse_args();r=run(args.seed)
    print(json.dumps({'flight_certificate':False,'known_free_voxels':r['map']['known_free_voxel_count'],
                      'reservation':r['static_capture_epoch_reservation'],'tests':r['tests'],
                      'ares_source_bytes_unchanged':r['ares_source_bytes_unchanged']},indent=2))
