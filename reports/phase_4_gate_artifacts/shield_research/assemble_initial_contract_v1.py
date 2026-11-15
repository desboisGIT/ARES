"""Build SHA-bound initial-set admissions from PUBLIC bounded SIL histories.

The replay file's separate simulation_truth_for_replay_only block is never read.
The global profile-set constants are public model bounds, not sampled parameters.
This proves admission under the hard sensor/backend construction, not safety of
the earlier bootstrap flight or availability of a physical sensor supplier.
"""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

from initial_terminal_membership_v1 import initial_terminal_membership
from reanchoring_shield_v1 import ProofBundle
from trim_history_v2 import rk4_trim_bound
import verified_battery_v1 as battery
import zero_rate_history_v1 as domain

HERE = Path(__file__).resolve().parent


def encode(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


def assemble(proof, capture):
    if (not capture['sensor_boundary_exact_squared_error_checks_pass']
            or capture['normalization_backend'] != 'explicit_norm_verified_sqrt_v1'
            or not capture['normalization_witness']['every_sqrt_point_exact_bin_verified']
            or capture['source_sha256'] != hashlib.sha256((HERE/'public_trim_capture_v1.py').read_bytes()).hexdigest()):
        raise ValueError('enforced bounded-source/verified-backend capture required')
    history = capture['measured_history']
    d = domain.rational_domain()
    rates, broad_motor = domain.rate_stage_lemma(d), domain.broad_motor_arithmetic()
    p = proof.certificate['physical_parameter_intervals_outward']
    mass_lower, mass_upper = map(F, p['mass'])
    drag_upper, lag_upper = F(p['drag'][1]), F(p['motor_lag'][1])
    command_max = F.from_float(capture['maximum_issued_thrust_N'])
    motor_upper = (max(proof.mg_upper*(1+battery.U)+battery.ETA, command_max)
                   +broad_motor['standalone_settle_motor_error_per_H']
                   /(1-broad_motor['macro_decay_upper']))
    if motor_upper >= F(13, 2):
        raise ValueError('public command/motor bound does not support power lemma')
    energy_upper = capture['full_rigid_body_steps_per_case']*battery.updater_bound()['energy_jump_upper_J']
    if not energy_upper < 8000:
        raise ValueError('public bootstrap energy budget cannot support new horizon')
    rows = []
    for record in capture['cases']:
        case, hold = record['case'], record['public_hold_thrust_N']
        check = domain.history_case(history, hold, case, d, rates, broad_motor)
        trim = rk4_trim_bound(
            times=[F(row['time']) for row in history],
            measured_vertical_velocity=[row['v'][case][2] for row in history],
            velocity_error=F(1, 100), control_period=F(1, 100), physics_period=F(1, 500),
            hold_thrust=hold, public_force_upper=F(13, 2), mass_lower=mass_lower,
            drag_upper=drag_upper, authority_drop_per_period_upper=proof.authority_drop,
            motor_lag_upper=lag_upper, transport_upper=F(2000001, 100000000),
            constant_command_settle=F(7, 10), stage_up_axis_error=F(1, 5000),
            stage_angular_rate_upper=1, velocity_roundoff_per_period=F(53, 100000000),
            motor_roundoff_per_period=broad_motor['standalone_settle_motor_error_per_H'])
        def upper_grid(value):
            scaled = value*2**40
            return F(-((-scaled).numerator//(-scaled).denominator), 2**40)
        # Exact derivations may have thousands of numerator/denominator digits.
        # Outward dyadic admission endpoints keep runtime decisions bounded in
        # size; the initial inclusion below verifies the enlarged family.
        force = upper_grid(mass_upper/F(19, 20)*trim['conditional_trim_acceleration_bound'])
        motor = upper_grid(trim['motor_tracking_force_bound'])
        final = {name: history[-1][name][case] for name in ('p', 'v', 'q', 'w')}
        membership = initial_terminal_membership(**final, anchor=final['p'],
            correlated_trim_force_upper=force, motor_tracking_error=motor,
            shapes=proof.shapes)
        rows.append({'case': case,
            'conditional_initial_admission_proved': bool(check['conditional_history_inputs_and_stages_pass']
                                                        and membership['initial_set_included']),
            'public_final_packet': final, 'anchor': final['p'], 'hold_thrust': hold,
            'trim_force_interval': (-force, force), 'motor_residual_interval': (-motor, motor),
            'issued_queue': [(hold, 0., 0., 0.)]*2,
            'public_initial_energy_upper_J': energy_upper,
            'domain_check': check, 'trim_lemma': trim, 'membership': membership,
            'exact_current_orientation_required': True,
            'hard_service_construction_enforced_in_this_SIL_capture': True,
            'flight_certificate': False})
    return {'schema_version': 'ares.public_history_initial_admission.v1',
        'all_cases_retained': True, 'case_count': len(rows),
        'all_initial_admissions_proved': all(row['conditional_initial_admission_proved'] for row in rows),
        'cases': rows, 'public_motor_upper_N': motor_upper,
        'public_bootstrap_energy_upper_J': energy_upper,
        'uses_sampled_airframe_parameters': False, 'uses_motor_or_energy_truth': False,
        'bootstrap_safety_proved': False, 'physical_supplier_available': False,
        'mathematical_proof_sha256': proof.source_sha256, 'flight_certificate': False}


if __name__ == '__main__':
    capture_path = HERE/'public_trim_capture_v1.json'
    capture = json.loads(capture_path.read_text())
    public = {key: capture[key] for key in (
        'cases', 'measured_history', 'full_rigid_body_steps_per_case',
        'maximum_issued_thrust_N', 'source_sha256',
        'sensor_boundary_exact_squared_error_checks_pass',
        'normalization_backend', 'normalization_witness')}
    public_path = HERE/'public_trim_history_v1.json'
    public_path.write_text(json.dumps(public, indent=2)+'\n')
    result = assemble(ProofBundle(HERE.parents[1]/'phase_4_shield_artifacts/cpu_rpi_certificate_v1'), public)
    result['capture_sha256'] = hashlib.sha256(capture_path.read_bytes()).hexdigest()
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['public_history_sha256'] = hashlib.sha256(public_path.read_bytes()).hexdigest()
    HERE.joinpath('initial_contract_v1.json').write_text(json.dumps(encode(result), indent=2)+'\n')
    print(json.dumps({'all_initial_admissions_proved': result['all_initial_admissions_proved'],
        'cases': [{'case': row['case'], 'passed': row['conditional_initial_admission_proved'],
            'maximum_membership': {name: float(c['maximum']) for name, c in row['membership']['channels'].items()}}
            for row in result['cases']],
        'public_bootstrap_energy_upper_J': float(result['public_bootstrap_energy_upper_J'])}, indent=2))
