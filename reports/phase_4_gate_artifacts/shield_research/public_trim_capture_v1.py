"""Full-quadrotor public-data trim capture experiment; not a certificate.

Simulated sensor construction may read plant state. The bootstrap controller
receives only its bounded p/v packet and the current simulation's exact-q
abstraction. It never receives sampled mass, gravity, motor thrust or energy.
Initial motor thrust is not used as a hover-thrust oracle.
"""
from collections import deque
from fractions import Fraction as F
import importlib.util
import hashlib
import json
from pathlib import Path
import time

import torch

from ares.flight_control import acceleration_to_action
from ares.flight_randomization import sample_flight_profile
from ares.phase2 import default_profile

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE/(name+'.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sensor_packet(adapter, generator):
    # ||error||2 <= sqrt(3)*r/3 < r, before a separately bounded Float32
    # measurement-addition error. This bootstrap is an experiment, not the
    # final supplier contract or its floating-point input proof.
    ep = (torch.rand(adapter.position.shape, generator=generator)*2-1)*(.005/3)
    ev = (torch.rand(adapter.velocity.shape, generator=generator)*2-1)*(.01/3)
    ew = (torch.rand(adapter.angular_velocity.shape, generator=generator)*2-1)*(.001/3)
    packet = {'p': adapter.position.detach().clone()+ep,
              'v': adapter.velocity.detach().clone()+ev,
              'q': adapter.quaternion.detach().clone(),
              'w': adapter.angular_velocity.detach().clone()+ew}
    # Truth remains inside this simulated sensor boundary. Exact checks enforce
    # the returned representation's hard error, including Float32 addition.
    for name, actual, radius in (('p', adapter.position, F(1, 200)),
                                 ('v', adapter.velocity, F(1, 100)),
                                 ('w', adapter.angular_velocity, F(1, 1000))):
        for source, measured in zip(actual.tolist(), packet[name].tolist()):
            error2 = sum((F.from_float(a)-F.from_float(b))**2 for a, b in zip(source, measured))
            if error2 > radius*radius:
                raise RuntimeError('retained sensor construction failure: '+name)
    return packet


def run():
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    generator = torch.Generator().manual_seed(19347)
    profiles = [sample_flight_profile(default_profile(), 9100, mode)
                for mode in ('low', 'high')]
    profiles += [sample_flight_profile(default_profile(), 9101+i) for i in range(6)]
    n = len(profiles)
    Adapter = module('verified_norm_backend_v1/backend').make_adapter_class()
    battery = module('verified_battery_v1')
    adapter = Adapter(profiles, dt=.01)
    adapter.reset(position=[6., 6., 2.], velocity=[.015, -.01, .005],
                  quaternion=[.005, -.004, .003, 1.],
                  angular_velocity=[.005, -.004, .003])
    anchor = torch.tensor([[6., 6., 2.]]).expand(n, -1)
    hold = torch.full((n,), 4.903325)  # nominal, independent of every plant
    history = deque([torch.tensor([[4.903325, 0., 0., 0.]]).expand(n, -1).clone()
                     for _ in range(3)], maxlen=3)
    # Only the simulated plant's transport/battery use the sampled parameters.
    delay = torch.tensor([p['parameters']['compute_latency_s']['nominal']/.01
                          for p in profiles], dtype=torch.float32)
    low, high = delay.floor().long(), delay.ceil().long()
    fraction = (delay-low)[:, None]
    capacity = [float(x) for x in adapter.parameters['battery_energy_j']]
    energy = [0.]*n
    index = torch.arange(n)
    measurements = []
    issued_thrust_window = deque(maxlen=100)
    maximum_command = 0.
    minimum_body_clearance = 100.
    settled_hold = None
    start = time.monotonic()
    with torch.no_grad():
        for step in range(3271):
            packet = sensor_packet(adapter, generator)
            if step < 3000:
                hold = (hold-.5*.5*(packet['p'][:, 2]-2.)*.01).clamp(3.8, 6.2)
                acceleration = -1.5*(packet['p']-anchor)-2.5*packet['v']
                bias = torch.zeros_like(acceleration)
                bias[:, 2] = 9.80665-hold/.5
                packet['controller_bias'] = bias
                packet['gaze'] = packet['p']+torch.tensor([1., 0., 0.])
                command = acceleration_to_action(packet, acceleration)
                issued_thrust_window.append(command[:, 0].clone())
            else:
                if settled_hold is None:
                    # The public integrator alone omits steady PD correction.
                    # Capture the recent ISSUED command average, without an
                    # actuator/mass/energy read. The history lemma may reject
                    # this guess; averaging itself establishes no trim bound.
                    settled_hold = torch.stack(tuple(issued_thrust_window)).mean(0)
                command = torch.cat((settled_hold[:, None], torch.zeros(n, 3)), -1)
                if step >= 3070:
                    measurements.append({'time': str(F(step-3070, 100)),
                        'p': packet['p'].tolist(), 'v': packet['v'].tolist(),
                        'q': packet['q'].tolist(), 'w': packet['w'].tolist()})
            maximum_command = max(maximum_command, float(command[:, 0].max()))
            if maximum_command > 6.5:
                raise RuntimeError('retained bootstrap failure: public force bound exceeded')
            if step == 3270:
                break
            history.appendleft(command)
            stack = torch.stack(tuple(history))
            physical = stack[low, index]*(1-fraction)+stack[high, index]*fraction
            authority = torch.tensor([battery.authority32(e, c) for e, c in zip(energy, capacity)])
            physical = torch.cat((physical[:, :1]*authority[:, None], physical[:, 1:]), -1)
            adapter.step(physical)
            energy = [battery.energy_step32(e, float(f)) for e, f in zip(energy, adapter.applied_thrust)]
            p = adapter.position
            # Evaluator only: analytic room [0,12]x[0,12]x[0,4], max body .17.
            clearance = torch.cat((p, torch.tensor([12., 12., 4.])-p), -1).min()-.17
            minimum_body_clearance = min(minimum_body_clearance, float(clearance))
    trim = module('trim_history_v2')
    records = []
    for case in range(n):
        result = trim.rk4_trim_bound(
            times=[F(row['time']) for row in measurements],
            measured_vertical_velocity=[row['v'][case][2] for row in measurements],
            velocity_error=F(1, 100), control_period=F(1, 100), physics_period=F(1, 500),
            hold_thrust=float(settled_hold[case]), public_force_upper=F(13, 2),
            mass_lower=F(44999998, 100000000), drag_upper=F(20000001, 100000000),
            authority_drop_per_period_upper=F(64, 10000000), motor_lag_upper=F(4000001, 100000000),
            transport_upper=F(2000001, 100000000), constant_command_settle=F(7, 10),
            stage_up_axis_error=F(1, 5000), stage_angular_rate_upper=1,
            velocity_roundoff_per_period=F(53, 100000000), motor_roundoff_per_period=F(1, 50000))
        delta = result['conditional_trim_acceleration_bound']
        records.append({'case': case, 'public_hold_thrust_N': float(settled_hold[case]),
            'public_final_packet': measurements[-1]['p'][case],
            'public_trim_acceleration_upper': float(delta),
            'public_equilibrium_offset_upper_m': float(delta*F(55000006, 100000000)
                *result['position_offset_coefficient_per_mass_upper']),
            'independent_stage_and_numeric_hypotheses_established': False})
    report = {'schema_version': 'ares.public_trim_capture_experiment.v1',
        'cases': records, 'full_rigid_body_steps_per_case': 3270,
        'bootstrap_controller_reads_sampled_parameters': False,
        'bootstrap_controller_reads_motor_or_battery_truth': False,
        'initial_hover_command': 4.903325,
        'maximum_issued_thrust_N': maximum_command,
        'minimum_physical_body_clearance_m': minimum_body_clearance,
        'physical_collisions': int(minimum_body_clearance <= 0),
        'maximum_internal_energy_J': max(energy),
        'measured_history': measurements,
        'profile_modes_and_seeds': [('low', 9100), ('high', 9100)]
                                  + [('random', 9101+i) for i in range(6)],
        'simulation_truth_for_replay_only': {
            'final_state': adapter.state.tolist(),
            'final_motor_thrust': adapter.applied_thrust.tolist(),
            'last_two_issued_commands': [entry.tolist() for entry in list(history)[:2]],
            'energy_words_J': energy,
            'never_pass_to_guard_or_controller': True},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'settling_motor_numerical_budget_per_period_N': '1/50000',
        'history_velocity_numerical_budget_per_period_mps': '53/100000000',
        'sensor_boundary_exact_squared_error_checks_pass': True,
        'normalization_backend': 'explicit_norm_verified_sqrt_v1',
        'normalization_witness': adapter.norm_witness.snapshot(),
        'flight_certificate': False, 'bootstrap_safety_proved': False,
        'elapsed_s': time.monotonic()-start}
    HERE.joinpath('public_trim_capture_v1.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'measured_history'}, indent=2))


if __name__ == '__main__':
    run()
