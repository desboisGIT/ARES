"""Exact conditional trim lemma for the sampled RK4 model, research only.

Unlike the earlier continuous-time lemma, this uses the RK4 velocity balance
directly. Every number used in a decision is rational. Numerical budgets and
stage-domain bounds are explicit hypotheses, never inferred from measurements.
No sampled mass, gravity, motor state or simulator state enters the function.
"""
from fractions import Fraction as F
import math


def rational(value):
    if isinstance(value, F):
        return value
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError('finite real or rational input required')
    return F.from_float(float(value))


def decay_upper(value):
    """exp(-x) <= reciprocal of a finite positive exponential series."""
    if value < 0:
        raise ValueError('nonnegative decay argument required')
    term = total = F(1)
    for k in range(1, 65):
        term *= value/k
        total += term
    return 1/total


def rk4_trim_bound(*, times, measured_vertical_velocity, velocity_error,
                   control_period, physics_period, hold_thrust,
                   public_force_upper, mass_lower, drag_upper,
                   authority_drop_per_period_upper, motor_lag_upper,
                   transport_upper, constant_command_settle,
                   stage_up_axis_error, stage_angular_rate_upper,
                   velocity_roundoff_per_period,
                   motor_roundoff_per_period,
                   nominal_mass=F(1, 2), position_gain=F(3, 2),
                   battery_lower=F(19, 20)):
    """Return a sufficient |b_end*F_hold/m-g| bound for the discrete plant.

    Required independently established hypotheses:
    - All issued thrust commands equal hold_thrust throughout settle + history;
      the positive motor recurrence, bounded delay and battery authority apply.
    - force_upper bounds initial motor thrust AND every command in that window.
    - Packet errors are deterministic, at the stated current-time macro grid.
    - Every real RK4 stage has |omega| <= stage_angular_rate_upper and
      |R_homogeneous(q).zz - 1| <= stage_up_axis_error. Reference quaternions
      are unit at each substep start, as in the analytic projective model.
    - Authority is nonincreasing with each macro decrement at most the supplied
      bound. The two roundoff bounds enclose the ACTUAL CPU macro transition.

    The output validates algebra under those hypotheses, not their suppliers.
    """
    t = list(map(rational, times))
    v = list(map(rational, measured_vertical_velocity))
    if len(t) != len(v) or len(t) < 2 or any(b <= a for a, b in zip(t, t[1:])):
        raise ValueError('strictly ordered complete packet history required')
    (ev, H, h, hold, force, mass, drag, db, lag, delay, settle,
     up, omega, epsv, epsf, m0, kp, bmin) = map(rational, (
        velocity_error, control_period, physics_period, hold_thrust,
        public_force_upper, mass_lower, drag_upper,
        authority_drop_per_period_upper, motor_lag_upper,
        transport_upper, constant_command_settle, stage_up_axis_error,
        stage_angular_rate_upper, velocity_roundoff_per_period,
        motor_roundoff_per_period, nominal_mass, position_gain, battery_lower))
    if (min(H, h, hold, force, mass, lag, m0, kp, bmin) <= 0
            or min(ev, drag, db, delay, up, omega, epsv, epsf) < 0
            or hold > force or settle < delay+H or bmin > 1 or db > 1):
        raise ValueError('invalid physical/history hypothesis')
    if (H/h).denominator != 1 or any(((x-t[0])/H).denominator != 1 for x in t):
        raise ValueError('packet and substep timestamps must lie on the exact macro grid')
    if h*omega >= 1:
        raise ValueError('stage quaternion norm enclosure unavailable')
    duration = t[-1]-t[0]
    gap = max(b-a for a, b in zip(t, t[1:]))
    # Every RK4 quaternion stage is a degree <=3 polynomial of the held
    # quaternion differential equation. Bounding its terms by the geometric
    # series gives ||q_stage|| <= 1/(1-h*omega). The fourth stage uses a full
    # h increment, so a half-step geometric bound would be unsound.
    # Isotropic drag is multiplied
    # by ||q_stage||^4, exactly; it must not be silently replaced by one.
    drag_factor = 1/(1-h*omega)**4
    gamma = drag*drag_factor/mass
    if 2*gamma*gap >= 1:
        raise ValueError('sampling gap cannot close the stage acceleration bound')
    battery_rate = db/H
    # Sample-and-hold target drift plus a stable positive motor recurrence.
    # H/(1-exp(-H/lag)) <= lag+H; numerical per-macro errors form a geometric
    # series too. Removing one period covers the transport/settling phase.
    decay = decay_upper((settle-delay-H)/lag)
    macro_decay = decay_upper(H/lag)
    motor_error = (decay*(force+hold) + hold*battery_rate*(lag+H)
                   + epsf/(1-macro_decay))
    # Numerical motor error can exceed the convex-hull force endpoint. Bound
    # the axis-error product using |M| <= hold+motor_error, rather than silently
    # assuming the issued-command maximum is an exact motor saturation.
    fixed_error = (motor_error+(hold+motor_error)*up)/mass
    # RK4 weighted velocity increments telescope exactly. Macro rounding
    # introduces at most epsv/H into the averaged acceleration balance.
    numerical_rate = epsv/H
    endpoint_rate = (abs(v[-1]-v[0])+2*ev)/duration + numerical_rate
    observed_speed = max(map(abs, v))+ev
    # Use the LEFT packet for every intervening RK4 stage: its distance is at
    # most gap*A plus all intervening macro numerical jumps. No ODE solution
    # or unproved continuous interpolation is needed.
    known_speed = observed_speed+numerical_rate*gap
    authority_force_drift = hold*battery_rate*(duration+H)/mass
    acceleration = ((endpoint_rate+2*gamma*known_speed+2*fixed_error
                     +2*authority_force_drift)/(1-2*gamma*gap))
    stage_speed = known_speed+acceleration*gap
    mismatch = (endpoint_rate+gamma*stage_speed+fixed_error
                +authority_force_drift)
    return {
        'schema_version': 'ares.conditional_discrete_trim_lemma.v2',
        'conditional_trim_acceleration_bound': mismatch,
        'conditional_stage_acceleration_bound': acceleration,
        'conditional_stage_vertical_speed_bound': stage_speed,
        'motor_tracking_force_bound': motor_error,
        'stage_drag_multiplier_upper': drag_factor,
        'position_offset_coefficient_per_mass_upper': 1/(bmin*m0*kp),
        'history_duration': duration,
        'maximum_packet_gap': gap,
        'all_decisions_exact_rational': True,
        'uses_continuous_ODE_solution': False,
        'hypotheses_independently_established': False,
        'flight_certificate': False,
    }


def example():
    return rk4_trim_bound(
        times=[F(i, 100) for i in range(101)],
        measured_vertical_velocity=[F(i, 50000) for i in range(101)],
        velocity_error=F(1, 100), control_period=F(1, 100),
        physics_period=F(1, 500), hold_thrust=5, public_force_upper=F(13, 2),
        mass_lower=F(44999998, 100000000), drag_upper=F(20000001, 100000000),
        authority_drop_per_period_upper=F(7, 1000000),
        motor_lag_upper=F(4000001, 100000000), transport_upper=F(2000001, 100000000),
        constant_command_settle=F(7, 10), stage_up_axis_error=F(1, 5000),
        stage_angular_rate_upper=1, velocity_roundoff_per_period=F(53, 100000000),
        motor_roundoff_per_period=F(7, 1000000))


if __name__ == '__main__':
    import json
    from pathlib import Path
    result = example()
    serial = {k: ({'rational': str(v), 'display_only': float(v)}
                  if isinstance(v, F) else v) for k, v in result.items()}
    Path(__file__).with_suffix('.json').write_text(json.dumps(serial, indent=2)+'\n')
    print(json.dumps(serial, indent=2))
