"""Conditional real-model public trim-history lemma, not a flight certificate.

All decision arithmetic is rational over supplied binary64 values. The caller
must separately establish deterministic measurement, upright-cone, motor and
numerical-error assumptions. Current Gaussian measurements cannot supply them.
"""
from fractions import Fraction as F
import math


def number(x):
    if isinstance(x, bool) or not math.isfinite(float(x)):
        raise ValueError('finite real input required')
    return F.from_float(float(x))


def sqrt_upper(x, digits=12):
    """Exact rational upper bound, independently checked by squaring."""
    if x < 0:
        raise ValueError('nonnegative radicand required')
    scale=10**digits
    n=math.isqrt(x.numerator*scale*scale//x.denominator)
    if F(n*n,scale*scale)<x:
        n+=1
    result=F(n,scale)
    assert result*result>=x
    return result


def exp_negative_upper(x, terms=48):
    """e^-x <= 1 / sum_{k=0}^n x^k/k!, x>=0; no libm assumption."""
    if x < 0:
        raise ValueError('nonnegative decay argument required')
    total=term=F(1)
    for k in range(1,terms+1):
        term*=x/k
        total+=term
    return 1/total


def trim_bound(*, times, measured_vertical_velocity, velocity_error,
               hold_thrust, public_force_upper, mass_lower, drag_upper,
               battery_energy_lower, motor_lag_upper, transport_upper,
               constant_command_settle, up_axis_error,
               motor_numerical_force_error=0., acceleration_numerical_error=0.,
               nominal_mass=.5, position_gain=1.5, battery_lower=.95,
               battery_update_interval=.01):
    """Bound |b(T)*F_hold/m-g| using only public history and declared bounds.

    Force upper must enclose initial motor thrust and every issued command over
    settling/history. Motor error is an already established total force-error
    budget, not an unverified per-operation precision estimate. Timestamps must
    bracket the complete measurement interval; maximum gap H is computed.
    """
    t=list(map(number,times));v=list(map(number,measured_vertical_velocity))
    if len(t)!=len(v) or len(t)<2 or any(b<=a for a,b in zip(t,t[1:])):
        raise ValueError('ordered full velocity history required')
    args=list(map(number,(velocity_error,hold_thrust,public_force_upper,mass_lower,
        drag_upper,battery_energy_lower,motor_lag_upper,transport_upper,
        constant_command_settle,up_axis_error,motor_numerical_force_error,
        acceleration_numerical_error,nominal_mass,position_gain,battery_lower,
        battery_update_interval)))
    ev,hold,force,mass,drag,energy,lag,delay,settle,up,ef,ea,m0,kp,bmin,hb=args
    if (min(ev,drag,delay,up,ef,ea)<0 or min(hold,force,mass,energy,lag,m0,kp,bmin)<=0
            or hb<=0 or hold>force or settle<delay+hb or not 0<bmin<=1 or up>2):
        raise ValueError('invalid physical/history bounds')
    duration=t[-1]-t[0];gap=max(b-a for a,b in zip(t,t[1:]))
    gamma=drag/mass
    if gamma*gap>=1:
        raise ValueError('sampling gap cannot close acceleration envelope')
    # Applied motor thrust is a convex recurrence of bounded commands and its
    # bounded initial state. The battery model uses 8*T^(3/2) J/s and coefficient .2.
    power=8*force*sqrt_upper(force)
    battery_rate=F(1,5)*power/energy
    # A staircase battery changes by at most B*h_b per update. Its endpoint
    # difference is bounded by B*(elapsed+h_b), not a continuous Lipschitz bound.
    decay=exp_negative_upper((settle-delay-hb)/lag)
    motor_error=decay*(force+hold)+hold*battery_rate*(lag+hb)+ef
    beta=hold*battery_rate/mass
    fixed_error=(motor_error+force*up)/mass+ea+beta*hb
    endpoint=(abs(v[-1]-v[0])+2*ev)/duration
    observed_velocity=max(map(abs,v))+ev
    acceleration=(endpoint+2*gamma*observed_velocity+2*fixed_error+
                  F(3,2)*beta*duration)/(1-gamma*gap)
    velocity=observed_velocity+acceleration*gap/2
    mismatch=endpoint+gamma*velocity+fixed_error+beta*duration/2
    # |z_eq-anchor| = m*|a_T|/(b*m0*kp). Caller supplies a separate mass upper
    # when using this coefficient; do not substitute sampled mass.
    return {'conditional_trim_acceleration_bound':mismatch,
            'conditional_acceleration_bound':acceleration,
            'conditional_vertical_speed_bound':velocity,
            'battery_rate_bound':battery_rate,'motor_tracking_force_bound':motor_error,
            'position_offset_coefficient_per_mass_upper':1/(bmin*m0*kp),
            'duration':duration,'maximum_sample_gap':gap,
            'flight_certificate':False,'supplier_contract_established':False,
            'floating_plant_error_budget_established':False}


def demo():
    return trim_bound(times=[i/100 for i in range(101)],
        measured_vertical_velocity=[.002*i/100 for i in range(101)],
        velocity_error=.01,hold_thrust=5.,public_force_upper=6.5,
        mass_lower=.44999998,drag_upper=.20000001,battery_energy_lower=43200.,
        motor_lag_upper=.04000001,transport_upper=.02000001,
        constant_command_settle=.7,up_axis_error=.0002)


if __name__=='__main__':
    import json
    result=demo()
    serial={k:({'numerator':str(v.numerator),'denominator':str(v.denominator),
                'display_only':float(v)} if isinstance(v,F) else v) for k,v in result.items()}
    print(json.dumps(serial,indent=2))
