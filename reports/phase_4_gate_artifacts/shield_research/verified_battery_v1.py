"""Exact binary32 battery primitive and conditional updater bounds.

Research CPU backend only. Replaces native pow(1.5) with a binary32 result whose
rounding bin is verified by squaring rational endpoints. The physical power law
and original multiplication/update ordering are retained. No flight certificate.
"""
from fractions import Fraction as F
import math
import struct

U = F(1, 2**24)
ETA = F(1, 2**150)


def bits(value):
    return struct.unpack('>I', struct.pack('>f', float(value)))[0]


def word(value):
    return struct.unpack('>f', struct.pack('>I', value))[0]


def frac(value):
    if isinstance(value, F):
        return value
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError('finite rational or real input required')
    return F.from_float(float(value))


def positive_bin(index):
    center = frac(word(index))
    previous = frac(word(index-1)) if index else -frac(word(1))
    following = frac(word(index+1))
    return (previous+center)/2, (center+following)/2


def rn32_positive(value):
    """Round a nonnegative rational, checking its exact rounding bin."""
    value = frac(value)
    if value < 0 or value > 20000:
        raise ValueError('positive finite research domain required')
    guess = bits(value)
    for index in range(max(0, guess-1), guess+2):
        lo, hi = positive_bin(index)
        even = index % 2 == 0
        if (value > lo or even and value == lo) and (value < hi or even and value == hi):
            return word(index)
    raise ArithmeticError('exact rational rounding bin not isolated')


def power32(thrust):
    """Verified RN32(thrust**1.5); guesses never establish correctness."""
    thrust = frac(thrust)
    if not 0 <= thrust <= F(13, 2):
        raise ValueError('verified power domain is [0,6.5] N')
    if not thrust:
        return 0.0
    cube = thrust**3
    guess = bits(float(thrust)**1.5)
    for index in range(max(0, guess-1), guess+2):
        lo, hi = positive_bin(index)
        even = index % 2 == 0
        lower = lo < 0 or cube > lo*lo or even and cube == lo*lo
        upper = cube < hi*hi or even and cube == hi*hi
        if lower and upper:
            return word(index)
    raise ArithmeticError('exact algebraic rounding bin not isolated')


def authority32(energy, capacity):
    """Original RNE32 division/product/subtraction/clamp evaluation."""
    energy, capacity = frac(energy), frac(capacity)
    if not 0 <= energy <= 10002 or not 43200 <= capacity <= 100000:
        raise ValueError('battery proof domain exceeded')
    quotient = rn32_positive(energy/capacity)
    product = rn32_positive(frac(word(bits(.2)))*frac(quotient))
    result = rn32_positive(1-frac(product))
    return max(.5, min(1., result))


def energy_step32(energy, thrust, dt=.01):
    """Original order: E += (RN32(T**1.5)*dt)*8, with exact word checks."""
    energy = frac(energy)
    if not 0 <= energy <= 10000:
        raise ValueError('energy proof domain exceeded')
    product = rn32_positive(frac(power32(thrust))*frac(word(bits(dt))))
    increment = rn32_positive(frac(product)*8)
    return rn32_positive(energy+frac(increment))


def updater_bound():
    """Universal rational inequalities over the entire stated CPU domain."""
    H = frac(word(bits(.01)))
    c = frac(word(bits(.2)))
    Emax, capacity = F(10000), F(43200)
    # sqrt(6.5) < 2.55 follows by squaring, so T**1.5 < 16.575.
    assert F(255, 100)**2 > F(13, 2)
    power = F(13, 2)*F(255, 100)
    power_word = power*(1+U)+ETA
    step_product = power_word*H*(1+U)+ETA
    increment = step_product*8*(1+U)+ETA
    energy_jump = increment+U*(Emax+increment)+ETA
    # Error relative to 1-c*E/C, with c the actual binary32 coefficient.
    def authority_error(maximum):
        q = maximum/capacity
        eq = U*q+ETA
        ez = c*eq+U*c*(q+eq)+ETA
        return ez+U+ETA  # 0 <= exact subtraction input <= 1
    error = authority_error(Emax+energy_jump)
    decrease = c*energy_jump/capacity+2*error
    lower = 1-c*Emax/capacity-authority_error(Emax)
    claimed = F(64, 10000000)
    assert decrease < claimed
    assert lower > F(19, 20)
    # For a ten-second terminal run starting at <=8000 J, this deliberately
    # looser per-update constant keeps the entire run in the audited domain.
    assert F(8000)+1000*energy_jump < Emax
    return {
        'schema_version': 'ares.verified_cpu_battery_lemma.v1',
        'conditional_arithmetic_lemma_passed': True,
        'flight_certificate': False,
        'native_pow_audited': False,
        'requires_verified_power_primitive': True,
        'requires_binary32_RNE_basic_operations_and_DAZ_FTZ_off': True,
        'thrust_interval_N': [0, 6.5],
        'energy_interval_J': [0, 10000],
        'capacity_lower_J': 43200,
        'period_word': str(H),
        'energy_jump_upper_J': energy_jump,
        'authority_decrease_upper': decrease,
        'declared_authority_decrease_per_period': claimed,
        'authority_lower': lower,
        'ten_second_initial_energy_upper_J': 8000,
        'monotonicity': 'positive RN32 primitives are monotone; E_next>=E, b_next<=b',
        'algebraic_rounding_bin_decisions_exact': True,
    }


if __name__ == '__main__':
    import json
    from pathlib import Path
    import random
    import torch
    torch.set_num_threads(1)
    generator = random.Random(4327)
    largest_native_power_difference = 0.
    cases = [0., word(1), word(0x007fffff), word(0x00800000), 1., 6.5]
    cases += [word(bits(generator.uniform(0., 6.5))) for _ in range(1000)]
    for thrust in cases:
        verified = power32(thrust)
        native = torch.tensor(thrust, dtype=torch.float32).pow(1.5).item()
        largest_native_power_difference = max(largest_native_power_difference, abs(verified-native))
        for energy in (0., 8000., 10000.):
            next_energy = energy_step32(energy, thrust)
            assert F.from_float(next_energy)-F.from_float(energy) <= updater_bound()['energy_jump_upper_J']
            for capacity in (43200., 52800.):
                old, new = authority32(energy, capacity), authority32(next_energy, capacity)
                assert 0 <= frac(old)-frac(new) <= F(64, 10000000)
    result = updater_bound()
    result.update(power_regression_cases=len(cases),
                  maximum_native_power_absolute_difference=largest_native_power_difference,
                  finite_regressions_are_not_the_universal_proof=True)
    serial = {k: ({'rational': str(v), 'display_only': float(v)}
                  if isinstance(v, F) else v) for k, v in result.items()}
    Path(__file__).with_suffix('.json').write_text(json.dumps(serial, indent=2)+'\n')
    print(json.dumps({k: v for k, v in serial.items() if not isinstance(v, dict)}, indent=2))
