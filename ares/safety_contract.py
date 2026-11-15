"""Exact conditional measurement and homogeneous-voxel input witnesses.

External hard bounds are declarations, not inferred sensor guarantees. This
module never reads simulator truth and never issues a flight certificate. State
boxes refer to capture time; reaching the present needs separate reachability.
All geometric/freshness inequalities use exact rationals of binary64 inputs.
"""
from dataclasses import dataclass, field
from fractions import Fraction
import math
from numbers import Real

SAFETY_CONTRACT_VERSION = 'ares.safety_contract.v1'


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real binary64 value')
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f'{name} must be finite') from error
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def _fraction(value):
    return Fraction.from_float(value)


def _vector(value, name):
    try:
        result = tuple(value)
    except TypeError as error:
        raise ValueError(f'{name} must have three finite coordinates') from error
    if len(result) != 3:
        raise ValueError(f'{name} must have three finite coordinates')
    return tuple(_fraction(_number(x, name)) for x in result)


def _outward(value, *, upper):
    """Closest outward binary64 endpoint of an exact rational, or fail closed."""
    result = float(value)
    if not math.isfinite(result):
        raise OverflowError('bound cannot be represented as finite binary64')
    while ((upper and _fraction(result) < value)
           or (not upper and _fraction(result) > value)):
        result = math.nextafter(result, math.inf if upper else -math.inf)
        if not math.isfinite(result):
            raise OverflowError('bound cannot be represented as finite binary64')
    return result


def _outer_box(lower, upper):
    return (tuple(_outward(x, upper=False) for x in lower),
            tuple(_outward(x, upper=True) for x in upper))


@dataclass(frozen=True)
class MeasurementContract:
    """Supplier-declared deterministic sets, in a named world frame and SI units.

    Error radii are Euclidean bounds; the implementation uses componentwise
    outer boxes. world_ray_error directly bounds the complete world direction
    error, including calibration/orientation errors. Depth is a first-hit range
    parameter along that same possibly nonunit ray, not implicitly arc length.
    Unsupported source/basis tags remain representable but fail validation.
    """
    position_error_m: float
    velocity_error_mps: float
    world_ray_error: float
    depth_error_m: float
    capture_time_s: float
    expires_s: float
    supplier: str
    provenance: str
    source_kind: str
    bound_basis: str
    coordinate_frame: str = 'world'
    units: str = 'SI'
    voxel_model: str = 'unspecified'
    depth_model: str = 'unspecified'

    def __post_init__(self):
        for name in ('position_error_m', 'velocity_error_mps', 'world_ray_error',
                     'depth_error_m', 'capture_time_s', 'expires_s'):
            value = _number(getattr(self, name), name)
            object.__setattr__(self, name, value)
            if name.endswith(('error_m', 'error_mps')) or name == 'world_ray_error':
                if _fraction(value) < 0:
                    raise ValueError(f'{name} must be nonnegative')
        if _fraction(self.expires_s) < _fraction(self.capture_time_s):
            raise ValueError('expiry must not precede capture')
        for name in ('supplier', 'provenance', 'source_kind', 'bound_basis',
                     'coordinate_frame', 'units', 'voxel_model', 'depth_model'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'explicit nonempty {name} required')


@dataclass(frozen=True)
class ContractResult:
    conditional_input_valid: bool
    free_witness: bool
    reason: str
    capture_time_s: float
    expires_s: float
    supplier: str
    provenance: str
    flight_certificate: bool = field(default=False, init=False)


@dataclass(frozen=True)
class StateBoxes(ContractResult):
    position_lower: tuple[float, float, float] | None = None
    position_upper: tuple[float, float, float] | None = None
    velocity_lower: tuple[float, float, float] | None = None
    velocity_upper: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class FreeWitness(ContractResult):
    ray_point_lower: tuple[float, float, float] | None = None
    ray_point_upper: tuple[float, float, float] | None = None
    minimum_interior_slack_lower_m: float | None = None
    first_hit_slack_lower_m: float | None = None


def _result(result_type, contract, valid, reason, *, free=False, **details):
    return result_type(valid, free, reason, contract.capture_time_s,
                       contract.expires_s, contract.supplier, contract.provenance, **details)


def validate_contract(contract, *, time_s):
    """Check declarations/freshness; this cannot authenticate a supplier's claim."""
    if not isinstance(contract, MeasurementContract):
        raise TypeError('explicit MeasurementContract required')
    now = _fraction(_number(time_s, 'time_s'))
    if contract.source_kind != 'hard_external' or contract.bound_basis != 'deterministic_guarantee':
        return _result(ContractResult, contract, False, 'source_not_deterministic_hard')
    if contract.coordinate_frame != 'world' or contract.units != 'SI':
        return _result(ContractResult, contract, False, 'unsupported_frame_or_units')
    if now < _fraction(contract.capture_time_s):
        return _result(ContractResult, contract, False, 'capture_in_future')
    if now > _fraction(contract.expires_s):
        return _result(ContractResult, contract, False, 'expired')
    return _result(ContractResult, contract, True, 'conditional_external_bounds')


def state_boxes(position, velocity, contract, *, time_s):
    """Outer boxes at capture epoch, conditional on externally supplied bounds.

    time_s checks freshness only. It does not move these boxes from capture time
    to current time, increase errors with age, or certify current flight state.
    """
    status = validate_contract(contract, time_s=time_s)
    if not status.conditional_input_valid:
        return _result(StateBoxes, contract, False, status.reason)
    p, v = _vector(position, 'position'), _vector(velocity, 'velocity')
    ep, ev = _fraction(contract.position_error_m), _fraction(contract.velocity_error_mps)
    try:
        pl, pu = _outer_box(tuple(x-ep for x in p), tuple(x+ep for x in p))
        vl, vu = _outer_box(tuple(x-ev for x in v), tuple(x+ev for x in v))
    except OverflowError:
        return _result(StateBoxes, contract, False, 'nonfinite_outward_bounds')
    return _result(StateBoxes, contract, True, 'capture_epoch_outer_boxes',
                   position_lower=pl, position_upper=pu, velocity_lower=vl, velocity_upper=vu)


def robust_free_witness(position, world_direction, measured_depth, voxel_lower,
                        voxel_upper, optical_range, contract, *, time_s):
    """Sufficient robust crossing of a supplier-declared homogeneous voxel.

    A common positive range s must be strictly before every possible first hit;
    the point of EVERY admissible actual ray must be strictly inside the voxel.
    Homogeneous/static occupancy then implies the whole voxel is free through
    contract expiry. A failed witness leaves the voxel unknown, never occupied
    or free. No map/scene/plant API, trig, direction norm or rotation is used.
    """
    status = validate_contract(contract, time_s=time_s)
    if not status.conditional_input_valid:
        return _result(FreeWitness, contract, False, status.reason)
    if (contract.voxel_model != 'homogeneous_static_until_expiry'
            or contract.depth_model != 'first_hit_range'):
        return _result(FreeWitness, contract, False, 'missing_homogeneous_first_hit_contract')
    p = _vector(position, 'position')
    d = _vector(world_direction, 'world_direction')
    lower, upper = _vector(voxel_lower, 'voxel_lower'), _vector(voxel_upper, 'voxel_upper')
    if any(lo >= hi for lo, hi in zip(lower, upper)):
        raise ValueError('voxel must have three strictly positive dimensions')
    try:
        depth = _fraction(_number(measured_depth, 'measured_depth'))
        s = _fraction(_number(optical_range, 'optical_range'))
    except ValueError:
        return _result(FreeWitness, contract, False, 'unavailable_depth_or_range')
    if depth <= 0 or s <= 0:
        return _result(FreeWitness, contract, False, 'nonpositive_depth_or_range')
    # Do not round depth-error, s*d, p+s*d, the uncertainty radius, or slack.
    # Each decision below is exact for the represented binary64 operands.
    hit_slack = depth-_fraction(contract.depth_error_m)-s
    radius = _fraction(contract.position_error_m)+s*_fraction(contract.world_ray_error)
    center = tuple(x+s*y for x, y in zip(p, d))
    point_lower = tuple(x-radius for x in center)
    point_upper = tuple(x+radius for x in center)
    interior = min(*(x-lo for x, lo in zip(point_lower, lower)),
                   *(hi-x for hi, x in zip(upper, point_upper)))
    try:
        pl, pu = _outer_box(point_lower, point_upper)
        details = dict(ray_point_lower=pl, ray_point_upper=pu,
                       minimum_interior_slack_lower_m=_outward(interior, upper=False),
                       first_hit_slack_lower_m=_outward(hit_slack, upper=False))
    except OverflowError:
        return _result(FreeWitness, contract, False, 'nonfinite_outward_bounds')
    if hit_slack <= 0:
        return _result(FreeWitness, contract, True, 'not_strictly_before_first_hit', **details)
    if interior <= 0:
        return _result(FreeWitness, contract, True, 'voxel_crossing_uncertain', **details)
    return _result(FreeWitness, contract, True, 'conditional_homogeneous_free_witness', free=True, **details)
