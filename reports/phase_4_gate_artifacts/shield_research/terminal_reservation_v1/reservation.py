"""Exact research-only terminal geometry and fail-closed reservation metadata.

No scene/ground-truth access. Caller declarations are not validated physical
proofs. This module cannot issue a flight certificate.
"""
from fractions import Fraction as F
from itertools import product
import math
from functools import lru_cache, cached_property
from dataclasses import dataclass


def rational(value):
    if isinstance(value, bool):
        raise ValueError('boolean is not a numeric bound')
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('finite bound required')
        return F.from_float(value)
    if isinstance(value, (int, str, F)):
        return F(value)
    raise ValueError('exact rational string, integer or binary64 required')


def vector(values):
    return tuple(map(rational, values))


def spd_ldl(matrix):
    """Exact positive-definiteness check with an unpivoted LDL decomposition."""
    q = tuple(vector(row) for row in matrix)
    n = len(q)
    if not 1 <= n <= 6 or any(len(row) != n for row in q):
        raise ValueError('square matrix dimension must be between 1 and 6')
    if any(q[i][j] != q[j][i] for i in range(n) for j in range(n)):
        raise ValueError('symmetric matrix required')
    lower = [[F(i == j) for j in range(n)] for i in range(n)]
    diagonal = []
    for j in range(n):
        pivot = q[j][j] - sum(lower[j][k] ** 2 * diagonal[k] for k in range(j))
        if pivot <= 0:
            return {'spd': False, 'matrix': q, 'pivot': pivot}
        diagonal.append(pivot)
        for i in range(j + 1, n):
            lower[i][j] = (q[i][j] - sum(lower[i][k] * lower[j][k] * diagonal[k]
                                       for k in range(j))) / pivot
    return {'spd': True, 'matrix': q, 'lower': lower, 'diagonal': diagonal}


def box_in_metric_ellipsoid(matrix, lower, upper):
    """Test box inclusion in E(Q)={x:x'Qx<=1}; origin-centered coordinates."""
    ldl = spd_ldl(matrix)
    lo, hi = vector(lower), vector(upper)
    n = len(ldl['matrix'])
    if len(lo) != n or len(hi) != n or any(a > b for a, b in zip(lo, hi)):
        raise ValueError('ordered box with matching dimension required')
    if not ldl['spd']:
        return {'included': False, 'spd': False, 'flight_certificate': False}
    q = ldl['matrix']
    values = [sum(x[i] * q[i][j] * x[j] for i in range(n) for j in range(n))
              for x in product(*(tuple(dict.fromkeys(pair)) for pair in zip(lo, hi)))]
    maximum = max(values)
    # Convexity of an SPD quadratic ensures its box maximum occurs at a vertex.
    return {'included': maximum <= 1, 'spd': True, 'maximum': maximum,
            'corners_checked': len(values), 'flight_certificate': False}


@dataclass(frozen=True)
class ExactShapeMetric:
    shape: tuple
    inverse: tuple

    @cached_property
    def integer_metric(self):
        denominator = math.lcm(*(value.denominator for row in self.inverse for value in row))
        return denominator, tuple(tuple(value.numerator*(denominator//value.denominator)
                                        for value in row) for row in self.inverse)

    def quadratic(self, point):
        point = vector(point)
        if len(point) != len(self.shape):
            raise ValueError('point dimension does not match shape')
        # One common denominator turns many Fraction products/GCDs into exact
        # integer operations; symmetry and zero coordinates reduce the work.
        denominator = math.lcm(*(value.denominator for value in point))
        active = [(i,value.numerator*(denominator//value.denominator))
                  for i,value in enumerate(point) if value]
        metric_denominator, matrix = self.integer_metric
        numerator = sum(x*x*matrix[i][i] +
                        2*sum(x*y*matrix[i][j] for j,y in active if j>i)
                        for i,x in active)
        return F(numerator,metric_denominator*denominator*denominator)

    def box(self, lower, upper):
        lo, hi = vector(lower), vector(upper)
        if len(lo) != len(self.shape) or len(hi) != len(lo) or any(a>b for a,b in zip(lo,hi)):
            raise ValueError('ordered box with matching dimension required')
        corners = list(product(*(tuple(dict.fromkeys(pair)) for pair in zip(lo,hi))))
        maximum = max(self.quadratic(x) for x in corners)
        return {'included':maximum<=1,'spd':True,'maximum':maximum,
                'corners_checked':len(corners),'flight_certificate':False,
                'shape_matrix_inverse':[list(row) for row in self.inverse],
                'convention':"x'Q^-1 x <= 1"}


@lru_cache(maxsize=128)
def _shape_metric(shape):
    checked = spd_ldl(shape)
    if not checked['spd']:
        return None
    n = len(shape)
    augmented = [list(shape[i]) + [F(i == j) for j in range(n)] for i in range(n)]
    for j in range(n):
        pivot = augmented[j][j]
        augmented[j] = [value / pivot for value in augmented[j]]
        for i in range(n):
            if i != j:
                factor = augmented[i][j]
                augmented[i] = [a-factor*b for a,b in zip(augmented[i],augmented[j])]
    return ExactShapeMetric(shape,tuple(tuple(row[n:]) for row in augmented))


def prepare_shape_metric(shape_matrix):
    """Cache exact SPD proof/inverse by immutable represented shape values."""
    return _shape_metric(tuple(vector(row) for row in shape_matrix))


def box_in_ellipsoid(shape_matrix, lower, upper, *, metric=None):
    """E(Q)={x:x'Q^-1 x<=1}; optional metric must match the exact shape."""
    shape = tuple(vector(row) for row in shape_matrix)
    trusted_metric = _shape_metric(shape)
    if metric is not None and (not isinstance(metric, ExactShapeMetric) or metric != trusted_metric):
        raise ValueError('precomputed metric does not match verified shape inverse')
    metric = trusted_metric
    if metric is None:
        return {'included':False,'spd':False,'flight_certificate':False}
    return metric.box(lower,upper)


def free_grid_reserve(lower, upper, *, origin, shape, resolution, free_cells,
                      body_radius=0, interstep_reserve=0):
    """Every CLOSED cell meeting the inflated closed AABB must be explicit free.

    Interior support is strict. Even a zero-volume touch of an unknown cell or
    map support boundary rejects. Only caller-supplied known-free indices enter.
    """
    lo, hi, start = vector(lower), vector(upper), vector(origin)
    if len(lo) != 3 or len(hi) != 3 or len(start) != 3 or len(shape) != 3:
        raise ValueError('3D geometry required')
    if any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('positive integer grid dimensions required')
    step, radius, reserve = map(rational, (resolution, body_radius, interstep_reserve))
    if step <= 0 or radius < 0 or reserve < 0 or any(a > b for a, b in zip(lo, hi)):
        raise ValueError('ordered box and nonnegative inflation required')
    inflation = radius + reserve
    lo = tuple(x - inflation for x in lo)
    hi = tuple(x + inflation for x in hi)
    if any(lo[k] <= start[k] or hi[k] >= start[k] + shape[k] * step for k in range(3)):
        return {'reserved': False, 'reason': 'support_boundary', 'flight_certificate': False}
    ranges = []
    for k in range(3):
        a, b = (lo[k] - start[k]) / step, (hi[k] - start[k]) / step
        # [i,i+1] intersects [a,b] iff i>=ceil(a)-1 and i<=floor(b).
        first = -((-a.numerator) // a.denominator) - 1
        last = b.numerator // b.denominator
        ranges.append(range(first, last + 1))
    free = set(map(tuple, free_cells))
    touched = list(product(*ranges))
    missing = [cell for cell in touched if cell not in free]
    return {'reserved': not missing, 'reason': 'explicit_free' if not missing else 'unknown_or_occupied',
            'intersecting_cells': touched, 'unavailable_cells': missing,
            'inflated_lower': lo, 'inflated_upper': hi, 'flight_certificate': False}


class ReservationCache:
    """Monotonic proposal clock; rejection never replaces the previous record.

    Retained records can expire: use current() before relying on any metadata.
    Obligations are caller declarations with named known artifacts, not checked
    RPI proofs. Geometry alone can never commit a reservation.
    """
    REQUIRED = ('geometry', 'input_contract', 'backup_reachability',
                'terminal_invariance', 'arithmetic', 'timing')

    def __init__(self):
        self.last_proposal_time = None
        self.cached = None
        self.last_observed_time = None

    def current(self, time):
        now = rational(time)
        if self.last_observed_time is not None and now < self.last_observed_time:
            return None
        self.last_observed_time = now
        if self.cached is None or now > self.cached['backup_deadline'] or now > self.cached['supplier_expiry']:
            return None
        return self.cached.copy()

    def propose(self, *, time, backup_deadline, supplier_expiry, obligations, known_artifacts):
        now, deadline, expiry = map(rational, (time, backup_deadline, supplier_expiry))
        def reject(reason):
            return {'committed': False, 'reason': reason, 'cached_preserved': self.cached is not None,
                    'flight_certificate': False}
        if self.last_observed_time is not None and now <= self.last_observed_time:
            return reject('duplicate_or_rollback_time')
        self.last_proposal_time = now
        self.last_observed_time = now
        if not now <= deadline <= expiry:
            return reject('stale_or_crossed_deadlines')
        known = set(known_artifacts)
        for name in self.REQUIRED:
            value = obligations.get(name)
            if (not isinstance(value, dict) or value.get('valid') is not True
                    or not isinstance(value.get('artifact'), str) or not value['artifact']
                    or value['artifact'] not in known):
                return reject('unproved_obligation:' + name)
        self.cached = {'committed_at': now, 'backup_deadline': deadline,
                       'supplier_expiry': expiry,
                       'proof_artifacts': {name: obligations[name]['artifact'] for name in self.REQUIRED},
                       'flight_certificate': False, 'caller_declarations_only': True}
        return {'committed': True, 'reason': 'explicit_obligations_recorded', 'flight_certificate': False}
