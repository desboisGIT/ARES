"""Independent outward-binary64 audit of small symmetric matrix inequalities.

Every elementary NumPy operation is followed by nextafter outwards. No BLAS,
eigenvalue tolerance, numerical solver status or sampled vertex subset is used.
Input intervals must be independently proved to contain the exact coefficients.
IEEE binary64 RNE, gradual underflow and correctly rounded basic operations /
nextafter are explicit platform hypotheses. This is not a flight certificate.
"""
from fractions import Fraction as F
import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def iv(lo, hi=None):
    lo = np.asarray(lo, dtype=np.float64)
    hi = lo.copy() if hi is None else np.asarray(hi, dtype=np.float64)
    if lo.shape != hi.shape or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
        raise ValueError('ordered finite binary64 interval arrays required')
    return lo, hi


def exact_rational_interval(value):
    value = F(value)
    candidate = float(value)
    represented = F.from_float(candidate)
    lo = np.nextafter(candidate, -np.inf) if represented > value else candidate
    hi = np.nextafter(candidate, np.inf) if represented < value else candidate
    return iv(lo, hi)


def down(value):
    return np.nextafter(value, -np.inf)


def up(value):
    return np.nextafter(value, np.inf)


def add(a, b):
    return down(a[0]+b[0]), up(a[1]+b[1])


def neg(a):
    return -a[1], -a[0]


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    products = np.stack((a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]))
    return down(products.min(axis=0)), up(products.max(axis=0))


def square(a):
    left, right = a[0]*a[0], a[1]*a[1]
    lo = np.where((a[0] <= 0) & (a[1] >= 0), 0., np.maximum(0., down(np.minimum(left, right))))
    return lo, up(np.maximum(left, right))


def divide(a, b):
    if np.any((b[0] <= 0) & (b[1] >= 0)):
        raise ArithmeticError('interval denominator contains zero')
    return mul(a, (down(1/b[1]), up(1/b[0])))


def entry(a, *indices):
    return a[0][(..., *indices)], a[1][(..., *indices)]


def mm(a, b):
    """Small matrix multiply using only explicit interval products and sums."""
    shape = np.broadcast_shapes(a[0].shape[:-2], b[0].shape[:-2])
    rows, inner, cols = a[0].shape[-2], a[0].shape[-1], b[0].shape[-1]
    if b[0].shape[-2] != inner:
        raise ValueError('matrix dimensions disagree')
    lo, hi = np.empty((*shape, rows, cols)), np.empty((*shape, rows, cols))
    for i in range(rows):
        for j in range(cols):
            value = iv(np.zeros(shape))
            for k in range(inner):
                value = add(value, mul(entry(a, i, k), entry(b, k, j)))
            lo[..., i, j], hi[..., i, j] = value
    return lo, hi


def transpose(a):
    return np.swapaxes(a[0], -1, -2), np.swapaxes(a[1], -1, -2)


def interval_ldl(a):
    """Sufficient SPD test for every symmetric matrix in the input intervals.

    The symmetric matrix is defined by its lower triangle. No pivot may include
    zero. A failure means this sufficient check failed, never a proof of SPD.
    """
    n = a[0].shape[-1]
    if a[0].shape[-2] != n or not 1 <= n <= 6:
        raise ValueError('square matrix dimension between one and six required')
    shape = a[0].shape[:-2]
    lower = {}
    diagonal = []
    minimum = np.full(shape, np.inf)
    for j in range(n):
        pivot = entry(a, j, j)
        for k in range(j):
            pivot = sub(pivot, mul(square(lower[j, k]), diagonal[k]))
        minimum = np.minimum(minimum, pivot[0])
        bad = (pivot[0] <= 0) | ~np.isfinite(pivot[0]) | ~np.isfinite(pivot[1])
        if np.any(bad):
            return {'passed': False, 'pivot': j, 'bad_batch_indices': np.argwhere(bad).tolist(),
                    'minimum_pivot_lower': float(np.min(minimum))}
        diagonal.append(pivot)
        for i in range(j+1, n):
            value = entry(a, i, j)
            for k in range(j):
                value = sub(value, mul(mul(lower[i, k], lower[j, k]), diagonal[k]))
            lower[i, j] = divide(value, pivot)
    return {'passed': True, 'minimum_pivot_lower': float(np.min(minimum))}


def _verify_arrays(data, *, alpha, chunk_size=2048, command_signs=(1,)):
    """Audit Q-AQA'/alpha-ww'/(1-alpha)>0 for EVERY supplied vertex/sign.

    w = command_sign*command_w(vertex) + additive_offsets(sign[,vertex]).
    Omitting the global negative sign is justified only when the independent
    exporter proves the supplied offsets are a centrally symmetric box: ww'
    is unchanged by global sign inversion. The schema never assumes this.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('positive integer chunk size required')
    if not command_signs or any(x not in (-1, 1) for x in command_signs):
        raise ValueError('explicit command signs in {-1,1} required')
    a = iv(data['A_lo'], data['A_hi'])
    q = iv(data['Q_lo'], data['Q_hi'])
    outer = iv(data['outer_Q_lo'], data['outer_Q_hi'])
    w = iv(data['command_w_lo'], data['command_w_hi'])
    offsets = iv(data['additive_offsets_lo'], data['additive_offsets_hi'])
    vertices, n, columns = a[0].shape
    if (n != columns or q[0].shape != (n, n) or outer[0].shape != (n, n)
            or w[0].shape != (vertices, n) or offsets[0].ndim not in (2, 3)
            or offsets[0].shape[-1] != n
            or offsets[0].ndim == 3 and offsets[0].shape[0] != vertices):
        raise ValueError('coefficient-array shapes disagree')
    if (not np.array_equal(q[0], q[0].T) or not np.array_equal(q[1], q[1].T)
            or not np.array_equal(outer[0], outer[0].T) or not np.array_equal(outer[1], outer[1].T)):
        raise ValueError('symmetric shape/outer-shape intervals required')
    alpha_exact = F(alpha)
    if not 0 < alpha_exact < 1:
        raise ValueError('alpha must be strictly between zero and one')
    weight = exact_rational_interval(alpha_exact)
    remaining = exact_rational_interval(1-alpha_exact)
    shape_check = interval_ldl(q)
    outer_check = interval_ldl(sub(outer, q))
    if not shape_check['passed'] or not outer_check['passed']:
        return {'passed': False, 'Q_SPD': shape_check, 'outer_minus_Q_SPD': outer_check,
                'flight_certificate': False, 'matrix_inequalities_checked': 0}
    signs = offsets[0].shape[-2]
    if signs < 1 or vertices < 1:
        raise ValueError('nonempty vertex and sign sets required')
    count = 0
    minimum = np.inf
    for begin in range(0, vertices, chunk_size):
        end = min(vertices, begin+chunk_size)
        current = a[0][begin:end], a[1][begin:end]
        base = sub(q, divide(mm(mm(current, q), transpose(current)), weight))
        command = w[0][begin:end], w[1][begin:end]
        for sign in command_signs:
            signed = command if sign == 1 else neg(command)
            for index in range(signs):
                offset = (offsets[0][index], offsets[1][index]) if offsets[0].ndim == 2 else (
                    offsets[0][begin:end, index], offsets[1][begin:end, index])
                total = add(signed, offset)
                column = total[0][..., :, None], total[1][..., :, None]
                row = total[0][..., None, :], total[1][..., None, :]
                remainder = sub(base, divide(mul(column, row), remaining))
                check = interval_ldl(remainder)
                count += end-begin
                minimum = min(minimum, check['minimum_pivot_lower'])
                if not check['passed']:
                    return {'passed': False, 'vertex_begin': begin, 'vertex_end': end,
                            'command_sign': sign, 'offset_sign_index': index, 'failure': check,
                            'matrix_inequalities_checked': count, 'flight_certificate': False}
    return {'passed': True, 'Q_SPD': shape_check, 'outer_minus_Q_SPD': outer_check,
            'parameter_vertices': vertices, 'additive_sign_patterns': signs,
            'command_signs': list(command_signs), 'matrix_inequalities_checked': count,
            'minimum_interval_LDL_pivot_lower': minimum,
            'input_coefficient_containment_independently_required': True,
            'every_supplied_vertex_and_sign_checked': True,
            'eigenvalue_or_solver_tolerance_used': False,
            'flight_certificate': False}


def verify_arrays(data, *, alpha, chunk_size=2048, command_signs=(1,)):
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        return _verify_arrays(data, alpha=alpha, chunk_size=chunk_size,
                              command_signs=command_signs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path, help='NPZ interval coefficients')
    parser.add_argument('--alpha', required=True, help='exact rational string')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=2048)
    parser.add_argument('--both-command-signs', action='store_true')
    args = parser.parse_args()
    # Gradual underflow remains a platform hypothesis; numerical underflow is
    # enclosed by nextafter. Overflow, division by zero and NaNs fail closed.
    np.seterr(over='raise', invalid='raise', divide='raise', under='ignore')
    guard_path = Path(__file__).resolve().with_name('cpu_arithmetic_guard_v1.py')
    spec = importlib.util.spec_from_file_location('lmi_cpu_mode_guard', guard_path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    before = guard.probe()
    if not before['cpu_preconditions_met']:
        raise RuntimeError('required CPU rounding/gradual-underflow mode unavailable')
    helper = ctypes.CDLL(before['helper']['library_path'])
    helper.guard_read_mxcsr.restype = ctypes.c_uint
    helper.guard_read_round.restype = ctypes.c_int
    helper.guard_nearest_constant.restype = ctypes.c_int
    start = time.monotonic()
    with np.load(args.bundle, allow_pickle=False) as data:
        result = verify_arrays(data, alpha=args.alpha, chunk_size=args.chunk_size,
                               command_signs=(-1, 1) if args.both_command_signs else (1,))
    after = guard.check_control_values(helper.guard_read_mxcsr(), helper.guard_read_round(),
                                       helper.guard_nearest_constant())
    if not after['passed']:
        raise RuntimeError('CPU rounding/gradual-underflow mode changed during audit')
    result.update(schema_version='ares.outward_interval_LMI_audit.v1',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  coefficient_bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
                  alpha_exact=args.alpha, elapsed_s=time.monotonic()-start,
                  CPU_arithmetic_mode_checked_before_and_after=True,
                  CPU_mode_guard_sha256=hashlib.sha256(guard_path.read_bytes()).hexdigest(),
                  CPU_mode_before=before['control_state'], CPU_mode_after=after,
                  platform_hypotheses='IEEE binary64 RNE basic operations, gradual underflow, correctly rounded nextafter')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
