"""Exact conditional initial terminal-set membership; research only.

No plant parameters, hidden motor state, gravity or energy are accepted. Public
packet bounds and proof-domain hypotheses must be supplied independently.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).with_name('terminal_reservation_v1')))
from reservation import rational, box_in_ellipsoid


def exact_vector(values, width, name):
    result = tuple(map(rational, values))
    if len(result) != width:
        raise ValueError(name + ' has wrong dimension')
    return result


def sqrt_bounds(value):
    value = rational(value)
    if value < 0:
        raise ValueError('nonnegative radicand required')
    scale = 2**90
    lower = F(math.isqrt(value.numerator * scale*scale // value.denominator), scale)
    upper = lower if lower*lower == value else lower + F(1, scale)
    assert lower*lower <= value <= upper*upper
    return lower, upper


def projective_xyz(q, overapproximation):
    q = exact_vector(q, 4, 'quaternion')
    error = rational(overapproximation)
    if error < 0 or q[3] <= 0:
        raise ValueError('positive-qw chart and nonnegative state overapproximation required')
    norm2 = sum(x*x for x in q)
    shell = F(8, 2**24)
    if not (1-shell)**2 <= norm2 <= (1+shell)**2:
        raise ValueError('raw quaternion outside required 8u norm shell')
    lower, upper = sqrt_bounds(norm2)
    if lower <= 0:
        raise ValueError('nonzero norm required')
    bounds = []
    for value in q[:3]:
        endpoints = (value/lower, value/upper)
        bounds.append((min(endpoints)-error, max(endpoints)+error))
    return bounds


def quadratic(matrix, x):
    return sum(x[i]*matrix[i][j]*x[j] for i in range(len(x)) for j in range(len(x)))


def initial_terminal_membership(*, p, v, q, w, anchor, correlated_trim_force_upper,
                                motor_tracking_error, shapes,
                                ep=F(1, 200), ev=F(1, 100), ew=F(1, 1000),
                                q_state_overapprox_error=F(1, 1000000),
                                battery_interval=(F(19, 20), F(1))):
    p, v, w, anchor = (exact_vector(values, 3, name) for values, name in
                       ((p, 'position'), (v, 'velocity'), (w, 'angular velocity'), (anchor, 'anchor')))
    ep, ev, ew, qe, dmax, motor = map(rational, (ep, ev, ew, q_state_overapprox_error,
                                      correlated_trim_force_upper, motor_tracking_error))
    if min(ep, ev, ew, qe, dmax, motor) < 0:
        raise ValueError('error and force bounds must be nonnegative')
    battery = exact_vector(battery_interval, 2, 'battery interval')
    if not F(19,20) <= battery[0] <= battery[1] <= 1:
        raise ValueError('battery authority domain must lie inside [.95,1]')
    if not isinstance(shapes, dict) or any(name not in shapes for name in ('planar_x', 'vertical', 'yaw')):
        raise ValueError('explicit planar_x, vertical and yaw shape matrices required')
    xyz = projective_xyz(q, qe)
    offset = tuple(a-b for a,b in zip(p, anchor))
    def band(center, error): return center-error, center+error
    def channel(shape, bounds):
        if len(shape) != len(bounds):
            raise ValueError('shape dimension does not match channel')
        result = box_in_ellipsoid(shape, [a for a,b in bounds], [b for a,b in bounds])
        return {'included':result['included'], 'maximum':result.get('maximum'),
                'corners_checked':result.get('corners_checked',0),
                'state_bounds':bounds, 'flight_certificate':False}
    zero = (F(0), F(0))
    channels = {
        'planar_x':channel(shapes['planar_x'], [band(offset[0],ep),band(v[0],ev),xyz[1],band(w[1],ew),zero,zero]),
        'planar_y':channel(shapes['planar_x'], [band(offset[1],ep),band(v[1],ev),
                          (-xyz[0][1],-xyz[0][0]),band(-w[0],ew),zero,zero]),
        'yaw':channel(shapes['yaw'], [xyz[2],band(w[2],ew),zero,zero]),
    }
    # Compute the exact inverse once; a zero box supplies only the SPD/inverse
    # check. Vertical image is multi-affine in b,d,ep,ev,deltaM, so lies in the
    # convex hull of these 32 vertices, preserving the public trim correlation.
    if len(shapes['vertical']) != 5:
        raise ValueError('vertical shape must have dimension five')
    inverse = box_in_ellipsoid(shapes['vertical'], [0]*5, [0]*5)
    if not inverse['spd']:
        channels['vertical'] = {'included':False,'maximum':None,'vertices_checked':0,'flight_certificate':False}
    else:
        vertices = []
        for b,d,pe,ve,delta in product(battery,(-dmax,dmax),(-ep,ep),(-ev,ev),(-motor,motor)):
            vertices.append((offset[2]+pe-d/F(3,4),v[2]+ve,b*d+delta,d,d))
        maximum = max(quadratic(inverse['shape_matrix_inverse'], x) for x in vertices)
        channels['vertical'] = {'included':maximum<=1,'maximum':maximum,
                                'vertices_checked':len(vertices),'vertices':vertices,'flight_certificate':False}
    return {'schema_version':'ares.initial_terminal_membership.v1',
            'initial_set_included':all(value['included'] for value in channels.values()),
            'channels':channels,'projective_xyz_bounds':xyz,
            'quaternion_norm_shell':'1 +/- 8*2^-24',
            'q_overapproximation_is_state_uncertainty_not_measurement_noise':True,
            'trim_force_domain':[-dmax,dmax],'motor_tracking_error':motor,
            'battery_interval':battery,'shape_convention':"E(Q)={x:x'Q^-1 x<=1}",
            'hypotheses_external':True,'hypotheses_independently_established':False,
            'rpi_proof_complete':False,'flight_certificate':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=initial_terminal_membership(**json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result,indent=2,default=str)+'\n')
