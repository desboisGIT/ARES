"""Exact conditional current terminal-set membership; research only.

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
from reservation import rational, box_in_ellipsoid, prepare_shape_metric


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


def ordered_interval(values, name):
    interval = exact_vector(values, 2, name)
    if interval[0] > interval[1]:
        raise ValueError(name + ' interval is reversed')
    return interval


def propagate_public_residuals(*, motor_residual_interval, trim_force_interval,
                               decay_interval, battery_interval, hold_thrust,
                               issued_commands, authority_drop_upper,
                               numerical_error, mass_gravity_upper):
    """Conditional public interval propagation, not a plant-parameter observer.

    External obligations: e encloses the actual macro decay; delayed effective
    U is in the outer hull of exactly three public causal command words; battery
    authority is monotone in [.95,1]; numerical_error encloses interpolation,
    battery and Float32 plant effects jointly. No such hypotheses are proved here.
    """
    y = ordered_interval(motor_residual_interval, 'motor residual')
    d = ordered_interval(trim_force_interval, 'trim force')
    decay = ordered_interval(decay_interval, 'decay')
    battery = ordered_interval(battery_interval, 'battery')
    hold, drop, numerical, mg = map(rational, (hold_thrust, authority_drop_upper,
                                              numerical_error, mass_gravity_upper))
    if (hold <= 0 or min(drop, numerical, mg) < 0 or drop > 1
            or not 0 <= decay[0] <= decay[1] <= 1
            or not F(19,20) <= battery[0] <= battery[1] <= 1):
        raise ValueError('invalid public propagation domain')
    if len(issued_commands) != 3:
        raise ValueError('three public current/history command words required')
    commands = [exact_vector(command, 4, 'issued command') for command in issued_commands]
    thrusts = [command[0] for command in commands]
    if min(thrusts) < 0:
        raise ValueError('nonnegative issued thrust required')
    command_hull = (min(thrusts),max(thrusts))
    # t=b*(U-hold) is independent of e and y. Its extrema occur at
    # four b/U corners. For e in [0,1], the mixture is monotone in y,t;
    # each remaining affine function of e reaches its extrema at an endpoint.
    targets = [b*(u-hold) for b,u in product(battery,command_hull)]
    target_lower, target_upper = min(targets), max(targets)
    lower_mixtures = [e*y[0]+(1-e)*target_lower for e in decay]
    upper_mixtures = [e*y[1]+(1-e)*target_upper for e in decay]
    motor_bounds = (min(lower_mixtures)-numerical,
                    max(upper_mixtures)+hold*drop+numerical)
    drift = mg*drop/F(19,20)**2
    return {'schema_version':'ares.public_terminal_residual_propagation.v1',
            'motor_residual_interval':motor_bounds,
            'trim_force_interval':(d[0]-drift,d[1]),
            'trim_drift_lower':-drift,'delayed_command_thrust_hull':command_hull,
            'vertices_checked':64,'actual_mixture_evaluations':4,
            'actual_target_corner_evaluations':4,'multi_affine_vertex_enclosure':True,
            'hypotheses_external':True,'hypotheses_independently_established':False,
            'numerical_error_includes_interpolation_battery_and_float32_plant':True,
            'flight_certificate':False}


def current_terminal_membership(*, p, v, q, w, anchor, hold_thrust,
                                issued_queue, trim_force_interval,
                                motor_residual_interval, shapes,
                                ep=F(1, 200), ev=F(1, 100), ew=F(1, 1000),
                                q_state_overapprox_error=F(1, 1000000),
                                battery_interval=(F(19, 20), F(1)), metrics=None):
    p, v, w, anchor = (exact_vector(values, 3, name) for values, name in
                       ((p, 'position'), (v, 'velocity'), (w, 'angular velocity'), (anchor, 'anchor')))
    ep, ev, ew, qe, hold = map(rational, (ep, ev, ew, q_state_overapprox_error, hold_thrust))
    if min(ep, ev, ew, qe) < 0 or hold <= 0:
        raise ValueError('nonnegative errors and positive public hold thrust required')
    trim = ordered_interval(trim_force_interval, 'trim force')
    motor = ordered_interval(motor_residual_interval, 'motor residual')
    if len(issued_queue) != 2:
        raise ValueError('two issued command tuples required')
    queue = [exact_vector(command, 4, 'issued command') for command in issued_queue]
    if any(command[0] < 0 for command in queue):
        raise ValueError('nonnegative issued thrust required')
    battery = exact_vector(battery_interval, 2, 'battery interval')
    if not F(19,20) <= battery[0] <= battery[1] <= 1:
        raise ValueError('battery authority domain must lie inside [.95,1]')
    if not isinstance(shapes, dict) or any(name not in shapes for name in ('planar_x', 'vertical', 'yaw')):
        raise ValueError('explicit planar_x, vertical and yaw shape matrices required')
    prepared = {name: prepare_shape_metric(shapes[name]) for name in ('planar_x','vertical','yaw')}
    if metrics is not None:
        for name in prepared:
            if name not in metrics or metrics[name] != prepared[name]:
                raise ValueError('precomputed metrics do not match supplied shapes')
        prepared = metrics
    xyz = projective_xyz(q, qe)
    offset = tuple(a-b for a,b in zip(p, anchor))
    def band(center, error): return center-error, center+error
    def channel(name, bounds):
        shape = shapes[name]
        if len(shape) != len(bounds):
            raise ValueError('shape dimension does not match channel')
        result = box_in_ellipsoid(shape, [a for a,b in bounds], [b for a,b in bounds],metric=prepared[name])
        return {'included':result['included'], 'maximum':result.get('maximum'),
                'corners_checked':result.get('corners_checked',0),
                'state_bounds':bounds, 'flight_certificate':False}
    channels = {
        'planar_x':channel('planar_x', [band(offset[0],ep),band(v[0],ev),xyz[1],band(w[1],ew),(queue[0][2],queue[0][2]),(queue[1][2],queue[1][2])]),
        'planar_y':channel('planar_x', [band(offset[1],ep),band(v[1],ev),
                          (-xyz[0][1],-xyz[0][0]),band(-w[0],ew),(-queue[0][1],-queue[0][1]),(-queue[1][1],-queue[1][1])]),
        'yaw':channel('yaw', [xyz[2],band(w[2],ew),(queue[0][3],queue[0][3]),(queue[1][3],queue[1][3])]),
    }
    # Compute the exact inverse once; a zero box supplies only the SPD/inverse
    # check. Vertical image is multi-affine in b,d,ep,ev,y, so lies in the
    # convex hull of these 32 vertices, preserving the public trim correlation.
    if len(shapes['vertical']) != 5:
        raise ValueError('vertical shape must have dimension five')
    vertical_metric = prepared['vertical']
    if vertical_metric is None:
        channels['vertical'] = {'included':False,'maximum':None,'vertices_checked':0,'flight_certificate':False}
    else:
        vertices = []
        for b,d,pe,ve,y in product(*(tuple(dict.fromkeys(pair)) for pair in (battery,trim,(-ep,ep),(-ev,ev),motor))):
            vertices.append((offset[2]+pe-d/F(3,4),v[2]+ve,b*d+y,
                             queue[0][0]-hold+d,queue[1][0]-hold+d))
        maximum = max(vertical_metric.quadratic(x) for x in vertices)
        channels['vertical'] = {'included':maximum<=1,'maximum':maximum,
                                'vertices_checked':len(vertices),'vertices':vertices,'flight_certificate':False}
    return {'schema_version':'ares.current_terminal_membership.v1',
            'current_set_included':all(value['included'] for value in channels.values()),
            'channels':channels,'projective_xyz_bounds':xyz,
            'quaternion_norm_shell':'1 +/- 8*2^-24',
            'q_overapproximation_is_state_uncertainty_not_measurement_noise':True,
            'trim_force_interval':trim,'motor_residual_interval':motor,
            'hold_thrust':hold,'issued_queue':queue,
            'battery_interval':battery,'shape_convention':"E(Q)={x:x'Q^-1 x<=1}",
            'hypotheses_external':True,'hypotheses_independently_established':False,
            'rpi_proof_complete':False,'flight_certificate':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=current_terminal_membership(**json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result,indent=2,default=str)+'\n')
