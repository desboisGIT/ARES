"""SIL-only truth boundary. Never hand these objects to a controller/guard.

The deterministic sensor guarantee is enforced with exact rational checks on
represented operands. This is an analytic fixture, not a hardware guarantee.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
import math
import random


def exact(x):
    return F.from_float(float(x))


def bounded_error(bound, seed, corner=None):
    """Three signed components <= E/3; squared norm <= E²/3 < E²."""
    limit = exact(bound)
    rng = random.Random(str(seed))
    signs = corner if corner is not None else [rng.choice((-1,1)) for _ in range(3)]
    if len(signs) != 3 or any(s not in (-1,1) for s in signs):
        raise ValueError('corner must have three signs')
    magnitudes = [F(1) if corner is not None else F(rng.randint(1,1024),1024) for _ in range(3)]
    return tuple(limit*s*m/F(3) for s,m in zip(signs,magnitudes))


def noisy_vector(truth, bound, seed, corner=None):
    nominal = tuple(exact(x) for x in truth)
    perturbation = bounded_error(bound,seed,corner)
    measured = tuple(float(a+b) for a,b in zip(nominal,perturbation))
    actual_error = tuple(exact(b)-a for a,b in zip(nominal,measured))
    # Covers the binary64 rounding of the returned representation too.
    if sum(e*e for e in actual_error) > exact(bound)**2:
        raise ValueError('represented sensor error exceeds the declared Euclidean bound')
    return measured


@dataclass(frozen=True)
class TruthState:
    position: tuple
    velocity: tuple
    quaternion: tuple = (0.,0.,0.,1.)
    angular_velocity: tuple = (0.,0.,0.)

    def __post_init__(self):
        if len(self.position)!=3 or len(self.velocity)!=3 or len(self.quaternion)!=4 or len(self.angular_velocity)!=3:
            raise ValueError('physical state widths must be p3/v3/q4/w3')
        if not all(math.isfinite(float(x)) for v in (self.position,self.velocity,self.quaternion,self.angular_velocity) for x in v):
            raise ValueError('finite physical state required')
        if abs(sum(float(x)**2 for x in self.quaternion)-1.) > 1e-5:
            raise ValueError('fixture quaternion must be unit length')


class AnalyticRenderer:
    """12×12×4 m homogeneous static .25 m voxels and first-hit rays.

    Closed occupied boundary walls and an interior full-span x=[9,9.25] slab.
    The origin must be in a free voxel. Range parameter is along the supplied
    possibly nonunit world direction, with exact slab intersections.
    """
    shape = (48,48,16)
    resolution = .25
    origin = (0.,0.,0.)

    def occupied(self, cell):
        x,y,z = cell
        return (not (0<=x<48 and 0<=y<48 and 0<=z<16)
                or x in (0,47,36) or y in (0,47) or z in (0,15))

    def _boxes(self):
        return (((0,0,0),(.25,12,4)),((11.75,0,0),(12,12,4)),
                ((0,0,0),(12,.25,4)),((0,11.75,0),(12,12,4)),
                ((0,0,0),(12,12,.25)),((0,0,3.75),(12,12,4)),
                ((9,0,0),(9.25,12,4)))

    def first_hit(self, position, direction):
        p,d = tuple(map(exact,position)),tuple(map(exact,direction))
        if not any(d): raise ValueError('nonzero world ray required')
        cell = tuple(math.floor(a/exact(self.resolution)) for a in p)
        if self.occupied(cell): raise ValueError('sensor origin must be physically free')
        hits = []
        for lower,upper in self._boxes():
            near,far = F(0),None
            for a,v,lo,hi in zip(p,d,map(exact,lower),map(exact,upper)):
                if v == 0:
                    if not lo <= a <= hi: break
                else:
                    enter,leave = sorted(((lo-a)/v,(hi-a)/v))
                    near = max(near,enter)
                    far = leave if far is None else min(far,leave)
                    if far < near: break
            else:
                if far is not None and near <= far and far >= 0: hits.append(near)
        if not hits: raise ValueError('no first hit in enclosed analytic room')
        return min(hits)

    def render_range(self, position, reported_direction, *, seed, ray_corner=None, depth_sign=None):
        # Actual physical direction differs nontrivially from reported direction.
        actual_direction = noisy_vector(reported_direction,.001,f'{seed}:ray',ray_corner)
        hit = self.first_hit(position,actual_direction)
        rng = random.Random(f'{seed}:depth')
        sign = rng.choice((-1,1)) if depth_sign is None else depth_sign
        if sign not in (-1,1): raise ValueError('depth sign must be ±1')
        measured = float(hit + exact(.01)*sign/F(3))
        if abs(exact(measured)-hit) > exact(.01):
            raise ValueError('represented depth error exceeds guarantee')
        return measured


class TestOracle:
    """Test-only access to physical truth; no truth is returned to map builders."""
    @staticmethod
    def free_map_valid(renderer, cells):
        return all(not renderer.occupied(tuple(c)) for c in cells)

    @staticmethod
    def state_error_valid(truth, packet):
        for actual,measured,bound in ((truth.position,packet.p,packet.contract.position_error_m),
                                      (truth.velocity,packet.v,packet.contract.velocity_error_mps),
                                      (truth.angular_velocity,packet.w,packet.angular_velocity_error_radps)):
            if sum((exact(b)-exact(a))**2 for a,b in zip(actual,measured)) > exact(bound)**2:
                return False
        return packet.q == truth.quaternion
