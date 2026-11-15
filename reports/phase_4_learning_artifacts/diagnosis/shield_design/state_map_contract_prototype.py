"""Report-only geometric bounded-contract prototype; no runtime certificates.

No scene, truth, sensor-noise draw, plant parameter sample or flight device enters
this API. Bounds are EXTERNAL ASSUMPTIONS, not inferred from innovation tests.
The geometry has a real-arithmetic proof. This floating-point research prototype
is not a validated numerical implementation and must not set certified=True.
"""
from dataclasses import dataclass
import math
import unittest
import numpy as np


@dataclass(frozen=True)
class MeasurementContract:
    position_error_m: float
    velocity_error_mps: float
    orientation_error_rad: float
    optical_depth_error_m: float
    ray_direction_error: float
    capture_time_s: float
    expires_s: float
    source: str

    def __post_init__(self):
        for key,value in vars(self).items():
            if key=='source':
                if not isinstance(value,str) or not value.strip():
                    raise ValueError('explicit external contract provenance required')
            elif not math.isfinite(value):
                raise ValueError('finite bounds and times required')
        if min(self.position_error_m,self.velocity_error_mps,self.orientation_error_rad,
               self.optical_depth_error_m,self.ray_direction_error)<0:
            raise ValueError('error bounds must be nonnegative')
        if self.expires_s<self.capture_time_s or self.orientation_error_rad>math.pi:
            raise ValueError('invalid validity interval or orientation bound')

    def require_live(self,time_s):
        if not math.isfinite(time_s) or not self.capture_time_s<=time_s<=self.expires_s:
            raise ValueError('contract is not live at requested time')


def vector(value):
    value=np.asarray(value,dtype=float)
    if value.shape!=(3,) or not np.isfinite(value).all():
        raise ValueError('three finite coordinates required')
    return value


def state_boxes(position,velocity,contract,*,time_s):
    """Outer boxes of supplied Euclidean error balls, conditional on the contract."""
    contract.require_live(time_s)
    p,v=vector(position),vector(velocity)
    return {'position_lower':p-contract.position_error_m,
            'position_upper':p+contract.position_error_m,
            'velocity_lower':v-contract.velocity_error_mps,
            'velocity_upper':v+contract.velocity_error_mps,
            'conditional_source':contract.source,'runtime_certificate':False}


def robust_free_witness(position,rotation,direction,measured_depth,voxel_lower,
                        voxel_upper,optical_range,contract,*,time_s):
    """A sufficient robust crossing witness for a homogeneous occupancy voxel.

    All physically possible rays must pass through the voxel interior at the
    common optical range s, strictly before any possible first hit. Missing or
    stale contracts are rejected. A missing witness means unknown, never free.
    """
    contract.require_live(time_s)
    p,d,lo,hi=map(vector,(position,direction,voxel_lower,voxel_upper))
    r=np.asarray(rotation,dtype=float)
    if r.shape!=(3,3) or not np.isfinite(r).all() or not np.allclose(r.T@r,np.eye(3),atol=1e-10):
        raise ValueError('nominal proper rotation required')
    if np.linalg.det(r)<0 or np.any(lo>=hi):
        raise ValueError('proper rotation and positive voxel dimensions required')
    if not math.isfinite(measured_depth) or not math.isfinite(optical_range):
        return {'free':False,'reason':'unavailable_depth','runtime_certificate':False}
    s=optical_range
    if s<=0 or s>=measured_depth-contract.optical_depth_error_m:
        return {'free':False,'reason':'range_not_strictly_before_first_hit','runtime_certificate':False}
    radius=(contract.position_error_m+s*(contract.ray_direction_error+
              2*math.sin(contract.orientation_error_rad/2)*np.linalg.norm(d)))
    center=p+r@d*s
    slack=float(min(np.min(center-radius-lo),np.min(hi-center-radius)))
    return {'free':slack>0,'reason':'conditional_robust_crossing' if slack>0 else 'crossing_uncertain',
            'nominal_point':center.tolist(),'ray_point_uncertainty_radius_m':radius,
            'minimum_interior_slack_m':slack,'conditional_source':contract.source,
            'runtime_certificate':False}


class ContractGeometryTests(unittest.TestCase):
    def contract(self,**changes):
        args=dict(position_error_m=.005,velocity_error_mps=.01,orientation_error_rad=.002,
                  optical_depth_error_m=.02,ray_direction_error=.001,capture_time_s=1.,
                  expires_s=1.1,source='explicit bounded simulation assumption')
        args.update(changes)
        return MeasurementContract(**args)

    def test_positive_3d_ray_witness_and_adversarial_perturbations(self):
        p=np.array([.5,.5,.5]);d=np.array([1.,.2,-.1]);s=1.;point=p+d*s
        lo,hi=point-.1,point+.1;c=self.contract()
        w=robust_free_witness(p,np.eye(3),d,2.,lo,hi,s,c,time_s=1.05)
        self.assertTrue(w['free']);self.assertFalse(w['runtime_certificate'])
        # Corners of three-dimensional pose/ray perturbations, plus nonzero
        # rotations about each axis. These are regression cases, not the proof.
        import itertools
        for signs in itertools.product([-1,1],repeat=3):
            e=np.array(signs)/3**.5
            for axis in np.eye(3):
                angle=c.orientation_error_rad;k=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
                r=np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k)
                physical=p+c.position_error_m*e+r@(d+c.ray_direction_error*e)*s
                self.assertTrue(np.all(physical>lo) and np.all(physical<hi))

    def test_same_nominal_ray_does_not_certify_uncertain_voxel_crossing(self):
        c=self.contract(position_error_m=.11)
        w=robust_free_witness([0,0,0],np.eye(3),[1,0,0],2.,[.9,-.1,-.1],[1.1,.1,.1],1.,c,time_s=1.)
        self.assertFalse(w['free'])

    def test_endpoint_depth_unknown_and_stale_contracts_reject(self):
        args=([0,0,0],np.eye(3),[1,0,0],1.,[.9,-.1,-.1],[1.1,.1,.1],1.,self.contract())
        self.assertFalse(robust_free_witness(*args,time_s=1.)['free'])
        with self.assertRaises(ValueError):robust_free_witness(*args,time_s=1.2)
        with self.assertRaises(ValueError):self.contract(source='')
        with self.assertRaises(ValueError):self.contract(velocity_error_mps=float('nan'))

    def test_state_ball_containment_and_no_implicit_truth_certificate(self):
        boxes=state_boxes([1,2,3],[.1,-.1,.05],self.contract(),time_s=1.)
        self.assertTrue(np.all(boxes['position_lower']<boxes['position_upper']))
        self.assertFalse(boxes['runtime_certificate'])


if __name__=='__main__':unittest.main()
