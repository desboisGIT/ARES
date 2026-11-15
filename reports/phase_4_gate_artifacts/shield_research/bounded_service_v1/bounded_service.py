"""Reusable public measurement/map interfaces for a conditional SIL fixture.

Only the supplier imports the SIL renderer. MapBuilder and reservation consumers
accept public packets, never a renderer, plant, profile sample or truth map.
"""
from dataclasses import asdict,dataclass
from fractions import Fraction as F
import math
from numbers import Real

from ares.safety_contract import MeasurementContract, robust_free_witness, state_boxes, validate_contract

VERSION = 'bounded_service_v1'
# Exact represented radii must not exceed the separate rational proof bounds.
POSITION_ERROR_M = math.nextafter(.005,0.)
VELOCITY_ERROR_MPS = math.nextafter(.01,0.)
ANGULAR_VELOCITY_ERROR_RADPS = math.nextafter(.001,0.)


def _real(value, name):
    if isinstance(value,bool) or not isinstance(value,Real):
        raise ValueError(f'{name} requires a finite numeric value, not bool')
    try:
        value=float(value)
    except (ValueError,OverflowError) as error:
        raise ValueError(f'{name} requires a finite binary64 value') from error
    if not math.isfinite(value):raise ValueError(f'{name} must be finite')
    return value


def _vector(value, width, name):
    try:values=tuple(value)
    except TypeError as error:raise ValueError(f'{name} must have {width} coordinates') from error
    if len(values)!=width:raise ValueError(f'{name} must have {width} coordinates')
    return tuple(_real(v,name) for v in values)


def _rounded(value, *, upper):
    result=float(value)
    if not math.isfinite(result):raise OverflowError('finite bound cannot be represented')
    while (F.from_float(result)<value if upper else F.from_float(result)>value):
        result=math.nextafter(result,math.inf if upper else -math.inf)
        if not math.isfinite(result):raise OverflowError('finite bound cannot be represented')
    return result


@dataclass(frozen=True)
class StatePacket:
    p: tuple
    v: tuple
    q: tuple
    contract: MeasurementContract
    w: tuple = (0.,0.,0.)
    angular_velocity_error_radps: float = ANGULAR_VELOCITY_ERROR_RADPS
    orientation_contract: str = 'exact_current_raw_quaternion'

    def __post_init__(self):
        if self.orientation_contract!='exact_current_raw_quaternion':
            raise ValueError('unsupported orientation contract')
        for name,width in (('p',3),('v',3),('w',3),('q',4)):
            object.__setattr__(self,name,_vector(getattr(self,name),width,name))
        error=_real(self.angular_velocity_error_radps,'angular_velocity_error_radps')
        if error<0:raise ValueError('angular velocity error must be nonnegative')
        object.__setattr__(self,'angular_velocity_error_radps',error)
        norm2=sum(F.from_float(a)**2 for a in self.q)
        shell=8*F(1,2**24) # Eight binary32 unit roundoffs on quaternion norm.
        if norm2==0 or not (1-shell)**2<=norm2<=(1+shell)**2:
            raise ValueError('q must be nonzero and within the represented unit-quaternion shell')
        if not isinstance(self.contract,MeasurementContract):raise ValueError('explicit measurement contract required')

    def body_rate_box(self, *, time_s):
        """Capture-epoch outer body-rate box from separate fixture extra."""
        status=validate_contract(self.contract,time_s=time_s)
        if not status.conditional_input_valid:
            return {'conditional_input_valid':False,'reason':status.reason,'flight_certificate':False}
        error=F.from_float(self.angular_velocity_error_radps)
        try:
            lower=tuple(_rounded(F.from_float(v)-error,upper=False) for v in self.w)
            upper=tuple(_rounded(F.from_float(v)+error,upper=True) for v in self.w)
        except OverflowError:
            return {'conditional_input_valid':False,'reason':'nonfinite_outward_bounds','flight_certificate':False}
        return {'conditional_input_valid':True,'lower':lower,'upper':upper,
                'capture_time_s':self.contract.capture_time_s,
                'angular_velocity_error_radps':self.angular_velocity_error_radps,
                'flight_certificate':False}

    def boxes(self, *, time_s):
        return state_boxes(self.p,self.v,self.contract,time_s=time_s)


@dataclass(frozen=True)
class RayPacket:
    state: StatePacket
    world_direction: tuple
    measured_depth: float
    ray_id: int


class BoundedSupplier:
    """SIL sensor boundary. Controllers receive returned packet values only."""
    def __init__(self, renderer, truth, *, seed=71):
        self._renderer,self._truth,self.seed = renderer,truth,seed
        self._issued=set()

    def capture(self, *, capture_time_s, expires_s, position_corner=None, velocity_corner=None, angular_velocity_corner=None):
        from sensor_renderer import noisy_vector
        c = MeasurementContract(position_error_m=POSITION_ERROR_M,velocity_error_mps=VELOCITY_ERROR_MPS,
            world_ray_error=.001,depth_error_m=.01,capture_time_s=capture_time_s,expires_s=expires_s,
            supplier=VERSION,provenance=f'{VERSION}:exact_rational_box_radius_over_3:seed={self.seed}',
            source_kind='hard_external',bound_basis='deterministic_guarantee',
            voxel_model='homogeneous_static_until_expiry',depth_model='first_hit_range')
        packet = StatePacket(noisy_vector(self._truth.position,c.position_error_m,f'{self.seed}:position',position_corner),
                           noisy_vector(self._truth.velocity,c.velocity_error_mps,f'{self.seed}:velocity',velocity_corner),
                           tuple(self._truth.quaternion),c,
                           noisy_vector(self._truth.angular_velocity,ANGULAR_VELOCITY_ERROR_RADPS,f'{self.seed}:angular_velocity',angular_velocity_corner))
        self._issued.add(packet)
        return packet

    def ray(self, state, direction, *, ray_id, ray_corner=None, depth_sign=None):
        if state not in self._issued:
            raise ValueError('packet was not issued by this fixture')
        # All captures from this supplier represent the same static preflight pose.
        # A moving integration must create a new supplier/epoch for each truth state.
        depth = self._renderer.render_range(self._truth.position,direction,
            seed=f'{self.seed}:{ray_id}',ray_corner=ray_corner,depth_sign=depth_sign)
        return RayPacket(state,tuple(direction),depth,ray_id)


class WitnessMap:
    """Known-free cells with retained ray witnesses; absence always means unknown."""
    def __init__(self, *, shape=(48,48,16), resolution=.25, origin=(0.,0.,0.)):
        try:self.shape=tuple(shape)
        except TypeError as error:raise ValueError('positive three-dimensional integer grid required') from error
        if len(self.shape)!=3 or any(type(n) is not int or n<=0 for n in self.shape):
            raise ValueError('positive three-dimensional integer grid required')
        self.resolution=_real(resolution,'resolution')
        if self.resolution<=0:raise ValueError('resolution must be positive')
        self.origin=_vector(origin,3,'origin')
        for o,n in zip(self.origin,self.shape):
            try:upper=o+n*self.resolution
            except OverflowError as error:raise ValueError('grid support is not representable') from error
            if not math.isfinite(upper) or o+self.resolution<=o or upper-self.resolution>=upper:
                raise ValueError('finite distinguishable voxel boundaries required')
        self.free = {}
        self.rejections = {}

    def observe(self, ray, cell, *, optical_range, time_s):
        cell = tuple(cell)
        if len(cell)!=3 or any(type(c) is not int or not 0<=c<n for c,n in zip(cell,self.shape)):
            raise ValueError('cell must lie inside public grid support')
        # Strict crossing is checked inside an inward-rounded sub-box of the
        # exact represented grid voxel, so non-dyadic arithmetic cannot enlarge it.
        exact_lower=tuple(F.from_float(o)+i*F.from_float(self.resolution) for o,i in zip(self.origin,cell))
        lower=tuple(_rounded(a,upper=True) for a in exact_lower)
        upper=tuple(_rounded(a+F.from_float(self.resolution),upper=False) for a in exact_lower)
        witness = robust_free_witness(ray.state.p,ray.world_direction,ray.measured_depth,
            lower,upper,optical_range,ray.state.contract,time_s=time_s)
        if witness.free_witness:
            self.free[cell] = (ray,witness,optical_range)
        else:
            self.rejections[cell] = witness.reason
        return witness

    def known_free(self, cell, *, time_s):
        entry = self.free.get(tuple(cell))
        return bool(entry and validate_contract(entry[0].state.contract,time_s=time_s).conditional_input_valid)

    def reservation(self, center, radius, *, time_s):
        """Static capture-epoch ball reservation only; no reachability certificate."""
        center=_vector(center,3,'center');radius=_real(radius,'radius')
        if radius<=0:raise ValueError('finite positive sphere radius required')
        p=tuple(F.from_float(v) for v in center);r=F.from_float(radius)
        res=F.from_float(float(self.resolution));origin=tuple(F.from_float(float(o)) for o in self.origin)
        lower=[math.ceil((a-r-o)/res)-1 for a,o in zip(p,origin)]
        upper=[math.floor((a+r-o)/res) for a,o in zip(p,origin)]
        missing=[];needed=[]
        from itertools import product
        for cell in product(*(range(a,b+1) for a,b in zip(lower,upper))):
            lo=tuple(o+i*res for o,i in zip(origin,cell));hi=tuple(a+res for a in lo)
            distance2=sum(max(lo_i-a,F(0),a-hi_i)**2 for a,lo_i,hi_i in zip(p,lo,hi))
            if distance2<=r*r:
                needed.append(cell)
                if not self.known_free(cell,time_s=time_s):missing.append(cell)
        return {'conditional_reservation_free':not missing,'required_cell_count':len(needed),
                'missing_cells':missing,'capture_epoch_only':True,'flight_certificate':False}

    @classmethod
    def from_dict(cls, data, *, time_s):
        """Reload only public rays, rechecking every witness; never trust labels."""
        if data.get('version')!=VERSION or data.get('flight_certificate') is not False:
            raise ValueError('unsupported conditional fixture map')
        result=cls(shape=data['shape'],resolution=data['resolution'],origin=data['origin'])
        for entry in data['known_free']:
            raw=entry['packet'];rawstate=dict(raw['state'])
            if rawstate.get('orientation_contract')!='exact_current_raw_quaternion':
                raise ValueError('missing or unsupported serialized orientation contract')
            rawstate['contract']=MeasurementContract(**rawstate['contract'])
            for name in ('p','v','q','w'):
                if name in rawstate:rawstate[name]=tuple(rawstate[name])
            ray=RayPacket(StatePacket(**rawstate),tuple(raw['world_direction']),raw['measured_depth'],raw['ray_id'])
            result.observe(ray,entry['cell'],optical_range=entry['optical_range'],time_s=time_s)
        return result

    def to_dict(self):
        return {'version':VERSION,'shape':self.shape,'resolution':self.resolution,'origin':self.origin,
                'conditional_fixture_only':True,'flight_certificate':False,
                'known_free':[{'cell':cell,'ray_id':ray.ray_id,'optical_range':s,
                    'packet':asdict(ray),'witness':asdict(witness)} for cell,(ray,witness,s) in sorted(self.free.items())],
                'rejections':[{'cell':cell,'reason':reason} for cell,reason in sorted(self.rejections.items())]}


def preflight_scan(supplier, state, *, time_s, half_width_cells=6):
    """Enumerate PUBLIC target centers; free labels come only from ray witnesses.

    The x=7..7.25 m target slab is deliberately unscanned/unknown. The renderer's
    occupied x=9..9.25 slab is queried adversarially, never copied into this map.
    """
    from itertools import product
    result=WitnessMap()
    anchor=tuple(math.floor((p-o)/result.resolution) for p,o in zip(state.p,result.origin))
    candidates=set(product(*(range(max(0,a-half_width_cells),min(n,a+half_width_cells+1))
                             for a,n in zip(anchor,result.shape))))
    candidates.update(((0,24,8),(47,24,8),(36,24,8),(40,24,8)))
    for ray_id,cell in enumerate(sorted(candidates)):
        if cell[0]==28:continue
        center=tuple(o+(i+.5)*result.resolution for o,i in zip(result.origin,cell))
        direction=tuple(c-p for c,p in zip(center,state.p))
        if sum(d*d for d in direction)<1e-20:continue
        ray=supplier.ray(state,direction,ray_id=ray_id)
        result.observe(ray,cell,optical_range=1.,time_s=time_s)
    return result
