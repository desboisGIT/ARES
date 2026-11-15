"""Exact-interval point backup commands, research only; no motion certificate.

Fixed-heading local PD backup, not the learned policy. The real geometric law is
enclosed using rational arithmetic, validated square roots and atan series.
Each emitted binary32 command has an exact absolute-error witness. This removes
libm/BLAS assumptions for command evaluation, not for the physical plant.
"""
from fractions import Fraction as F
import math
import struct


class I:
    def __init__(self,lo,hi=None):
        self.lo=lo if isinstance(lo,F) else F.from_float(float(lo))
        self.hi=self.lo if hi is None else (hi if isinstance(hi,F) else F.from_float(float(hi)))
        if self.lo>self.hi:raise ValueError('reversed interval')
    @staticmethod
    def cast(x):return x if isinstance(x,I) else I(x)
    def __add__(self,x):
        x=self.cast(x);return I(self.lo+x.lo,self.hi+x.hi)
    __radd__=__add__
    def __neg__(self):return I(-self.hi,-self.lo)
    def __sub__(self,x):return self+-self.cast(x)
    def __rsub__(self,x):return self.cast(x)+-self
    def __mul__(self,x):
        x=self.cast(x);values=[a*b for a in (self.lo,self.hi) for b in (x.lo,x.hi)]
        return I(min(values),max(values))
    __rmul__=__mul__
    def __truediv__(self,x):
        x=self.cast(x)
        if x.lo<=0<=x.hi:raise ValueError('denominator contains zero')
        return self*I(1/x.hi,1/x.lo)
    def sqrt(self):
        if self.lo<0:raise ValueError('negative radicand')
        scale=2**90
        def bounds(x):
            lower=F(math.isqrt(x.numerator*scale*scale//x.denominator),scale)
            upper=lower if lower*lower==x else lower+F(1,scale)
            assert lower*lower<=x<=upper*upper
            return lower,upper
        return I(bounds(self.lo)[0],bounds(self.hi)[1])
    def atan(self):
        if max(abs(self.lo),abs(self.hi))>F(1,2):
            raise ValueError('heading outside local atan domain')
        def bounds(x):
            if x<0:
                lo,hi=bounds(-x);return -hi,-lo
            total=F(0);power=x;square=x*x
            for k in range(33):
                total+=power/(2*k+1) if k%2==0 else -power/(2*k+1)
                following=power*square;remainder=following/(2*k+3)
                lo,hi=(total-remainder,total) if k%2==0 else (total,total+remainder)
                if hi-lo<=F(1,10**12) or k==32:return lo,hi
                power=following
        return I(bounds(self.lo)[0],bounds(self.hi)[1])


def output32(interval,tolerance=1e-6):
    candidate=struct.unpack('>f',struct.pack('>f',float((interval.lo+interval.hi)/2)))[0]
    error=max(abs(F.from_float(candidate)-interval.lo),abs(F.from_float(candidate)-interval.hi))
    if error>F.from_float(tolerance):raise ArithmeticError('command error budget exceeded')
    return candidate,error


def backup_point(position,velocity,quaternion,anchor,hover_thrust,*,
                 mass=.5,kp=1.5,kv=2.5,attitude_gain=6.):
    if any(not math.isfinite(float(a)) or a<=0 for a in (mass,kp,kv,attitude_gain,hover_thrust)):
        raise ValueError('finite positive controller constants required')
    if any(not math.isfinite(float(x)) for seq in (position,velocity,quaternion,anchor,[hover_thrust]) for x in seq):
        raise ValueError('finite inputs required')
    if not (len(position)==len(velocity)==len(anchor)==3 and len(quaternion)==4):
        raise ValueError('three-state vectors and xyzw quaternion required')
    p,v,q,target=([I(x) for x in seq] for seq in (position,velocity,quaternion,anchor))
    x,y,z,w=q
    norm2=x*x+y*y+z*z+w*w
    if norm2.lo<=0 or w.lo<=0:raise ValueError('positive-qw chart required')
    # R(q/|q|)=R_homogeneous(q)/|q|²: an exact rational rotation, no sqrt(q).
    rotation=[[w*w+x*x-y*y-z*z,2*(x*y-w*z),2*(x*z+w*y)],
              [2*(x*y+w*z),w*w-x*x+y*y-z*z,2*(y*z-w*x)],
              [2*(x*z-w*y),2*(y*z+w*x),w*w-x*x-y*y+z*z]]
    rotation=[[a/norm2 for a in row] for row in rotation]
    if rotation[0][0].lo<=0:
        raise ValueError('positive local heading branch required')
    force=[mass*(-kp*(p[i]-target[i])-kv*v[i]) for i in range(3)]
    force[2]=force[2]+hover_thrust
    if force[2].lo<=0:raise ValueError('positive upright force domain required')
    norm=sum(a*a for a in force).sqrt()
    bodyforce=[sum(rotation[j][i]*force[j] for j in range(3))/norm for i in range(3)]
    yaw=-(rotation[1][0]/rotation[0][0]).atan()
    if max(abs(yaw.lo),abs(yaw.hi))*attitude_gain>=1:
        raise ValueError('yaw contribution would saturate')
    rates=[-attitude_gain*bodyforce[1]+attitude_gain*yaw*rotation[2][0],
            attitude_gain*bodyforce[0]+attitude_gain*yaw*rotation[2][1],
            attitude_gain*yaw*rotation[2][2]]
    thrust=sum(force[i]*rotation[i][2] for i in range(3))
    if thrust.lo<=0 or thrust.hi>=F(49,2) or any(max(abs(a.lo),abs(a.hi))>=6 for a in rates):
        raise ValueError('actuator clamp would activate outside proof domain')
    pairs=[output32(a) for a in [thrust,*rates]]
    return {'command':[a for a,e in pairs],'exact_error':[e for a,e in pairs],
            'uniform_error_budget':1e-6,'command_roundoff_enclosed':True,
            'flight_certificate':False}


if __name__=='__main__':
    import json
    from pathlib import Path
    import torch
    from ares.flight_control import ControllerConfig,acceleration_to_action
    torch.set_num_threads(1)
    cases=0;largest=0.
    for sign in (-1,1):
        for k in range(10):
            p=[sign*.003*k,-.002*k,.001*k];v=[-.001*k,sign*.002*k,-.001*k]
            q=[.001*k,-.0007*k,.0004*k,1.];anchor=[0.,0.,0.];hold=4.5+.1*k
            result=backup_point(p,v,q,anchor,hold)
            observation={'p':torch.tensor([p],dtype=torch.float64),
                'v':torch.tensor([v],dtype=torch.float64),
                'q':torch.tensor([q],dtype=torch.float64),
                'gaze':torch.tensor([[p[0]+1,p[1],p[2]]],dtype=torch.float64),
                'controller_bias':torch.tensor([[0.,0.,9.80665-hold/.5]],dtype=torch.float64)}
            observation['q']=observation['q']/observation['q'].norm(dim=-1,keepdim=True)
            reference=acceleration_to_action(observation,-1.5*observation['p']-2.5*observation['v'],ControllerConfig())
            for actual,expected,error in zip(result['command'],reference[0].tolist(),result['exact_error']):
                assert abs(actual-expected)<=float(error)+2e-13
                largest=max(largest,float(error))
            cases+=1
    for bad_q in ([0.,0.,0.,0.],[0.,0.,1.,.1]):
        try:backup_point([0.,0.,0.],[0.,0.,0.],bad_q,[0.,0.,0.],5.)
        except ValueError:pass
        else:raise AssertionError('unsupported quaternion branch accepted')
    result={'valid_point_cases':cases,'maximum_certified_command_error':largest,
        'primitive_interval_law_proved':True,'all_error_checks_exact':True,
        'model_or_reachability_certificate':False,'flight_certificate':False}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
