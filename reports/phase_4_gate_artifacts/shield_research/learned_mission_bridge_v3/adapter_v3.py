"""Untrusted learned-command proposals; only ReanchoringShield admits motion.

Inputs are requested SI commands and public bounded packets, never plant truth.
Floating point inversion/line search here is deliberately NOT a proof checker.
"""
from itertools import product
import math
import struct
from fractions import Fraction as F
import numpy as np


def rotation(q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.dot(q,q)<=0:
        raise ValueError('finite nonzero xyzw quaternion required')
    x,y,z,w=q; n=float(q@q)
    return np.array([[w*w+x*x-y*y-z*z,2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),w*w-x*x+y*y-z*z,2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),w*w-x*x-y*y+z*z]])/n


def command_to_anchor(command, packet, hold_thrust, *, mass=.5,kp=1.5,kv=2.5,gain=6.):
    """Approximate inverse of fixed-heading backup, with yaw mismatch reported."""
    u=np.asarray(command,dtype=float);p=np.asarray(packet.p,float);v=np.asarray(packet.v,float)
    if u.shape!=(4,) or not np.isfinite(u).all() or u[0]<=0:
        raise ValueError('finite positive-thrust requested command required')
    R=rotation(packet.q)
    yaw=-math.atan2(R[1,0],R[0,0])
    fixed_yaw=gain*yaw*R[2,:]
    tilt=u[1:]-fixed_yaw
    xy=np.array([tilt[1]/gain,-tilt[0]/gain])
    if float(xy@xy)>=.95**2:
        xy*=.95/np.linalg.norm(xy)  # proposal shaping, never actuator acceptance
    fb=np.array([*xy,math.sqrt(1-float(xy@xy))])
    force=R@(u[0]/fb[2]*fb)
    anchor=p+(force-np.array([0.,0.,float(hold_thrust)])+mass*kv*v)/(mass*kp)
    return {'anchor':anchor.tolist(),'fixed_heading_yaw_mismatch_radps':float(u[3]-fixed_yaw[2]),
            'proposal_is_untrusted':True,'physical_command_not_issued':u.tolist()}


def feasible_fraction_from_public_set(membership, shapes, delta):
    """Floating candidate screen only; exact guard remains authoritative."""
    lo,hi=0.,1.
    for name,row in membership['channels'].items():
        Q=np.asarray(shapes['planar_x' if name=='planar_y' else name],float)
        P=np.linalg.inv(Q)
        if name=='vertical':vertices=np.asarray(row.get('vertices',[]),float);shift=float(delta[2])
        else:
            vertices=np.asarray(list(product(*[(float(a),float(b)) for a,b in row['state_bounds']])))
            shift=float(delta[0] if name=='planar_x' else delta[1] if name=='planar_y' else 0.)
        if not len(vertices):return None
        for x in vertices:
            a=shift*shift*P[0,0];b=-2*shift*float(P[0]@x);c=float(x@P@x)-1
            if a<1e-30:
                if abs(b)<1e-30:
                    if c>0:return None
                elif b>0:hi=min(hi,-c/b)
                else:lo=max(lo,-c/b)
            else:
                discr=b*b-4*a*c
                if discr<0:return None
                lower,upper=((-b-math.sqrt(discr))/(2*a),(-b+math.sqrt(discr))/(2*a))
                lo=max(lo,lower);hi=min(hi,upper)
            if hi<lo:return None
    if hi<0 or lo>1:return None
    # Stay inside the numeric predicted feasible interval. This is not admission.
    hi=min(1.,hi);lo=max(0.,lo)
    return hi*.98+lo*.02


def approximate_backup_word(packet,anchor,hold):
    R=rotation(packet.q);p=np.asarray(packet.p);v=np.asarray(packet.v)
    force=-.75*(p-np.asarray(anchor))-1.25*v+np.array([0.,0.,float(hold)])
    if force[2]<=0 or np.linalg.norm(force)<=0:raise ValueError('outside upright scoring branch')
    fb=R.T@force/np.linalg.norm(force);yaw=-math.atan2(R[1,0],R[0,0])
    return np.array([force@R[:,2],-6*fb[1]+6*yaw*R[2,0],6*fb[0]+6*yaw*R[2,1],6*yaw*R[2,2]])

def calibration_hold_xy_candidate(packet,old_hold,target_word,horizontal_anchor):
    """Public untrusted anchor; actual guard word is checked by the window."""
    q=tuple(F.from_float(float(x)) for x in packet.q);x,y,z,w=q;n2=sum(a*a for a in q)
    if n2<=0:raise ValueError('nonzero public orientation required')
    bz=(2*(x*z+w*y)/n2,2*(y*z-w*x)/n2,(w*w-x*x-y*y+z*z)/n2)
    if bz[2]<=0:raise ValueError('upright public orientation required')
    p=tuple(F.from_float(float(a)) for a in packet.p);v=tuple(F.from_float(float(a)) for a in packet.v)
    xy=tuple(F.from_float(float(a)) for a in horizontal_anchor)
    fx=-F(3,4)*(p[0]-xy[0])-F(5,4)*v[0];fy=-F(3,4)*(p[1]-xy[1])-F(5,4)*v[1]
    fz=(F.from_float(float(target_word))-fx*bz[0]-fy*bz[1])/bz[2]
    az=p[2]+(fz-F.from_float(float(old_hold))+F(5,4)*v[2])/F(3,4)
    return {'anchor':[float(xy[0]),float(xy[1]),float(az)],'target_word':float(target_word),
        'horizontal_hold_anchor':list(map(float,xy)),'untrusted_proposal':True,
        'exact_guard_issued_word_equality_still_required':True}

def propose_anchor(guard, packet, requested, initial_anchor,*,horizontal_fov_rad=math.radians(85)):

    base=np.asarray(guard.anchor if guard.anchor is not None else initial_anchor,float)
    try:
        intent=command_to_anchor(requested,packet,guard.hold)
        delta=np.asarray(intent['anchor'])-base
        # A fixed-heading backup cannot look sideways. When learned force
        # intent is outside its camera cone, retain lateral intent before
        # advancing longitudinally into a viewpoint that hides that space.
        R=rotation(packet.q);heading=R[:2,0];heading=heading/np.linalg.norm(heading)
        side=np.array([-heading[1],heading[0]])
        world_force=.75*(np.asarray(intent['anchor'])-np.asarray(packet.p))-1.25*np.asarray(packet.v)+np.array([0,0,float(guard.hold)])
        forward_force=float(world_force[:2]@heading);lateral_force=float(world_force[:2]@side)
        coverage_first=abs(lateral_force)>math.tan(horizontal_fov_rad/2)*max(abs(forward_force),.01)
        if coverage_first and delta[:2]@heading>0:
            delta[:2]-=heading*float(delta[:2]@heading)
        current=guard._membership(packet,tuple(base))
        fraction=feasible_fraction_from_public_set(current,guard.proof.shapes,delta)
        proposed=base if fraction is None else base+fraction*delta
        # Unknown space can block the diagonal intention while a component
        # still approaches the same requested physical command. No waypoint or
        # true geometry enters this finite candidate search.
        candidates=[]
        for mask in [(1,1,1),(0,1,1),(1,0,1),(1,1,0),(1,0,0),(0,1,0),(0,0,1)]:
            direction=delta*np.asarray(mask)
            part=feasible_fraction_from_public_set(current,guard.proof.shapes,direction)
            if part is None:continue
            candidate=base+part*direction
            if not guard._geometry(tuple(candidate))['reserved']:continue
            try:
                word=approximate_backup_word(packet,candidate,guard.hold)
                cost=float(np.sum(((np.asarray(word)-np.asarray(requested))/[5.,6.,6.,6.])**2))
                candidates.append((cost,candidate,part,mask))
            except (ValueError,ArithmeticError):continue
        selected_mask=None
        if candidates:
            _,proposed,fraction,selected_mask=min(candidates,key=lambda item:item[0])
        return {'coverage_first_lateral_intent':bool(coverage_first),'component_mask':selected_mask,'candidate_anchor':proposed.tolist(),'intent_anchor':intent['anchor'],
                'line_fraction':fraction,'candidate_screen_is_not_admission':True,
                'fixed_heading_yaw_mismatch_radps':intent['fixed_heading_yaw_mismatch_radps']}
    except (ValueError,ArithmeticError,FloatingPointError,np.linalg.LinAlgError) as exc:
        return {'candidate_anchor':base.tolist(),'intent_anchor':None,'line_fraction':None,
                'candidate_screen_is_not_admission':True,'proposal_error':type(exc).__name__+': '+str(exc)}


def word_intervention(requested, issued):
    """True whenever actual binary32 physical command words differ."""
    pack=lambda u:tuple(struct.pack('>f',float(x)) for x in u)
    return pack(requested)!=pack(issued)
