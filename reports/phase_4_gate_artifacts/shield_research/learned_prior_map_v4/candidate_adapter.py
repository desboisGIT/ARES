"""UNTRUSTED fast physical-command intent; only exact guard admits motion."""
from itertools import product
from functools import lru_cache
from pathlib import Path
import sys,math
import numpy as np
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'learned_command_bridge_v1'))
from adapter import command_to_anchor,rotation
@lru_cache(maxsize=8)
def inverse_shape(shape):return np.linalg.inv(np.array(shape))
def screen_data(guard,packet,base):
    p=np.asarray(packet.p)-base;v=np.asarray(packet.v);w=np.asarray(packet.w)
    q=np.asarray(packet.q);q=q/np.linalg.norm(q);queue=np.asarray(guard.queue,dtype=float)
    band=lambda c,e:(c-e,c+e)
    rows=[('planar_x',list(product(band(p[0],.005),band(v[0],.01),band(q[1],1e-6),band(w[1],.001),(queue[0,2],),(queue[1,2],)))),
      ('planar_y',list(product(band(p[1],.005),band(v[1],.01),band(-q[0],1e-6),band(-w[0],.001),(-queue[0,1],),(-queue[1,1],)))),
      ('yaw',list(product(band(q[2],1e-6),band(w[2],.001),(queue[0,3],),(queue[1,3],))))]
    vert=[]
    for b,d,pe,ve,y in product((.95,1.),map(float,guard.trim),(-.005,.005),(-.01,.01),map(float,guard.motor)):
        vert.append([p[2]+pe-d/.75,v[2]+ve,b*d+y,queue[0,0]-float(guard.hold)+d,queue[1,0]-float(guard.hold)+d])
    rows.append(('vertical',vert));out=[]
    for name,vertices in rows:
        shape=tuple(tuple(float(x) for x in row) for row in guard.proof.shapes['planar_x' if name=='planar_y' else name])
        P=inverse_shape(shape);X=np.array(vertices);out.append((name,float(P[0,0]),X@P[:,0],np.einsum('ni,ij,nj->n',X,P,X)-1))
    return out

def fraction(data,delta):
    lo,hi=0.,1.
    for name,p00,linear,c in data:
        shift=float(delta[0] if name=='planar_x' else delta[1] if name=='planar_y' else delta[2] if name=='vertical' else 0.)
        if abs(shift)<1e-15:
            if np.max(c)>0:return None
            continue
        a=shift*shift*p00;b=-2*shift*linear;disc=b*b-4*a*c
        if np.min(disc)<0:return None
        root=np.sqrt(np.maximum(disc,0));lo=max(lo,float(np.max((-b-root)/(2*a))));hi=min(hi,float(np.min((-b+root)/(2*a))))
        if hi<lo:return None
    lo=max(0.,lo);hi=min(1.,hi)
    return None if hi<lo else .98*hi+.02*lo

def approximate_word(packet,anchor,hold):
    R=rotation(packet.q);force=-.75*(np.asarray(packet.p)-anchor)-1.25*np.asarray(packet.v)+[0,0,float(hold)]
    if force[2]<=0:raise ValueError('scoring branch')
    fb=R.T@force/np.linalg.norm(force);yaw=-math.atan2(R[1,0],R[0,0])
    return np.array([force@R[:,2],-6*fb[1]+6*yaw*R[2,0],6*fb[0]+6*yaw*R[2,1],6*yaw*R[2,2]])

def propose_anchor(guard,packet,requested,initial_anchor):
    base=np.asarray(guard.anchor if guard.anchor is not None else initial_anchor,float)
    try:
        intent=command_to_anchor(requested,packet,guard.hold);delta=np.asarray(intent['anchor'])-base
        data=screen_data(guard,packet,base);candidates=[]
        for mask in [(1,1,1),(0,1,1),(1,0,1),(1,1,0),(1,0,0),(0,1,0),(0,0,1)]:
            direction=delta*np.asarray(mask);part=fraction(data,direction)
            if part is None:continue
            candidate=base+part*direction
            if not guard._geometry(tuple(candidate))['reserved']:continue
            word=approximate_word(packet,candidate,guard.hold)
            cost=float(np.sum(((word-np.asarray(requested))/[5,6,6,6])**2));candidates.append((cost,candidate,part,mask))
        if candidates:_,candidate,part,mask=min(candidates,key=lambda x:x[0])
        else:candidate=base;part=None;mask=None
        return {'candidate_anchor':candidate.tolist(),'intent_anchor':intent['anchor'],'line_fraction':part,'component_mask':mask,
          'floating_screen_is_not_admission':True,'verified_point_kernel_called_only_by_guard':True,
          'fixed_heading_yaw_mismatch_radps':intent['fixed_heading_yaw_mismatch_radps']}
    except (ValueError,ArithmeticError,FloatingPointError,np.linalg.LinAlgError) as exc:
        return {'candidate_anchor':base.tolist(),'intent_anchor':None,'line_fraction':None,'floating_screen_is_not_admission':True,'proposal_error':str(exc)}
