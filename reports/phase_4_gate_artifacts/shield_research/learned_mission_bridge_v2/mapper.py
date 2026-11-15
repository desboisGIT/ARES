"""Public target-ray mapper; truth rendering stays inside the supplied sensor.

Every added voxel passes the original exact robust first-hit witness test.
The forward camera limits are public calibration; no occupied grid is queried.
"""
from fractions import Fraction as F
import math
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'learned_command_bridge_v1'))
from adapter import rotation
class ForwardMapper:
    def __init__(self,known_map,*,horizontal_fov,vertical_fov,max_range,rays_per_frame=24):
        self.map=known_map;self.hfov=float(horizontal_fov);self.vfov=float(vertical_fov);self.range=float(max_range)
        self.rays=rays_per_frame;self.cursor=0;self.frames=0;self.attempted=0;self.admitted=0;self.range_rejected=0
        self.cells=np.array(list(np.ndindex(known_map.shape)))
        self.centers=np.asarray(known_map.origin)+(self.cells+.5)*known_map.resolution
    def observe(self,supplier,packet,*,time_s):
        delta=self.centers-np.asarray(packet.p);body=delta@rotation(packet.q)
        dist=np.linalg.norm(delta,axis=-1);forward=body[:,0]
        ray_error=.00101
        keep=(forward>.1)&(dist<self.range-.03)&(dist>.2)
        keep&=np.abs(body[:,1])+ray_error<(forward-ray_error)*math.tan(self.hfov/2)
        keep&=np.abs(body[:,2])+ray_error<(forward-ray_error)*math.tan(self.vfov/2)
        candidates=[j for j in np.flatnonzero(keep) if tuple(self.cells[j]) not in self.map.free]
        added=[];self.frames+=1
        if not candidates:return {'added':added,'attempted':0,'forward_FOV_maxrange_enforced':True}
        start=self.cursor%len(candidates);selected=(candidates[start:]+candidates[:start])[:self.rays];self.cursor+=self.rays
        for j in selected:
            direction=tuple(delta[j]);cell=tuple(map(int,self.cells[j]));self.attempted+=1
            ray=supplier.ray(packet,direction,ray_id=1000000+self.attempted)
            # This fixture's first-hit sensor returns no usable observation
            # when the finite hit is beyond the sampled camera range.
            if (ray.measured_depth+.01)*(dist[j]+.00101)+1e-9>self.range:
                self.range_rejected+=1;continue
            witness=self.map.observe(ray,cell,optical_range=1.,time_s=time_s)
            if witness.free_witness:added.append(cell);self.admitted+=1
        return {'added':added,'attempted':len(selected),'forward_FOV_maxrange_enforced':True}

def extend_guard_free_snapshot(guard,absolute_known_map,*,global_tick,epoch_tick):
    now=F(global_tick,100);deadline=F(epoch_tick+guard.deadline_tick,100)
    def outward(v,upper):
        x=float(v)
        if (F.from_float(x)<v if upper else F.from_float(x)>v):x=math.nextafter(x,math.inf if upper else -math.inf)
        return x
    eligible={c for c in absolute_known_map.free if absolute_known_map.known_free(c,time_s=outward(now,False))
        and absolute_known_map.known_free(c,time_s=outward(deadline,True))}
    old=set(guard._free_through_deadline);guard._free_through_deadline=tuple(sorted(old|eligible))
    return len(eligible-old)
