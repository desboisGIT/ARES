"""Evaluator-only swept sphere collision against occupied voxel boxes.

Each straight segment is checked continuously, including grazing contacts.
Clearance is capped at max(.4 m, radius)-radius and explicitly marked as a
lower bound when the local search cannot establish a smaller exact distance.
Leaving the scene body's valid bounds is reported separately from voxel contact.
"""
import numpy as np
import torch

from .contracts import validate_scene
from .evaluator import segment_box_distance
from .fast_world import _KERNEL

_COLLISION_KERNEL = _KERNEL.split('extern "C" __global__ void dynamics', 1)[0] + r'''
extern "C" __global__ void check_sweeps(
 const unsigned char* volume,int nx,int ny,int nz,float res,float ox,float oy,float oz,
 const float* starts,const float* ends,const float* radii,int batch,
 unsigned char* collision,float* clearance,unsigned char* bounds,unsigned char* lower_bound){
 int env=blockDim.x*blockIdx.x+threadIdx.x;if(env>=batch)return;
 const float* a=starts+env*3;const float* b=ends+env*3;
 float radius=radii[env],pad=fmaxf(.4f,radius),best=pad;
 float origin[3]={ox,oy,oz};int shape[3]={nx,ny,nz},low[3],high[3];
 collision[env]=0;bounds[env]=0;
 for(int k=0;k<3;k++){
  low[k]=max(0,(int)floorf((fminf(a[k],b[k])-pad-origin[k])/res));
  high[k]=min(shape[k]-1,(int)floorf((fmaxf(a[k],b[k])+pad-origin[k])/res));
  if(a[k]<origin[k]+radius||a[k]>origin[k]+shape[k]*res-radius||
     b[k]<origin[k]+radius||b[k]>origin[k]+shape[k]*res-radius)bounds[env]=1;
 }
 for(int x=low[0];x<=high[0];x++)for(int y=low[1];y<=high[1];y++)for(int z=low[2];z<=high[2];z++){
  if(!volume[(x*ny+y)*nz+z])continue;
  float lo[3]={ox+x*res,oy+y*res,oz+z*res},hi[3]={lo[0]+res,lo[1]+res,lo[2]+res};
  float distance=segment_box(a,b,lo,hi);
  best=fminf(best,distance);
  if(distance<=radius+1.e-7f)collision[env]=1;
 }
 clearance[env]=best-radius;lower_bound[env]=(best>=pad-1.e-6f);
}
'''


class FlightCollisionChecker:
    def __init__(self, scene, device='cpu'):
        validate_scene(scene)
        self.device = torch.device(device)
        grid = scene['grid']
        self.shape = tuple(grid['shape'])
        self.origin = np.asarray(grid['origin'], dtype=np.float32)
        self.resolution = float(grid['resolution'])
        self.volume = np.zeros(self.shape, dtype=np.uint8)
        indices = np.asarray(grid['occupied'], dtype=int)
        if len(indices):
            self.volume[tuple(indices.T)] = 1
        if self.device.type == 'cuda':
            import cupy as cp
            self.cp = cp
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            with torch.cuda.device(self.device), cp.cuda.Device(index):
                with cp.cuda.ExternalStream(torch.cuda.current_stream(self.device).cuda_stream):
                    self.device_volume = cp.asarray(self.volume)
                    self.kernel = cp.RawKernel(_COLLISION_KERNEL, 'check_sweeps', options=('--std=c++11',))
            self._ready = torch.cuda.Event()
            self._ready.record(torch.cuda.current_stream(self.device))
        elif self.device.type != 'cpu':
            raise ValueError('CPU or CUDA collision device required')

    @torch.no_grad()
    def check(self, start, end, radii):
        start = torch.as_tensor(start, device=self.device, dtype=torch.float32).detach().contiguous()
        end = torch.as_tensor(end, device=self.device, dtype=torch.float32).detach().contiguous()
        radii = torch.as_tensor(radii, device=self.device, dtype=torch.float32).detach().contiguous()
        if (start.ndim != 2 or start.shape[1] != 3 or start.shape[0] == 0
                or end.shape != start.shape or radii.shape != (start.shape[0],)):
            raise ValueError('start/end must be (B,3), radii (B,), nonempty')
        if (not torch.isfinite(start).all() or not torch.isfinite(end).all()
                or not torch.isfinite(radii).all() or not (radii > 0).all()):
            raise ValueError('finite positions and positive finite radii required')
        batch = start.shape[0]
        collision = torch.zeros(batch, dtype=torch.bool, device=self.device)
        bounds = torch.zeros_like(collision)
        lower_bound = torch.zeros_like(collision)
        clearance = torch.empty(batch, dtype=torch.float32, device=self.device)
        if self.device.type == 'cuda':
            cp = self.cp
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(self._ready)
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            with torch.cuda.device(self.device), cp.cuda.Device(index):
                with cp.cuda.ExternalStream(stream.cuda_stream):
                    i,f = np.int32,np.float32
                    self.kernel(((batch+127)//128,), (128,), (self.device_volume,*map(i,self.shape),
                        f(self.resolution),*map(f,self.origin),cp.from_dlpack(start),cp.from_dlpack(end),
                        cp.from_dlpack(radii),i(batch),cp.from_dlpack(collision),cp.from_dlpack(clearance),
                        cp.from_dlpack(bounds),cp.from_dlpack(lower_bound)))
        else:
            shape = np.asarray(self.shape)
            for index, (a,b,radius) in enumerate(zip(start.numpy(),end.numpy(),radii.numpy())):
                # Match the CUDA float32 broad phase, then solve box distances
                # in the independent double-precision reference routine.
                pad = max(np.float32(.4),radius)
                low = np.maximum(0,np.floor((np.minimum(a,b)-pad-self.origin)/self.resolution).astype(int))
                high = np.minimum(shape-1,np.floor((np.maximum(a,b)+pad-self.origin)/self.resolution).astype(int))
                best = float(pad)
                for x in range(low[0],high[0]+1):
                    for y in range(low[1],high[1]+1):
                        for z in range(low[2],high[2]+1):
                            if not self.volume[x,y,z]:
                                continue
                            lo = self.origin + np.array([x,y,z],dtype=np.float32)*self.resolution
                            distance = segment_box_distance(a.tolist(),b.tolist(),lo.tolist(),(lo+self.resolution).tolist())
                            best = min(best,distance)
                            if distance <= float(radius)+1e-7:
                                collision[index] = True
                upper = self.origin + shape*self.resolution-radius
                lower = self.origin+radius
                bounds[index] = bool(np.any(a<lower) or np.any(a>upper) or np.any(b<lower) or np.any(b>upper))
                clearance[index] = best-float(radius)
                lower_bound[index] = best >= float(pad)-1e-6
        return {'collision':collision, 'clearance':clearance, 'left_bounds':bounds,
                'clearance_is_lower_bound':lower_bound}
