"""Independent point-mass environments with forward-cone voxel observations.

The ground-truth geometry stays inside the evaluator. Policies receive copied,
observed-only grids. This ideal fast model intentionally excludes dynamics,
odometry error and material dropout; Phase 2 realism remains a separate model.
"""

from dataclasses import asdict, dataclass
import copy
import hashlib
from itertools import product
import json
import math

import numpy as np

from .contracts import validate_scene, validate_profile
from .evaluator import segment_box_distance

FAST_WORLD_VERSION = "ares.fast_world.v1"


@dataclass(frozen=True)
class FastWorldConfig:
    dt: float = .05
    max_speed: float = 2.
    max_yaw_rate: float = 1.5
    max_time_s: float = 3600.
    camera_width: int | None = None
    camera_height: int | None = None

    def __post_init__(self):
        for name in ("dt", "max_speed", "max_yaw_rate", "max_time_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("camera_width", "camera_height"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")


@dataclass(frozen=True)
class FastEpisode:
    world_digest: str
    seed: int
    home: tuple
    time_s: float
    position: tuple
    collisions: int
    left_bounds: bool
    landed_safely: bool
    minimum_clearance_m: float
    minimum_clearance_is_lower_bound: bool
    observations: tuple


def inflation_structure(radius, resolution):
    """Sphere/voxel intersection offsets at voxel-center configurations."""
    extent = math.ceil(radius / resolution + .5)
    offsets = np.indices((2*extent+1,) * 3).transpose(1, 2, 3, 0) - extent
    distance = np.maximum(np.abs(offsets)*resolution-resolution*.5, 0.)
    return np.sum(distance**2, axis=-1) <= radius**2 + 1e-12


_KERNEL = r'''
__device__ float distance2(float t,const float* a,const float* d,const float* lo,const float* hi) {
 float total=0.f;
 for(int k=0;k<3;k++){float p=a[k]+t*d[k]; float gap=fmaxf(fmaxf(lo[k]-p,p-hi[k]),0.f);total+=gap*gap;}
 return total;
}
__device__ float segment_box(const float* a,const float* b,const float* lo,const float* hi) {
 float d[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]}; float cuts[8]={0.f,1.f};int n=2;
 for(int k=0;k<3;k++)if(fabsf(d[k])>1.e-12f){
  float t0=(lo[k]-a[k])/d[k],t1=(hi[k]-a[k])/d[k];
  if(t0>0&&t0<1)cuts[n++]=t0;if(t1>0&&t1<1)cuts[n++]=t1;
 }
 for(int i=1;i<n;i++){float v=cuts[i];int j=i-1;while(j>=0&&cuts[j]>v){cuts[j+1]=cuts[j];j--;}cuts[j+1]=v;}
 float best=1.e30f;
 for(int i=0;i<n;i++)best=fminf(best,distance2(cuts[i],a,d,lo,hi));
 for(int i=0;i<n-1;i++){
  float mid=(cuts[i]+cuts[i+1])*.5f,numer=0,denom=0;
  for(int k=0;k<3;k++){float v=a[k]+mid*d[k];if(v<lo[k]||v>hi[k]){
   float plane=v<lo[k]?lo[k]:hi[k];numer+=d[k]*(a[k]-plane);denom+=d[k]*d[k];}}
  if(denom>0){float t=fmaxf(cuts[i],fminf(cuts[i+1],-numer/denom));best=fminf(best,distance2(t,a,d,lo,hi));}
 }
 return sqrtf(best);
}
extern "C" __global__ void dynamics(const unsigned char* volume,int nx,int ny,int nz,float res,
 float ox,float oy,float oz,float radius,float pad,float dt,float maxspeed,float maxyaw,int maxsteps,
 const float* actions,const float* homes,float* positions,float* velocities,float* yaws,
 int* steps,int* dwell,int* collisions,unsigned char* bounds,unsigned char* landed,unsigned char* done,
 float* clearance,int batch){
 int env=blockDim.x*blockIdx.x+threadIdx.x;if(env>=batch||done[env])return;
 const float* command=actions+env*5;float a[3],b[3],origin[3]={ox,oy,oz};int shape[3]={nx,ny,nz};
 float largest=fmaxf(fabsf(command[0]),fmaxf(fabsf(command[1]),fabsf(command[2])));
 float unit[3]={0,0,0};if(largest>0)for(int k=0;k<3;k++)unit[k]=command[k]/largest;
 float length=sqrtf(unit[0]*unit[0]+unit[1]*unit[1]+unit[2]*unit[2]);
 float magnitude=length>0?fminf(largest,maxspeed/length):0.f;
 for(int k=0;k<3;k++){a[k]=positions[3*env+k];velocities[3*env+k]=unit[k]*magnitude;b[k]=a[k]+velocities[3*env+k]*dt;}
 float best=pad;
 int low[3],high[3];
 for(int k=0;k<3;k++){
  low[k]=max(0,(int)floorf((fminf(a[k],b[k])-pad-origin[k])/res));
  high[k]=min(shape[k]-1,(int)floorf((fmaxf(a[k],b[k])+pad-origin[k])/res));
  if(b[k]<origin[k]+radius||b[k]>origin[k]+shape[k]*res-radius)bounds[env]=1;
 }
 for(int x=low[0];x<=high[0];x++)for(int y=low[1];y<=high[1];y++)for(int z=low[2];z<=high[2];z++){
  if(!volume[(x*ny+y)*nz+z])continue;
  float lo[3]={ox+x*res,oy+y*res,oz+z*res},hi[3]={lo[0]+res,lo[1]+res,lo[2]+res};
  best=fminf(best,segment_box(a,b,lo,hi));
 }
 clearance[env]=fminf(clearance[env],best-radius);
 if(best<=radius+1.e-7f)collisions[env]++;
 for(int k=0;k<3;k++)positions[3*env+k]=b[k];
 float yaw=yaws[env]+fmaxf(-maxyaw,fminf(maxyaw,command[3]))*dt;
 yaws[env]=atan2f(sinf(yaw),cosf(yaw));steps[env]++;
 float home_distance=0,velocity2=0;
 for(int k=0;k<3;k++){float diff=b[k]-homes[3*env+k];home_distance+=diff*diff;velocity2+=velocities[3*env+k]*velocities[3*env+k];}
 dwell[env]=(home_distance<=.01f+1.e-7f&&velocity2<=.01f)?dwell[env]+1:0;
 if(command[4]>0.5f&&dwell[env]*dt>=1.f-1.e-6f&&!collisions[env]&&!bounds[env])landed[env]=1;
 if(collisions[env]||bounds[env]||landed[env]||steps[env]>=maxsteps){done[env]=1;for(int k=0;k<3;k++)velocities[3*env+k]=0.f;}
}
extern "C" __global__ void cone(const unsigned char* volume,int nx,int ny,int nz,float res,
 float ox,float oy,float oz,const float* positions,const float* yaws,const unsigned char* done,
 int width,int height,float fx,float fy,float cx,float cy,float maxz,int words,unsigned int* observed,
 float* depth,int batch){
 int id=blockIdx.x*blockDim.x+threadIdx.x,pixels=width*height;if(id>=batch*pixels)return;
 int env=id/pixels,p=id%pixels;if(done[env])return;
 depth[id]=__int_as_float(0x7fffffff);
 float u=((p%width)-cx)/fx,v=((p/width)-cy)/fy,c=cosf(yaws[env]),s=sinf(yaws[env]);
 float direction[3]={c+s*u,s-c*u,-v};
 float pos[3]={positions[3*env],positions[3*env+1],positions[3*env+2]},origin[3]={ox,oy,oz};int shape[3]={nx,ny,nz};
 int cell[3],step[3];float crossing[3],delta[3];
 for(int k=0;k<3;k++){
  cell[k]=(int)floorf((pos[k]-origin[k])/res);if(cell[k]<0||cell[k]>=shape[k])return;
  step[k]=direction[k]>1.e-12f?1:(direction[k]<-1.e-12f?-1:0);
  if(step[k]){crossing[k]=(origin[k]+(cell[k]+(step[k]>0))*res-pos[k])/direction[k];delta[k]=fabsf(res/direction[k]);}
  else{crossing[k]=__int_as_float(0x7f800000);delta[k]=crossing[k];}
 }
 float t=0;
 for(int iteration=0;iteration<nx+ny+nz+3;iteration++){
  if(t>maxz||cell[0]<0||cell[0]>=nx||cell[1]<0||cell[1]>=ny||cell[2]<0||cell[2]>=nz)return;
  int index=(cell[0]*ny+cell[1])*nz+cell[2];
  atomicOr(observed+env*words+(index>>5),1u<<(index&31));
  if(volume[index]){if(t>0)depth[id]=t;return;}
  float next=fminf(crossing[0],fminf(crossing[1],crossing[2]));
  for(int k=0;k<3;k++)if(crossing[k]<=next+1.e-6f){cell[k]+=step[k];crossing[k]+=delta[k];}t=next;
 }
}
'''


class GroundTruthFastWorld:
    def __init__(self, scene, profile, *, num_envs=1, backend="cpu", seed=0, config=None):
        validate_scene(scene)
        validate_profile(profile)
        if type(num_envs) is not int or num_envs < 1 or backend not in ("cpu", "cuda") or type(seed) is not int:
            raise ValueError("positive num_envs, cpu/cuda backend, and integer seed required")
        self.config = config or FastWorldConfig()
        if not isinstance(self.config, FastWorldConfig):
            raise ValueError("config must be FastWorldConfig")
        self.scene, self.profile = copy.deepcopy(scene), copy.deepcopy(profile)
        self.seed, self.num_envs, self.backend = seed, num_envs, backend
        grid, params = scene["grid"], profile["parameters"]
        self.shape, self.origin = tuple(grid["shape"]), np.array(grid["origin"], dtype=float)
        self.resolution, self.radius = float(grid["resolution"]), float(params["body_radius_m"]["nominal"])
        self.volume = np.zeros(self.shape, dtype=np.uint8)
        self.volume[tuple(np.asarray(grid["occupied"], dtype=int).T)] = 1
        self.home = np.array(scene["spawn"], dtype=np.float32)
        self.width = self.config.camera_width or params["depth_width"]["nominal"]
        self.height = self.config.camera_height or params["depth_height"]["nominal"]
        self.hfov, self.vfov = params["depth_horizontal_fov_rad"]["nominal"], params["depth_vertical_fov_rad"]["nominal"]
        self.max_range = params["depth_max_range_m"]["nominal"]
        self.fx, self.fy = self.width/(2*math.tan(self.hfov/2)), self.height/(2*math.tan(self.vfov/2))
        self.cx, self.cy = (self.width-1)/2, (self.height-1)/2
        self.pad = max(self.radius, .4)
        self.voxels, self.words = math.prod(self.shape), (math.prod(self.shape)+31)//32
        if self._clearance(self.home, self.home) <= 0 or not self._inside(self.home):
            raise ValueError("spawn lacks body clearance")
        self.digest = hashlib.sha256(json.dumps({"scene": scene, "profile": profile,
            "config": asdict(self.config), "world_version": FAST_WORLD_VERSION,"seed": seed}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.xp = np
        if backend == "cuda":
            try:
                import cupy as cp
                self.xp = cp
                self._device_volume = cp.asarray(self.volume)
                self._dynamics_kernel = cp.RawKernel(_KERNEL, "dynamics", options=("--std=c++11",))
                self._cone_kernel = cp.RawKernel(_KERNEL, "cone", options=("--std=c++11",))
            except ImportError as exc:
                raise RuntimeError("CUDA fast world requires the cuda optional extra") from exc
        self.reset()

    def _host(self, value):
        return self.xp.asnumpy(value) if self.backend == "cuda" else np.asarray(value)

    def _inside(self, position):
        return bool(np.all(position >= self.origin+self.radius) and
                    np.all(position <= self.origin+np.array(self.shape)*self.resolution-self.radius))

    def _clearance(self, a, b):
        low = np.maximum(0, np.floor((np.minimum(a,b)-self.pad-self.origin)/self.resolution).astype(int))
        high = np.minimum(np.array(self.shape)-1, np.floor((np.maximum(a,b)+self.pad-self.origin)/self.resolution).astype(int))
        best = self.pad
        for x in range(low[0], high[0]+1):
            for y in range(low[1], high[1]+1):
                for z in range(low[2], high[2]+1):
                    if self.volume[x,y,z]:
                        lower = self.origin+np.array([x,y,z])*self.resolution
                        best = min(best, segment_box_distance(a,b,lower,lower+self.resolution))
        return best-self.radius

    def reset(self):
        xp, batch = self.xp, self.num_envs
        self._positions = xp.asarray(np.tile(self.home, (batch,1)))
        self._homes = self._positions.copy()
        self._velocities = xp.zeros((batch,3), dtype=xp.float32)
        self._yaws = xp.zeros(batch, dtype=xp.float32)
        self._steps = xp.zeros(batch, dtype=xp.int32)
        self._dwell = xp.zeros(batch, dtype=xp.int32)
        self._collisions = xp.zeros(batch, dtype=xp.int32)
        self._bounds = xp.zeros(batch, dtype=xp.uint8)
        self._landed = xp.zeros(batch, dtype=xp.uint8)
        self._done = xp.zeros(batch, dtype=xp.uint8)
        self._clearances = xp.full(batch, self.pad-self.radius, dtype=xp.float32)
        self._observed = xp.zeros((batch,self.words), dtype=xp.uint32)
        self._depth = xp.full((batch,self.height,self.width), xp.nan, dtype=xp.float32)
        # Occupancy intersecting the initial physical body is known free from
        # the setup guarantee, not from an omnidirectional sensor.
        # Use the actual launch point, which need not be a voxel center.
        # A full bounding box is required: near a voxel corner the sphere can
        # intersect diagonal cells excluded by center-based inflation offsets.
        launch = self.home.astype(float)
        lower = np.maximum(0, np.floor((launch-self.radius-self.origin)/self.resolution).astype(int))
        upper = np.minimum(np.asarray(self.shape)-1,
                           np.floor((launch+self.radius-self.origin)/self.resolution).astype(int))
        bits = np.zeros(self.words, dtype=np.uint32)
        for cell in product(*(range(lower[k], upper[k]+1) for k in range(3))):
            box_lower = self.origin+np.asarray(cell)*self.resolution
            gap = np.maximum(np.maximum(box_lower-launch, launch-(box_lower+self.resolution)), 0.)
            if float(np.dot(gap, gap)) <= self.radius**2+1e-12:
                i = np.ravel_multi_index(cell,self.shape)
                bits[i//32] |= np.uint32(1 << (i%32))
        self._observed[:] = xp.asarray(bits)
        self._sense()

    def synchronize(self):
        if self.backend == "cuda":
            self.xp.cuda.get_current_stream().synchronize()

    def step(self, actions):
        xp = self.xp
        values = xp.asarray(actions, dtype=xp.float32)
        if values.shape != (self.num_envs,5):
            raise ValueError("actions must have shape (num_envs,5)")
        if not bool(xp.isfinite(values).all()) or not bool(((values[:,4]==0)|(values[:,4]==1)).all()):
            raise ValueError("actions must be finite; land flag must be 0 or 1")
        if self.backend == "cuda":
            f, i = np.float32, np.int32
            self._dynamics_kernel(((self.num_envs+127)//128,), (128,), (
                self._device_volume,*map(i,self.shape),f(self.resolution),*map(f,self.origin),
                f(self.radius),f(self.pad),f(self.config.dt),f(self.config.max_speed),f(self.config.max_yaw_rate),
                i(math.ceil(self.config.max_time_s/self.config.dt)),values,self._homes,self._positions,self._velocities,
                self._yaws,self._steps,self._dwell,self._collisions,self._bounds,self._landed,self._done,self._clearances,i(self.num_envs)))
        else:
            for env in range(self.num_envs):
                if self._done[env]:
                    continue
                velocity = values[env,:3].copy()
                largest = float(np.max(np.abs(velocity)))
                if largest:
                    unit = velocity/largest
                    velocity = unit*min(largest,self.config.max_speed/float(np.linalg.norm(unit)))
                a = self._positions[env].copy()
                b = (a+velocity*self.config.dt).astype(np.float32)
                clearance = self._clearance(a,b)
                self._clearances[env] = min(self._clearances[env],clearance)
                self._collisions[env] += int(clearance<=1e-7)
                self._bounds[env] |= not self._inside(b)
                self._positions[env], self._velocities[env] = b, velocity
                yaw = float(self._yaws[env])+float(np.clip(values[env,3],-self.config.max_yaw_rate,self.config.max_yaw_rate))*self.config.dt
                self._yaws[env] = math.atan2(math.sin(yaw),math.cos(yaw))
                self._steps[env] += 1
                at_home = np.linalg.norm(b-self.home)<=.1000001 and np.linalg.norm(velocity)<=.1
                self._dwell[env] = self._dwell[env]+1 if at_home else 0
                if values[env,4] and self._dwell[env]*self.config.dt>=1-1e-6 and not self._collisions[env] and not self._bounds[env]:
                    self._landed[env] = 1
                if self._collisions[env] or self._bounds[env] or self._landed[env] or self._steps[env]*self.config.dt>=self.config.max_time_s:
                    self._done[env], self._velocities[env] = 1, 0
        self._sense()

    def _sense(self):
        if self.backend == "cuda":
            f, i = np.float32, np.int32
            total = self.num_envs*self.width*self.height
            self._cone_kernel(((total+255)//256,), (256,), (self._device_volume,*map(i,self.shape),f(self.resolution),
                *map(f,self.origin),self._positions,self._yaws,self._done,i(self.width),i(self.height),
                f(self.fx),f(self.fy),f(self.cx),f(self.cy),f(self.max_range),i(self.words),self._observed,self._depth,i(self.num_envs)))
            return
        yy,xx = np.indices((self.height,self.width))
        u,v = (xx.ravel()-self.cx)/self.fx,(yy.ravel()-self.cy)/self.fy
        for env in range(self.num_envs):
            if self._done[env]:
                continue
            pos = self._positions[env].astype(float)
            c,s = math.cos(float(self._yaws[env])), math.sin(float(self._yaws[env]))
            directions = np.stack((c+s*u,s-c*u,-v),axis=1)
            cell = np.tile(np.floor((pos-self.origin)/self.resolution).astype(int),(len(u),1))
            step = np.where(directions>1e-12,1,np.where(directions < -1e-12,-1,0))
            with np.errstate(divide="ignore",invalid="ignore"):
                crossing = np.where(step,(self.origin+(cell+(step>0))*self.resolution-pos)/directions,np.inf)
                delta = np.where(step,np.abs(self.resolution/directions),np.inf)
            t,active = np.zeros(len(u)),np.ones(len(u),dtype=bool)
            depth = np.full(len(u),np.nan,dtype=np.float32)
            for _ in range(sum(self.shape)+3):
                active &= (t<=self.max_range)&np.all((cell>=0)&(cell<self.shape),axis=1)
                ids = np.flatnonzero(active)
                if not len(ids):
                    break
                index = np.ravel_multi_index(tuple(cell[ids].T),self.shape)
                np.bitwise_or.at(self._observed[env],index//32,np.left_shift(np.uint32(1),(index%32).astype(np.uint32)))
                hit = self.volume[tuple(cell[ids].T)]>0
                depth[ids[hit]] = np.where(t[ids[hit]]>0,t[ids[hit]],np.nan)
                active[ids[hit]] = False
                ids = ids[~hit]
                if not len(ids):
                    continue
                nxt = np.min(crossing[ids],axis=1)
                axes = crossing[ids]<=nxt[:,None]+1e-6
                cell[ids] += step[ids]*axes
                crossing[ids] += np.where(axes,delta[ids],0)
                t[ids] = nxt
            self._depth[env] = depth.reshape(self.height,self.width)

    def known_grid(self, env_index=0):
        self._index(env_index)
        bits = self._host(self._observed[env_index])
        indices = np.arange(self.voxels)
        observed = (bits[indices//32] >> (indices%32).astype(np.uint32))&1
        return np.where(observed.reshape(self.shape),self.volume+1,0).astype(np.uint8)

    def _index(self, index):
        if type(index) is not int or not 0<=index<self.num_envs:
            raise ValueError("environment index out of range")

    def ground_truth_state(self, env_index=0):
        self._index(env_index)
        i = env_index
        return {"position": self._host(self._positions[i]).copy(), "velocity": self._host(self._velocities[i]).copy(),
            "yaw": float(self._host(self._yaws[i])), "time_s": int(self._host(self._steps[i]))*self.config.dt,
            "collisions": int(self._host(self._collisions[i])), "left_bounds": bool(self._host(self._bounds[i])),
            "landed_safely": bool(self._host(self._landed[i])), "done": bool(self._host(self._done[i])),
            "minimum_clearance_m": float(self._host(self._clearances[i])),
            "minimum_clearance_is_lower_bound": bool(self._host(self._clearances[i])>=self.pad-self.radius-1e-6)}

    def observation(self, env_index=0):
        state = self.ground_truth_state(env_index)
        return {key: state[key] for key in ("position","velocity","yaw","time_s")} | {
            "known_grid": self.known_grid(env_index), "home": self.home.astype(float).copy(),
            "resolution": self.resolution, "origin": self.origin.copy(), "body_radius_m": self.radius,
            "max_speed": self.config.max_speed, "max_yaw_rate": self.config.max_yaw_rate,"dt": self.config.dt,
            "camera": {"hfov_rad": self.hfov,"vfov_rad": self.vfov,"max_range_m": self.max_range},
            "depth_m": self._host(self._depth[env_index]).copy()}

    def snapshot_for_scoring(self, env_index=0):
        state, grid = self.ground_truth_state(env_index), self.known_grid(env_index)
        cells = np.argwhere(grid>0)
        observed = tuple((tuple(int(v) for v in cell),int(grid[tuple(cell)])) for cell in cells)
        return FastEpisode(self.digest,self.seed,tuple(float(v) for v in self.home),state["time_s"],
            tuple(float(v) for v in state["position"]),state["collisions"],state["left_bounds"],state["landed_safely"],
            state["minimum_clearance_m"],state["minimum_clearance_is_lower_bound"],observed)
