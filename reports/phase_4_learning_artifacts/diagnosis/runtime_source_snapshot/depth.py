"""Batched ground-truth voxel ray casting with CPU reference and CUDA kernels.

Depth is optical-axis Z in metres, not Euclidean range. Camera axes are right,
down, forward. Pose rotations map those axes into the z-up world. Only depth_m
belongs on a runtime sensor interface; the separately named ground_truth block
is evaluator-only. NumPy and CuPy are optional, installed with simulation extras.
"""

from dataclasses import dataclass
import hashlib
import math

from .contracts import validate_scene, validate_profile

MATERIALS = ("normal", "shiny", "black", "glass", "mirror")
DEPTH_RENDERER_VERSION = "ares.depth_renderer.v1"
OPTICAL_TO_BODY = ((0., 0., 1.), (-1., 0., 0.), (0., -1., 0.))


@dataclass(frozen=True)
class FrameClock:
    fps: float
    latency_s: float

    def __post_init__(self):
        if (type(self.fps) not in (int, float) or not math.isfinite(self.fps) or self.fps <= 0
                or type(self.latency_s) not in (int, float) or not math.isfinite(self.latency_s) or self.latency_s < 0):
            raise ValueError("fps must be positive and latency finite/nonnegative")

    def timestamps(self, frame_index):
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative integer")
        captured = frame_index / self.fps
        return captured, captured + self.latency_s


@dataclass
class DepthBatch:
    depth_m: object
    timestamp_s: float
    available_at_s: float
    ground_truth: dict
    backend: str


_TRACE = r'''
extern "C" __global__ void trace_depth(
 const signed char* volume, int nx, int ny, int nz, float res,
 float ox, float oy, float oz, const float* positions, const float* rotations,
 int width, int height, float fx, float fy, float cx, float cy,
 float min_z, float max_z, int batch, float* depth, signed char* material) {
 int id = blockIdx.x * blockDim.x + threadIdx.x;
 int pixels = width * height;
 if (id >= batch * pixels) return;
 int env = id / pixels, pixel = id % pixels;
 float u = ((pixel % width) - cx) / fx;
 float v = ((pixel / width) - cy) / fy;
 const float* r = rotations + env * 9;
 float direction[3] = {r[0]*u+r[1]*v+r[2], r[3]*u+r[4]*v+r[5], r[6]*u+r[7]*v+r[8]};
 float position[3] = {positions[3*env], positions[3*env+1], positions[3*env+2]};
 float origin[3] = {ox,oy,oz}; int shape[3] = {nx,ny,nz};
 float enter = 0.0f, leave = max_z;
 depth[id] = __int_as_float(0x7fffffff); material[id] = -1;
 for (int a=0;a<3;a++) {
   if (fabsf(direction[a]) < 1.e-12f) {
     if (position[a] < origin[a] || position[a] >= origin[a] + shape[a]*res) return;
   } else {
     float t0=(origin[a]-position[a])/direction[a];
     float t1=(origin[a]+shape[a]*res-position[a])/direction[a];
     enter=fmaxf(enter,fminf(t0,t1)); leave=fminf(leave,fmaxf(t0,t1));
   }
 }
 if (enter > leave || leave < 0.f) return;
 int cell[3], step[3]; float crossing[3], delta[3];
 for (int a=0;a<3;a++) {
   float p = position[a] + direction[a] * (enter + 1.e-6f);
   cell[a]=(int)floorf((p-origin[a])/res);
   cell[a]=max(0,min(shape[a]-1,cell[a]));
   step[a]=(direction[a]>1.e-12f)?1:((direction[a]<-1.e-12f)?-1:0);
   if (step[a]) {
     float edge=origin[a]+(cell[a]+(step[a]>0))*res;
     crossing[a]=(edge-position[a])/direction[a]; delta[a]=fabsf(res/direction[a]);
   } else { crossing[a]=__int_as_float(0x7f800000); delta[a]=crossing[a]; }
 }
 float t=enter;
 for (int iteration=0;iteration<nx+ny+nz+3;iteration++) {
   if (cell[0]<0 || cell[0]>=nx || cell[1]<0 || cell[1]>=ny || cell[2]<0 || cell[2]>=nz || t>leave) return;
   int value=volume[(cell[0]*ny+cell[1])*nz+cell[2]];
   if (value) {
     if (t>min_z && t<=max_z) {depth[id]=t; material[id]=(signed char)(value-1);}
     return;
   }
   float next=fminf(crossing[0],fminf(crossing[1],crossing[2]));
   for (int a=0;a<3;a++) if (crossing[a]<=next+1.e-6f) {cell[a]+=step[a]; crossing[a]+=delta[a];}
   t=next;
 }
}
'''

_NOISE = r'''
__device__ unsigned int mix_bits(unsigned int x) {
 x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16);
}
__device__ float uniform01(unsigned int key) {return (mix_bits(key)>>8)*5.9604644775390625e-8f;}
__device__ bool valid_depth(float z, int m, float near_z, float far_z) {return isfinite(z) && z>near_z && z<=far_z && m>=0;}
extern "C" __global__ void sensor_noise(
 const float* ideal, const signed char* materials, const unsigned int* env_ids,
 int width, int height, int batch, float near_z, float far_z, float base, float quadratic,
 int edge_width, float edge_threshold, float range_probability, const float* holes,
 unsigned int seed_low, unsigned int seed_high, float* measured) {
 int id=blockIdx.x*blockDim.x+threadIdx.x, pixels=width*height;
 if (id>=batch*pixels) return;
 int env=id/pixels, p=id%pixels, row=p/width, col=p%width;
 float z=ideal[id]; int material=materials[id];
 measured[id]=__int_as_float(0x7fffffff);
 if (!valid_depth(z,material,near_z,far_z)) return;
 if (edge_width>0) {
   int radius=edge_width-1;
   for (int dy=-radius;dy<=radius;dy++) for(int dx=-radius;dx<=radius;dx++) {
     int y=row+dy,x=col+dx;
     if (y<0||y>=height||x<0||x>=width) continue;
     int center=env*pixels+y*width+x;
     float z0=ideal[center]; bool v0=valid_depth(z0,materials[center],near_z,far_z);
     for(int k=0;k<4;k++) {
       int y1=y+(k==0)-(k==1),x1=x+(k==2)-(k==3);
       if(y1<0||y1>=height||x1<0||x1>=width) continue;
       int other=env*pixels+y1*width+x1;
       float z1=ideal[other]; bool v1=valid_depth(z1,materials[other],near_z,far_z);
       if(v0!=v1 || (v0&&v1&&fabsf(z0-z1)>edge_threshold)) return;
     }
   }
 }
 unsigned long long counter=(unsigned long long)env_ids[env]*(unsigned long long)pixels+(unsigned int)p;
 unsigned int key=mix_bits(seed_low ^ mix_bits(seed_high ^ (unsigned int)(counter>>32)) ^ mix_bits((unsigned int)counter));
 float u=fmaxf(1.e-7f,uniform01(key^0xa511e9b3u));
 float gaussian=sqrtf(-2.f*logf(u))*cosf(6.28318530718f*uniform01(key^0x63d83595u));
 float sigma=sqrtf(base*base+quadratic*quadratic*z*z*z*z);
 float value=z+gaussian*sigma;
 if(uniform01(key^0xb5297a4du)<holes[material] ||
    uniform01(key^0x68e31da4u)<range_probability*(z/far_z)*(z/far_z) || value<=near_z || value>far_z) return;
 measured[id]=value;
}
'''


class GroundTruthDepthRenderer:
    def __init__(self, scene, profile, *, backend="cpu", seed=0, camera=None):
        import numpy as np
        from .recording import CameraIntrinsics
        from .sensors import DepthNoiseModel
        validate_scene(scene)
        validate_profile(profile)
        if backend not in ("cpu", "cuda") or type(seed) is not int:
            raise ValueError("backend must be cpu/cuda and seed an integer")
        self.np, self.backend, self.seed = np, backend, seed
        if camera is not None and not isinstance(camera, CameraIntrinsics):
            raise ValueError("camera override must be calibrated CameraIntrinsics")
        self.camera = CameraIntrinsics.from_profile(profile) if camera is None else camera
        self.model = DepthNoiseModel.from_profile(profile)
        self.clock = FrameClock(profile["parameters"]["depth_fps"]["nominal"],
                                profile["parameters"]["compute_latency_s"]["nominal"])
        self.origin = np.asarray(scene["grid"]["origin"], dtype=np.float32)
        self.shape = tuple(scene["grid"]["shape"])
        self.resolution = float(scene["grid"]["resolution"])
        self.volume = np.zeros(self.shape, dtype=np.int8)
        for surface in scene["surfaces"]:
            indices = np.asarray(surface["voxel_indices"], dtype=int)
            if len(indices):
                self.volume[tuple(indices.T)] = MATERIALS.index(surface["material"]) + 1
        if backend == "cuda":
            try:
                import cupy as cp
                self.cp = cp
                self.device_volume = cp.asarray(self.volume)
                self._trace = cp.RawKernel(_TRACE, "trace_depth", options=("--std=c++11",))
                self._noise = cp.RawKernel(_NOISE, "sensor_noise", options=("--std=c++11",))
                self._holes = cp.asarray(self.model.material_hole_probability, dtype=cp.float32)
            except (ImportError, RuntimeError) as exc:
                raise RuntimeError("CUDA depth requires a working CuPy CUDA installation; no silent CPU fallback") from exc

    def _cpu(self, positions, rotations):
        np, camera = self.np, self.camera
        depth = np.full((len(positions), camera.height, camera.width), np.nan, dtype=np.float32)
        material = np.full(depth.shape, -1, dtype=np.int8)
        origin, shape, resolution = self.origin.astype(float), self.shape, self.resolution
        for env, (position, rotation) in enumerate(zip(positions.astype(float), rotations.astype(float))):
            for row in range(camera.height):
                for col in range(camera.width):
                    direction = rotation @ np.array([(col-camera.cx)/camera.fx, (row-camera.cy)/camera.fy, 1.])
                    enter, leave = 0., self.model.max_range_m
                    hit = True
                    for axis in range(3):
                        if abs(direction[axis]) < 1e-12:
                            if not origin[axis] <= position[axis] < origin[axis]+shape[axis]*resolution:
                                hit = False
                                break
                        else:
                            ts = ((origin[axis]-position[axis])/direction[axis],
                                  (origin[axis]+shape[axis]*resolution-position[axis])/direction[axis])
                            enter, leave = max(enter, min(ts)), min(leave, max(ts))
                    if not hit or enter > leave or leave < 0:
                        continue
                    cell = np.floor((position+direction*(enter+1e-6)-origin)/resolution).astype(int)
                    cell = np.clip(cell, 0, np.asarray(shape)-1)
                    step = np.where(direction > 1e-12, 1, np.where(direction < -1e-12, -1, 0))
                    crossing, delta = np.full(3, np.inf), np.full(3, np.inf)
                    for axis in range(3):
                        if step[axis]:
                            boundary = origin[axis]+(cell[axis]+(step[axis]>0))*resolution
                            crossing[axis] = (boundary-position[axis])/direction[axis]
                            delta[axis] = abs(resolution/direction[axis])
                    t = enter
                    for _ in range(sum(shape)+3):
                        if any(not 0 <= cell[i] < shape[i] for i in range(3)) or t > leave:
                            break
                        value = self.volume[tuple(cell)]
                        if value:
                            if self.model.min_range_m < t <= self.model.max_range_m:
                                depth[env, row, col], material[env, row, col] = t, value - 1
                            break
                        t = float(np.min(crossing))
                        axes = crossing <= t+1e-6
                        cell[axes] += step[axes]
                        crossing[axes] += delta[axes]
        return depth, material

    def render(self, positions, rotations, *, frame_index=0, noise=True, environment_ids=None):
        np, c = self.np, self.camera
        captured, available = self.clock.timestamps(frame_index)
        if type(noise) is not bool:
            raise ValueError("noise must be boolean")
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim == 1:
            positions = positions[None]
        if positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions) or not np.isfinite(positions).all():
            raise ValueError("positions must be a finite nonempty (B,3) array")
        batch = len(positions)
        rotations = np.asarray(rotations, dtype=np.float32)
        if rotations.shape == (3, 3):
            rotations = np.broadcast_to(rotations, (batch, 3, 3))
        if (rotations.shape != (batch, 3, 3) or not np.isfinite(rotations).all()
                or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5, rtol=0)
                or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5, rtol=0)):
            raise ValueError("rotations must be proper camera-to-world rotation matrices")
        if environment_ids is None:
            environment_ids = np.arange(batch, dtype=np.uint32)
        else:
            environment_ids = np.asarray(environment_ids)
            if (environment_ids.shape != (batch,) or not np.issubdtype(environment_ids.dtype, np.integer)
                    or np.any(environment_ids < 0) or np.any(environment_ids > 2**32-1)
                    or len(np.unique(environment_ids)) != batch):
                raise ValueError("environment_ids must be unique uint32-range integers, one per pose")
            environment_ids = environment_ids.astype(np.uint32)
        if self.backend == "cpu":
            ideal, material = self._cpu(positions, rotations)
            if noise:
                from .sensors import apply_depth_noise
                # Preserve stable identities even when CPU batches are split.
                measured = np.stack([apply_depth_noise(ideal[i], material[i], self.model,
                                      self.seed ^ (int(eid) << 32), frame_index)
                                     for i, eid in enumerate(environment_ids)])
            else:
                measured = ideal.copy()
        else:
            cp = self.cp
            ideal = cp.empty((batch, c.height, c.width), dtype=cp.float32)
            material = cp.empty(ideal.shape, dtype=cp.int8)
            grid = ((batch*c.width*c.height+255)//256,)
            f = np.float32
            self._trace(grid, (256,), (self.device_volume, *map(np.int32, self.shape), f(self.resolution),
                *map(f, self.origin), cp.asarray(np.ascontiguousarray(positions)), cp.asarray(np.ascontiguousarray(rotations)),
                np.int32(c.width), np.int32(c.height), f(c.fx), f(c.fy), f(c.cx), f(c.cy),
                f(self.model.min_range_m), f(self.model.max_range_m), np.int32(batch), ideal, material))
            if noise:
                measured = cp.empty_like(ideal)
                digest = hashlib.sha256(f"ares.cuda_depth.v1:{self.seed}:{frame_index}".encode()).digest()
                low, high = int.from_bytes(digest[:4], "little"), int.from_bytes(digest[4:8], "little")
                m = self.model
                self._noise(grid, (256,), (ideal, material, cp.asarray(environment_ids), np.int32(c.width), np.int32(c.height),
                    np.int32(batch), f(m.min_range_m), f(m.max_range_m), f(m.base_std_m), f(m.quadratic_std_per_m2),
                    np.int32(m.edge_width_px), f(m.edge_threshold_m), f(m.max_range_dropout_probability), self._holes,
                    np.uint32(low), np.uint32(high), measured))
            else:
                measured = ideal.copy()
        return DepthBatch(measured, captured, available,
                          {"ideal_depth_m": ideal, "material_ids": material}, self.backend)

    def to_host(self, array):
        return self.cp.asnumpy(array) if self.backend == "cuda" else self.np.asarray(array)

    def synchronize(self):
        if self.backend == "cuda":
            self.cp.cuda.get_current_stream().synchronize()
