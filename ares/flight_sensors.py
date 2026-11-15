"""Episode-randomized forward depth sensors for full rigid-body flight batches.

CUDA rays and noise run on Torch's current stream with zero-copy DLPack views.
Material labels are explicitly evaluator-only; policy features contain depth and
validity only. Optical Z depth is used, never an omnidirectional hidden range.
"""
import copy
import hashlib
import json
import math
from importlib.resources import files

import numpy as np
import torch

from .depth import GroundTruthDepthRenderer, OPTICAL_TO_BODY, _TRACE, _NOISE
from .vendor.diffaero.math import quat_rotate


def _episode_profile(sampled):
    if "schema_version" in sampled:
        sampled = {key: value["nominal"] for key, value in sampled["parameters"].items()}
    profile = json.loads(files("ares").joinpath("data/configs/drone_sim_v2.json").read_text())
    for name, value in sampled.items():
        if name not in profile["parameters"]:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (list, tuple, np.ndarray)):
            value = list(value)
            bounds = [[item, item] for item in value]
        else:
            bounds = [value, value]
        profile["parameters"][name] = {"nominal": value, "range": bounds}
    return profile


def _per_environment_trace():
    source = _TRACE.replace("float fx, float fy, float cx, float cy,\n float min_z, float max_z,",
                            "const float* camera, const float* limits,")
    marker = "int env = id / pixels, pixel = id % pixels;"
    source = source.replace(marker, marker + "\n float fx=camera[env*4],fy=camera[env*4+1],cx=camera[env*4+2],cy=camera[env*4+3];\n float min_z=limits[env*2],max_z=limits[env*2+1];")
    return source


def _per_environment_noise():
    source = _NOISE.replace("float near_z, float far_z, float base, float quadratic,\n int edge_width, float edge_threshold, float range_probability, const float* holes,",
                            "const float* settings, const float* holes,")
    marker = "int env=id/pixels, p=id%pixels, row=p/width, col=p%width;"
    source = source.replace(marker, marker + "\n const float* s=settings+env*7;\n float near_z=s[0],far_z=s[1],base=s[2],quadratic=s[3];\n int edge_width=(int)s[4]; float edge_threshold=s[5],range_probability=s[6];")
    return source.replace("holes[material]", "holes[env*5+material]")


class FlightSensors:
    def __init__(self, scene, sampled_profiles, device="cpu", seed=0, environment_ids=None):
        if not isinstance(sampled_profiles, (list, tuple)) or not sampled_profiles:
            raise ValueError("one sampled profile per batch member required")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.device, self.seed = torch.device(device), seed
        self.profiles = [_episode_profile(item) for item in sampled_profiles]
        self.batch_size = len(self.profiles)
        identities = np.arange(self.batch_size, dtype=np.uint32) if environment_ids is None else np.asarray(environment_ids)
        if (identities.shape != (self.batch_size,) or not np.issubdtype(identities.dtype, np.integer)
                or np.any(identities < 0) or np.any(identities > 2**32-1)
                or len(np.unique(identities)) != self.batch_size):
            raise ValueError("one unique uint32 sensor identity per environment required")
        self.environment_ids = identities.astype(np.uint32)
        # Scene volume is shared; only calibration/noise models vary per episode.
        from .contracts import validate_profile
        from .recording import CameraIntrinsics
        from .sensors import DepthNoiseModel
        from .depth import FrameClock
        first = GroundTruthDepthRenderer(scene, self.profiles[0], backend="cpu", seed=seed)
        self.renderers = [first]
        for profile in self.profiles[1:]:
            validate_profile(profile)
            renderer = copy.copy(first)
            renderer.camera = CameraIntrinsics.from_profile(profile)
            renderer.model = DepthNoiseModel.from_profile(profile)
            renderer.clock = FrameClock(profile["parameters"]["depth_fps"]["nominal"],
                                        profile["parameters"]["compute_latency_s"]["nominal"])
            self.renderers.append(renderer)
        self.width, self.height = self.renderers[0].camera.width, self.renderers[0].camera.height
        if any((r.camera.width, r.camera.height) != (self.width, self.height) for r in self.renderers):
            raise ValueError("camera dimensions must agree within a batch")
        self.max_range = torch.tensor([r.model.max_range_m for r in self.renderers], device=self.device)
        self.optical_to_body = torch.tensor(OPTICAL_TO_BODY, dtype=torch.float32, device=self.device)
        if self.device.type == "cuda":
            import cupy as cp
            self.cp = cp
            with torch.cuda.device(self.device), cp.cuda.Device(self.device.index if self.device.index is not None else torch.cuda.current_device()):
                with cp.cuda.ExternalStream(torch.cuda.current_stream(self.device).cuda_stream):
                    self.volume = cp.asarray(self.renderers[0].volume)
                    self.camera = cp.asarray([[r.camera.fx, r.camera.fy, r.camera.cx, r.camera.cy] for r in self.renderers], dtype=cp.float32)
                    self.limits = cp.asarray([[r.model.min_range_m, r.model.max_range_m] for r in self.renderers], dtype=cp.float32)
                    self.settings = cp.asarray([[r.model.min_range_m, r.model.max_range_m, r.model.base_std_m,
                        r.model.quadratic_std_per_m2, r.model.edge_width_px, r.model.edge_threshold_m,
                        r.model.max_range_dropout_probability] for r in self.renderers], dtype=cp.float32)
                    self.holes = cp.asarray([r.model.material_hole_probability for r in self.renderers], dtype=cp.float32)
                    self.env_ids = cp.asarray(self.environment_ids, dtype=cp.uint32)
                    self.trace_kernel = cp.RawKernel(_per_environment_trace(), "trace_depth", options=("--std=c++11",))
                    self.noise_kernel = cp.RawKernel(_per_environment_noise(), "sensor_noise", options=("--std=c++11",))
            # Device constants may be reused from other Torch streams later.
            self._ready = torch.cuda.Event()
            self._ready.record(torch.cuda.current_stream(self.device))
        elif self.device.type != "cpu":
            raise ValueError("CPU or CUDA sensor device required")

    @torch.no_grad()
    def observe(self, position, quaternion, frame_index, *, noise=True):
        if type(frame_index) is not int or frame_index < 0 or type(noise) is not bool:
            raise ValueError("nonnegative frame index and boolean noise required")
        position = torch.as_tensor(position, dtype=torch.float32, device=self.device).detach().contiguous()
        quaternion = torch.as_tensor(quaternion, dtype=torch.float32, device=self.device).detach()
        if position.shape != (self.batch_size, 3) or quaternion.shape != (self.batch_size, 4):
            raise ValueError("pose batch shape mismatch")
        if not torch.isfinite(position).all() or not torch.isfinite(quaternion).all():
            raise ValueError("pose must be finite")
        norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        if (norm < 1e-8).any():
            raise ValueError("quaternion must be nonzero")
        quaternion = quaternion / norm
        axes = torch.eye(3, device=self.device)
        rotation = torch.stack([quat_rotate(quaternion, axes[i].expand(self.batch_size, -1)) for i in range(3)], -1)
        rotation = (rotation @ self.optical_to_body).contiguous()
        if self.device.type == "cpu":
            depths, materials = [], []
            for index, renderer in enumerate(self.renderers):
                frame = renderer.render(position[index].numpy(), rotation[index].numpy(), frame_index=frame_index,
                                        noise=noise, environment_ids=self.environment_ids[index:index+1])
                depths.append(frame.depth_m[0])
                materials.append(frame.ground_truth["material_ids"][0])
            depth = torch.from_numpy(np.stack(depths))
            material = torch.from_numpy(np.stack(materials))
        else:
            cp, r = self.cp, self.renderers[0]
            torch.cuda.current_stream(self.device).wait_event(self._ready)
            with torch.cuda.device(self.device), cp.cuda.Device(self.device.index if self.device.index is not None else torch.cuda.current_device()):
                with cp.cuda.ExternalStream(torch.cuda.current_stream(self.device).cuda_stream):
                    depth = torch.empty((self.batch_size, self.height, self.width), device=self.device)
                    material = torch.empty(depth.shape, device=self.device, dtype=torch.int8)
                    ideal = torch.empty_like(depth) if noise else depth
                    grid = ((depth.numel()+255)//256,)
                    f, i = np.float32, np.int32
                    self.trace_kernel(grid, (256,), (self.volume, *map(i, r.shape), f(r.resolution), *map(f, r.origin),
                        cp.from_dlpack(position), cp.from_dlpack(rotation), i(self.width), i(self.height), self.camera,
                        self.limits, i(self.batch_size), cp.from_dlpack(ideal), cp.from_dlpack(material)))
                    if noise:
                        digest = hashlib.sha256(f"ares.cuda_depth.v1:{self.seed}:{frame_index}".encode()).digest()
                        low, high = int.from_bytes(digest[:4], "little"), int.from_bytes(digest[4:8], "little")
                        self.noise_kernel(grid, (256,), (cp.from_dlpack(ideal), cp.from_dlpack(material), self.env_ids,
                            i(self.width), i(self.height), i(self.batch_size), self.settings, self.holes,
                            np.uint32(low), np.uint32(high), cp.from_dlpack(depth)))
        valid = torch.isfinite(depth)
        normalized = torch.where(valid, depth / self.max_range[:, None, None], 0.)
        features = torch.stack((normalized, valid.to(torch.float32)), dim=1)
        return {"depth": features, "raw_depth": depth, "material_ids": material,
                "material_ids_ground_truth_only": True}
