"""Batched differentiable DiffAero full rigid body, with SI flight actions.

State: world position, xyzw body-to-world quaternion, world velocity, body
angular velocity. Action: collective thrust in N then desired body rates rad/s.
This is a simulation interface, not a calibrated real-flight controller.
"""
import math

import torch

from .vendor.diffaero.quadrotor import QuadrotorModel

UPSTREAM_SHA = "291ea14196aefbebcf7387dd71f7e096c83878b7"
DIFFAERO_ADAPTER_VERSION = "ares.diffaero.v2"


def _values(profile):
    """Accept an ARES profile or an episode's already sampled parameter dict."""
    source = profile.get("parameters", profile)
    return {key: value["nominal"] if isinstance(value, dict) and "nominal" in value else value
            for key, value in source.items()}


class DiffAeroAdapter:
    def __init__(self, profile, batch_size=1, *, device="cpu", dt=.02,
                 integration_dt=.002, rate_gain=30., max_body_rate=6.,
                 controller_inertia=(.002, .002, .003), compile_dynamics=False):
        if isinstance(profile, (list, tuple)):
            profiles = [_values(item) for item in profile]
            if batch_size != 1 and batch_size != len(profiles):
                raise ValueError("batch_size must match profile list length")
            batch_size = len(profiles)
        else:
            profiles = [_values(profile)] * batch_size
        if (not isinstance(batch_size, int) or batch_size < 1 or not profiles
                or not math.isfinite(dt) or not math.isfinite(integration_dt)
                or dt <= 0 or integration_dt <= 0 or integration_dt > .002
                or not math.isfinite(rate_gain) or rate_gain <= 0
                or not math.isfinite(max_body_rate) or max_body_rate <= 0):
            raise ValueError("positive batch, intervals (integration <= .002 s), gain and rate limit required")
        self.batch_size, self.device = batch_size, torch.device(device)
        self.dt = float(dt)
        self.substeps = max(1, math.ceil(dt / integration_dt))
        self.integration_dt = dt / self.substeps
        self.max_body_rate = float(max_body_rate)
        self.rate_gain = float(rate_gain)
        if self.rate_gain * self.integration_dt > 1:
            raise ValueError("rate gain too large for stable integration")
        # Firmware has a fixed nominal inertia estimate, never episode truth.
        nominal_inertia = torch.as_tensor(controller_inertia, dtype=torch.float32)
        if (nominal_inertia.shape != (3,) or not torch.isfinite(nominal_inertia).all()
                or not (nominal_inertia > 0).all()):
            raise ValueError("controller_inertia must be a finite positive three-vector")
        self.controller_inertia = tuple(float(value) for value in controller_inertia)
        nominal_inertia = nominal_inertia.to(self.device).expand(batch_size, -1)
        self.profiles = profiles
        self.parameters = {}
        for name in profiles[0]:
            self.parameters[name] = torch.as_tensor([item[name] for item in profiles],
                                                     device=self.device, dtype=torch.float32)
        for name in ("mass_kg", "inertia_kg_m2", "max_thrust_n", "motor_lag_s", "gravity_mps2"):
            values = self.parameters[name]
            if not torch.isfinite(values).all() or not (values > 0).all():
                raise ValueError(f"{name} must be finite and positive")
        self.mass = self.parameters["mass_kg"]
        self.gravity = self.parameters["gravity_mps2"]
        self.max_thrust = self.parameters["max_thrust_n"]
        self.motor_lag = self.parameters["motor_lag_s"]
        drag = self.parameters["linear_drag"]
        if not torch.isfinite(drag).all() or not (drag >= 0).all():
            raise ValueError("drag must be finite and nonnegative")
        self.model = QuadrotorModel(self.mass, self.parameters["inertia_kg_m2"],
                                   drag[:, None].expand(-1, 3), self.gravity,
                                   torch.full((batch_size, 3), rate_gain, device=self.device),
                                   controller_inertia=nominal_inertia)
        if type(compile_dynamics) is not bool:
            raise ValueError("compile_dynamics must be boolean")
        if compile_dynamics and self.device.type != "cuda":
            raise ValueError("compiled dynamics require a CUDA device")
        self.compile_dynamics = compile_dynamics
        if self.compile_dynamics:
            self.model.step = torch.compile(self.model.step, fullgraph=True)
        self.reset()

    def _tensor(self, value, width):
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if tensor.shape == (width,):
            tensor = tensor[None].expand(self.batch_size, -1)
        if tensor.shape != (self.batch_size, width) or not torch.isfinite(tensor).all():
            raise ValueError(f"expected finite ({self.batch_size}, {width}) tensor")
        return tensor

    def reset(self, position=None, velocity=None, quaternion=None, angular_velocity=None):
        zero = torch.zeros(self.batch_size, 3, device=self.device)
        p = zero if position is None else self._tensor(position, 3)
        v = zero if velocity is None else self._tensor(velocity, 3)
        w = zero if angular_velocity is None else self._tensor(angular_velocity, 3)
        q = self._tensor([0., 0., 0., 1.] if quaternion is None else quaternion, 4)
        norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        if (norm < 1e-8).any():
            raise ValueError("quaternion must have nonzero norm")
        self.state = torch.cat((p, q / norm, v, w), -1)
        self.applied_thrust = self.mass * self.gravity
        self.last_action = torch.cat((self.applied_thrust[:, None], zero), -1)
        return self.state

    def step(self, action):
        action = self._tensor(action, 4)
        thrust = torch.minimum(action[:, 0].clamp_min(0), self.max_thrust)
        rates = action[:, 1:].clamp(-self.max_body_rate, self.max_body_rate)
        self.last_action = torch.cat((thrust[:, None], rates), -1)
        decay = torch.exp(-self.integration_dt / self.motor_lag)
        for _ in range(self.substeps):
            self.applied_thrust = thrust + (self.applied_thrust - thrust) * decay
            # Upstream controller uses thrust relative to the episode's weight.
            upstream_action = torch.cat(((self.applied_thrust / (self.mass * self.gravity))[:, None], rates), -1)
            self.state = self.model.step(self.state, upstream_action, self.integration_dt)
        return self.state

    def detach(self):
        """Truncate the rollout graph between optimization windows explicitly."""
        self.state = self.state.detach()
        self.applied_thrust = self.applied_thrust.detach()
        self.last_action = self.last_action.detach()
        self.model._acc = self.model._acc.detach()

    @property
    def position(self):
        return self.state[:, :3]

    @property
    def quaternion(self):
        return self.state[:, 3:7]

    @property
    def velocity(self):
        return self.state[:, 7:10]

    @property
    def angular_velocity(self):
        return self.state[:, 10:13]

    @property
    def metadata(self):
        return {"version": DIFFAERO_ADAPTER_VERSION, "upstream_sha": UPSTREAM_SHA,
                "model": "DiffAero QuadrotorModel full rigid body with body-rate controller",
                "state_order": "position_xyz,quaternion_xyzw,velocity_xyz,body_angular_velocity_xyz",
                "action_units": ["N", "rad/s", "rad/s", "rad/s"],
                "dt_s": self.dt, "integration_dt_s": self.integration_dt,
                "rate_gain_per_s": self.rate_gain, "max_body_rate_rad_s": self.max_body_rate,
                "nominal_controller_inertia_kg_m2": list(self.controller_inertia),
                "torch_compile_dynamics": self.compile_dynamics,
                "inertia_uncertainty": "controller uses fixed nominal inertia; rigid body uses sampled episode inertia",
                "rate_loop_calibrated": False, "license": "BSD-3-Clause"}
