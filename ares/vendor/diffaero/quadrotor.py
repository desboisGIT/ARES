"""DiffAero full 13-state rigid body, with tensor-only profile construction.

Derived from upstream dynamics/quadrotor.py; BSD-3-Clause, see LICENSE.
"""
import torch
from .controller import RateController
from .math import rk4, quat_rotate, quat_inv, quat_axis, quat_mul


class QuadrotorModel:
    def __init__(self, mass, inertia, drag, gravity, rate_gain, controller_inertia):
        self.mass = mass
        self._J = torch.diag_embed(inertia)
        self._J_inv = torch.diag_embed(inertia.reciprocal())
        self._D = torch.diag_embed(drag)
        self._G_vec = torch.stack((torch.zeros_like(gravity), torch.zeros_like(gravity), -gravity), -1)
        # The firmware estimate is deliberately independent of actual inertia.
        self.controller = RateController(mass, torch.diag_embed(controller_inertia), gravity, rate_gain)
        self.device = mass.device
        self._acc = torch.zeros(len(mass), 3, device=self.device)

    def dynamics(self, X, U):
        p, q, v, w = X[..., :3], X[..., 3:7], X[..., 7:10], X[..., 10:13]
        thrust, torque = self.controller(q, w, U)
        M = torque - torch.cross(w, torch.matmul(self._J, w.unsqueeze(-1)).squeeze(-1), dim=-1)
        w_dot = torch.matmul(self._J_inv, M.unsqueeze(-1)).squeeze(-1)
        fdrag = quat_rotate(q, (self._D @ quat_rotate(quat_inv(q), v).unsqueeze(-1)).squeeze(-1))
        thrust_acc = quat_axis(q, 2) * (thrust / self.mass).unsqueeze(-1)
        acc = thrust_acc + self._G_vec - fdrag / self.mass.unsqueeze(-1)
        self._acc = acc
        q_dot = 0.5 * quat_mul(q, torch.cat((w, torch.zeros_like(w[:, :1])), dim=-1))
        return torch.concat([v, q_dot, acc, w_dot], dim=-1)

    def step(self, state, action, dt):
        new_state = rk4(self.dynamics, state, action, dt=dt, M=1)
        q = new_state[..., 3:7]
        # Keep the full normalization gradient and avoid in-place autograd writes.
        q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.cat((new_state[..., :3], q, new_state[..., 7:]), dim=-1)
