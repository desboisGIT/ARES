"""DiffAero RateController, adapted to a body-angular-velocity state.

Derived from upstream dynamics/controller.py; BSD-3-Clause, see LICENSE.
"""
import torch


class RateController:
    def __init__(self, mass, inertia, gravity, gain):
        self.mass, self.inertia, self.gravity = mass, inertia, gravity
        self.K_angvel = gain

    def __call__(self, q_xyzw, w, action):
        # Upstream rotates w into body coordinates despite the rigid-body state
        # and quaternion derivative already requiring body-frame w. Correct here.
        actual_angvel_b = w
        desired_angvel_b = action[:, 1:]
        angvel_err = desired_angvel_b - actual_angvel_b
        cross = torch.cross(actual_angvel_b,
                            (self.inertia @ actual_angvel_b.unsqueeze(-1)).squeeze(-1), dim=1)
        cross = cross / torch.maximum(cross.norm(dim=-1, keepdim=True) / 100,
                                      torch.ones_like(cross[:, :1])).detach()
        angacc = self.K_angvel * angvel_err
        torque = (self.inertia @ angacc.unsqueeze(-1)).squeeze(-1) + cross
        thrust = action[:, 0] * self.gravity * self.mass
        return thrust, torque
