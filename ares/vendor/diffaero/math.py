# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch
from torch import Tensor

def rk4(f, X0, U, dt, M=1):
    # type: (Callable[[Tensor, Tensor], Tensor], Tensor, Tensor, float, int) -> Tensor
    DT = dt / M
    X1 = X0
    for _ in range(M):
        k1 = DT * f(X1, U)
        k2 = DT * f(X1 + 0.5 * k1, U)
        k3 = DT * f(X1 + 0.5 * k2, U)
        k4 = DT * f(X1 + k3, U)
        X1 = X1 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return X1

def quat_rotate(quat_xyzw: Tensor, v: Tensor) -> Tensor:
    q_w = quat_xyzw[..., -1]
    q_vec = quat_xyzw[..., :3]
    a = v * (q_w ** 2 - q_vec.pow(2).sum(dim=-1)).unsqueeze(-1)
    b = 2. * q_w.unsqueeze(-1) * torch.cross(q_vec, v, dim=-1)
    c = 2. * q_vec * (q_vec * v).sum(dim=-1, keepdim=True)
    return a + b + c

def quat_axis(quat_xyzw: Tensor, axis: int = 0) -> Tensor:
    basis_vec = torch.zeros(quat_xyzw.shape[0], 3, device=quat_xyzw.device)
    basis_vec[..., axis] = 1
    return quat_rotate(quat_xyzw, basis_vec)

def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).reshape(shape)

    return quat

def quat_inv(quat_xyzw: Tensor) -> Tensor:
    return torch.cat([-quat_xyzw[..., :3], quat_xyzw[..., 3:4]], dim=-1)
