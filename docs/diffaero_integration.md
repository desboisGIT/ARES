# DiffAero integration

ARES vendors a minimal BSD-3-Clause subset of the official
[DiffAero implementation](https://github.com/flyingbitac/diffaero), pinned to
commit `291ea14196aefbebcf7387dd71f7e096c83878b7`. The associated paper is
[arXiv:2509.10247](https://arxiv.org/abs/2509.10247).

`ares/diffaero_adapter.py` runs the upstream full quadrotor rigid-body equations,
RK4 integrator, quaternion math, anisotropic body-frame drag and rate controller.
It does not substitute the Phase 3 point-mass model. Inputs and state remain
PyTorch tensors throughout integration, including actuator filtering and rate
feedback, so loss gradients flow through complete rollout windows on CPU or GPU.
Call `detach()` explicitly when truncating a rollout graph.

The dynamics dependency is PyTorch. The upstream Taichi rendering stack is
independent of its PyTorch dynamics. Upstream's full dependency list also includes
PyTorch3D, Open3D, Hydra, logging, video and training packages; those are unnecessary
for this minimal subset. No separate upstream package installation is needed.
The implementation was checked using Python 3.14 and PyTorch 2.11.0+cu130.

## Interface

```python
from ares.diffaero_adapter import DiffAeroAdapter

# Either an ARES profile (uses nominal values), a sampled parameter dictionary,
# or a list of sampled dictionaries, one per independent batch member.
model = DiffAeroAdapter(profile, batch_size=64, device="cuda", dt=.02)
model.reset(position=[0, 0, 1])
state = model.step(actions)  # shape [64,4], N followed by body rates in rad/s
```

State order is `[position_xyz, quaternion_xyzw, world_velocity_xyz,
body_angular_velocity_xyz]`, shape `[batch,13]`. The quaternion rotates body
vectors into the world frame. Positive thrust points along body +Z; world gravity
points down. The adapter exposes `.position`, `.quaternion`, `.velocity`,
`.angular_velocity`, `.parameters` (all supplied numeric profile fields as batched
tensors), `.applied_thrust`, and `.metadata`.

The default policy interval is 20 ms and the internal integration interval is
2 ms (500 Hz). A smaller integration interval can be requested. The collective
thrust command is bounded to `[0,max_thrust_n]`, and rate commands are bounded to
±6 rad/s by default. First-order thrust lag uses each episode's `motor_lag_s`;
its held-input update is exact at each integration boundary. Hover thrust is the
initial actuator state. The endpoint actuator thrust is held during each 2 ms RK4
rigid-body step. These saturation operations retain ordinary PyTorch piecewise
continuous gradients, including zero gradients outside saturation bounds.

Mass, gravity, drag, all three diagonal inertia values, maximum thrust and motor
lag are consumed from the supplied profile. Actual rigid-body inertia is sampled
per episode. The rate controller always uses a separate fixed nominal estimate,
by default `controller_inertia=(.002, .002, .003)` kg·m². This finite positive
three-vector is configurable and recorded in metadata. The sampled actual
inertia is never supplied to the rate controller. Both its rate-error torque and
its gyroscopic feed-forward use the nominal estimate; the rigid-body Euler
response uses the sampled inertia, preserving model mismatch. Other profile fields are retained
for the scene/sensor wrapper. This module does not independently randomize them:
the caller supplies reproducibly sampled episode dictionaries, including
individual dictionaries when each drone in a batch needs different parameters.

## Patches and limits

The complete license, source paths and patch notes live in
`ares/vendor/diffaero/LICENSE` and `ares/vendor/diffaero/UPSTREAM.md`.

The important upstream correction is angular-velocity convention: upstream's
quaternion derivative and Euler angular dynamics consume body angular velocity,
but its rate controller rotates that same state from world into body again.
ARES consistently uses body angular velocity in all three places. A regression
check verifies that changing yaw does not change the response to the same body
rate command. Full diagonal inertia preserves separately sampled Jx and Jy.
Quaternion normalization is out of place, with its complete autograd derivative.
The upstream artificial gradient-decay function is omitted.

Upstream expects normalized thrust relative to vehicle weight. The adapter
converts physical Newton commands using each episode's own mass and gravity
before calling the upstream controller. ARES adds the explicit thrust-lag state;
upstream does not implement motor lag. The controller's default rate gain is
30 s⁻¹, an explicit uncalibrated simulation assumption. It is recorded in
metadata and configurable. Rate gain times integration interval is constrained
to at most one for numerical stability. Upstream's bounded gyroscopic
feed-forward term is retained.

The model has no per-motor allocation/saturation, calibrated firmware gains,
propeller flow model or contact response. Scene clearance/collision handling,
speed limits, compute latency, depth rendering and sensor noise belong to the
flight environment. Successful differentiation here establishes a simulator
integration, not real-flight fidelity or the Phase 4 success gate.

## Inertia uncertainty correction

Adapter version `ares.diffaero.v2` separates controller and actual inertia.
Previously the controller used the episode's exact inertia and cancelled the
sampled inertia out of the angular response. That made inertia randomization
ineffective. A single 1 rad/s roll command from rest over 20 ms now produces
approximately 0.486583 rad/s with the declared low inertia and 0.420422 rad/s
with high inertia. The former implementation produced 0.451188 rad/s in both
cases. This probe holds all other parameters fixed. CPU and CUDA both pass the
regression, including finite nonzero gradients through the mismatched models.
The fixed controller estimate remains an uncalibrated simulation assumption.
Physical Newton commands and per-episode mass/gravity normalization are unchanged.

## Verification

Run `.venv/bin/python -m unittest tests.test_diffaero_adapter -v`.
The six checks passed on CPU and the available CUDA device:

- Nominal hover and unit quaternion preservation; body-rate response invariant
  to yaw, catching the upstream frame inconsistency.
- Multistep derivatives for all four action channels compared to central finite
  differences through the complete rollout.
- Independent episode masses, distinct diagonal inertias and exact actuator lag
  updates for batched profile dictionaries.
- CUDA batch rollout with finite, nonzero action gradients.

The checks also cover the low/high inertia response probe, fixed controller
inertia on CPU/CUDA, finite gradients under inertia mismatch, and validation of
the configurable nominal inertia estimate.

## Optional CUDA compilation

`DiffAeroAdapter(..., compile_dynamics=True)` wraps the pure full rigid-body
RK4 step with `torch.compile(..., fullgraph=True)`. This option requires CUDA
and defaults to false; CPU execution remains eager. The flight environment
selects CUDA compilation explicitly. Metadata records `torch_compile_dynamics`.
Compilation preserves the body-rate model, fixed nominal controller inertia,
actuator lag and full rollout gradients. No surrogate dynamics are introduced.

On the available CUDA machine with PyTorch 2.11.0+cu130, a batch of 64 drones
with a 20 ms policy interval took approximately 71.0 ms per adapter step plus
backward in eager mode and 14.3 ms compiled, about a 4.95× improvement. Initial
compilation and first backward took 15.5 s. A new same-shaped adapter instance
reused the compiled cache and completed its first step/backward in 13.9 ms.
These are local simulator timings, not onboard policy inference measurements.
Compile caches specialize on shapes, precision, device and graph constants;
other configurations can incur another initial compilation cost.

The CUDA regression compares multistep states and action gradients against eager
execution using heterogeneous actual inertias. The initial benchmark's maximum
state difference was 5.96e-8 and maximum action-gradient difference 2.98e-8.
