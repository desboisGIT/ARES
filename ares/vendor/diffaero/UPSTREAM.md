# DiffAero source provenance

Source: https://github.com/flyingbitac/diffaero
Commit: `291ea14196aefbebcf7387dd71f7e096c83878b7`
Paper: https://arxiv.org/abs/2509.10247
License: BSD-3-Clause (complete text in LICENSE).

This is a minimal, modified source subset, not an installation of the upstream
training or rendering stack. `quadrotor.py` derives from
`dynamics/quadrotor.py`, `controller.py` from `dynamics/controller.py`, and
`math.py` retains the required functions from `utils/math.py`. The math file
also retains its NVIDIA BSD-3-Clause notice. No upstream weights are included.

Changes:

- Remove Hydra/OmegaConf, PyTorch3D, logger, rendering and unused helpers;
  construct directly from batched PyTorch tensors. The upstream Taichi renderer
  is separate from its PyTorch dynamics and is not required for differentiation.
- Support all three diagonal inertia entries, rather than tying Jx=Jy, and
  batched gravity/mass/drag. Drop unused motor mixing parameters, which the
  upstream force/torque controller does not consume.
- Correct the upstream controller's angular-velocity frame: rigid-body Euler
  equations and quaternion derivative require body-frame angular velocity; do
  not rotate this same body-frame state from world to body again.
- Use a fixed nominal controller inertia estimate distinct from the sampled
  rigid-body inertia. Both controller rate torque and gyroscopic feed-forward
  use the nominal estimate; Euler dynamics use actual episode inertia. This
  preserves inertia uncertainty rather than cancelling it with exact knowledge.
  The adapter exposes and records this estimate (default diagonal .002/.002/.003
  kg m^2). This is an uncalibrated firmware-model assumption.
- Keep upstream normalized-thrust controller input internally, with the ARES
  adapter converting Newtons using each episode's mass and gravity.
- Replace in-place quaternion normalization with an out-of-place operation and
  retain its full derivative. Disable upstream gradient decay (ordinary exact
  autograd is used). Preserve RK4, quaternion dynamics, body-frame drag,
  gyroscopic torque, rate-error torque and feed-forward cancellation.
- ARES adds the first-order thrust actuator and repeated 500 Hz integration in
  `ares/diffaero_adapter.py`; upstream has no motor-lag model. Rate-loop gain is
  an explicit simulation assumption, not a measured firmware fit.

The bounded upstream gyroscopic feed-forward term is retained. No motor mixing,
per-motor saturation, propeller aerodynamics, contact response, or calibrated
firmware model is claimed. Collision handling belongs to ARES's scene wrapper.
