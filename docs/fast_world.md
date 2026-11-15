# Fast exploration world

`GroundTruthFastWorld` keeps a scene's hidden occupancy inside the evaluator.
Independent environments share its geometry, with separate pose, velocity,
yaw, clock, collision state and observed-map bitsets. They do not collide with
one another. `observation(i)` returns a copied observed map, depth image and
public vehicle state. Planners receive this dictionary, never the world object.

Actions are `[world_vx, world_vy, world_vz, yaw_rate, land_flag]` in SI units.
The point mass directly tracks capped velocity (2 m/s by default), with capped
yaw rate (1.5 rad/s), at 50 ms steps. Collision checks continuously sweep the
15 cm spherical hull against occupied voxel boxes; endpoint checks alone are
insufficient. Collision, leaving the body-safe bounds, timeout or accepted
landing terminates that environment. Terminal environments freeze and acquire
no further observations. A collision endpoint is the attempted endpoint, not
a physically resolved contact pose. Clearance saturates at a documented lower
bound far from obstacles; the result flags whether this bound is in use.

The ideal pinhole cone uses the profile's 87° horizontal / 58° vertical field
of view, 32×24 pixels and 4 m maximum optical-Z range. Exact voxel DDA marks
traversed free voxels and the first occupied voxel, never voxels behind it.
Rays without a return can still certify ideal free space to their range limit.
Depth holes are NaN internally and null in JSON. A cone is acquired initially
and after each nonterminal step; its clock is therefore 20 Hz here. The initial
physical hull is known free from the launch guarantee. Other unknown voxels
remain unknown. This is deliberately a different acquisition model from both
Phase 0's six-axis witness and Phase 2's noisy 90 fps sensor model.

Landing requires a command within 10 cm of home, speed at most 0.1 m/s, and
at least one second of stationary dwell. An early request is consumed without
landing. Manual sessions latch each request until one physics tick consumes it,
and expire stale controls after 0.4 s. Start begins an independent wall-clock
thread; HTTP polling, braking and window focus do not pause the mission clock.

`ares.exploration_scorer.v1` reuses the frozen Phase 0 metric arithmetic, with
a new acquisition/evidence adapter. Its denominator is the fixed six-axis
observable voxel union from all body-clear, home-connected voxel centers.
It is independent of the policy path and of the submitted report. Coverage
must reach 95%, free/occupied floor-plan mean IoU 0.8, all reachable rooms must
match one-to-one by polygon IoU 0.8, with zero collisions and safe return/landing.
Unsupported map cells cannot borrow hidden truth. Reports include only observed
voxels; inferred room rectangles never fill unknown map cells. The old scorer
and scene contracts retain their versions.

The CUDA backend runs dynamics, exact collision, cone rays and atomic map
updates on the GPU. CPU implements the reference model; analytic tests exercise
actual CUDA parity when available. Policy planning, scoring and optional frame
logging run outside the vectorized kernel timing. The throughput harness counts
actual active individual environment steps and synchronizes before reading wall
time. It excludes initialization, JIT warmup and observation downloads. Frozen
environment calls do not inflate the numerator.

This model has perfect pose, direct velocity response, ideal voxel returns and
no material dropout, IMU error or actuator dynamics. It supports algorithm
development and throughput measurement; Phase 2's real camera calibration,
mapping and onboard-computer gates remain separate and pending.
