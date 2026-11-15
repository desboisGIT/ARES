# Recorded-data odometry and occupancy mapping

The Phase 2 CPU baseline consumes recorded optical Z depth and optional body IMU
samples. It uses NumPy only, without GPU or platform-specific hooks, and the same
classes can run against a simulator or an ARM recording replay. Performance on
an onboard board, three real-place maps, fitted real depth noise and measured
loop error require hardware and real recordings; synthetic tests do not satisfy
those Phase 2 gates.

## Pose estimation

`DepthInertialOdometry(camera, initial_pose=None)` aligns each incoming depth
frame to an accepted keyframe with projective point-to-plane ICP. The caller
can configure `max_points` (900 by default), `max_iterations` (18), correspondence
distance, residual threshold and maximum condition number. Sampling is
deterministic. Depth normals come from weighted 5×5 local plane fits: inverse
optical depth is affine on a plane. Depth-edge neighbors and nonplanar fits are
discarded. The measured 3D points themselves are retained; averaging points at
plane intersections would bias alignment. Projection and distance/normal checks reject
invalid correspondences; Huber weights reduce large residuals without deleting
minority surface directions. A six-parameter least-squares increment
updates the camera-to-reference transform, which composes with the reference
camera-to-world pose.

`update(depth_m, timestamp_s, imu_accel_mps2=None, imu_gyro_rps=None)` returns a
dictionary containing:

- `pose`: NumPy 4×4 camera-to-world transform.
- `status`: `initialized`, `tracking`, or `lost`.
- `can_integrate`: true only for an accepted initial or tracked frame.
- `reason`: initialization, alignment or rejection reason.
- `rmse_m`, `correspondences`, `rank`, `condition_number`, and `iterations`.
- `normal_span_ratio` and `normal_span_rank`: support for distinct surface-normal
  directions, independent of the gyro prior.
- `covariance`: approximate 6×6 least-squares covariance, or `None` on failure.
- `gyro_prior_used` and `accelerometer_translation_used`.

Covariance order is rotation XYZ (radians), translation XYZ (metres), in the
reference camera frame. This local residual-based covariance has a small noise
floor and does not model accumulated drift, calibration or systematic errors.
Condition derives from the six-DoF point-to-plane normal matrix. Effective rank
also requires three supported translation-normal directions. `normal_span_ratio`
is the smallest/largest eigenvalue ratio of the normal outer-product matrix;
`normal_span_rank` counts eigenvalues above a relative threshold
(`min_normal_span_ratio=.01`). This conservative heuristic prevents noise and
mixed corner normals from supplying a missing lateral constraint in two-plane
views. These are geometry diagnostics, not a calibrated navigation uncertainty
guarantee. Covariance uses depth residuals and does not incorporate the gyro prior
or accumulated pose uncertainty.

An initial pose is an explicit frame anchor, not an estimated absolute position.
If omitted, it is identity in optical axes. A level camera in the specified
Z-up world should receive the camera-to-body rotation from `recording_format.md`
as its initial rotation. Initial depth must provide observable six-DoF geometry
before mapping is allowed. A lone plane cannot observe translation parallel to
the plane or rotation about its normal, so a blank stationary wall correctly
returns `lost` rather than claiming good stationary tracking. Two independent
surface directions likewise cannot observe all three translation axes; noisy
depth must not turn that missing direction into a valid motion estimate.

The optional gyro both seeds rotation and supplies a rotation residual throughout
ICP, using the default camera/body extrinsic rotation. Successive update samples
are integrated by the trapezoidal rule and composed until keyframe replacement.
If only the current sample is available, the first interval uses constant
angular velocity. A missing sample breaks prior coverage until a new keyframe;
the estimator does not fill the gap with invented IMU data. Replay should
synchronize timestamps; this interface does not integrate the entire high-rate
IMU stream or estimate biases.

`gyro_noise_std_rps` is an explicit positive constructor setting, default .005
rad/s when no measured profile value is supplied. `gyro_min_angle_std_rad`
(default .0005 rad) is an algorithmic uncertainty floor for timing, calibration
and integration approximation. For keyframe interval `dt`, the prior angle
standard deviation is `hypot(gyro_noise_std_rps*dt, gyro_min_angle_std_rad)`.
The rate term grows linearly with interval rather than assuming independent
noise averages away. ICP residual scale converts this angle uncertainty into
least-squares row weights. These assumptions need real sensor calibration;
neither the floor nor resulting covariance is a guaranteed error bound.

The accelerometer sample is validated but translation integration is explicitly
unsupported (`accelerometer_translation_used=False`). There is no bias estimator,
stereo feature tracking, flow fusion, relocalization against a global map or loop
closure. This is depth odometry with a rotation prior, not a complete VIO system.

The reference is promoted after estimated translation of .1 m or rotation of
.1 rad by default (`keyframe_translation_m`, `keyframe_rotation_rad`). Holding a
keyframe across small motions avoids accumulating independent frame-to-frame
noise at every update. Each alignment starts with the last accepted pose and
available integrated rotation; poses compose from the reference pose rather
than repeatedly adding the same relative transform.

Sparse depth, rank deficiency, ill-conditioned geometry, large ICP steps and high
residuals reject frames. On failure the returned pose is stale, the accepted pose
and reference stay unchanged, and `can_integrate` is false. A later frame can
recover if it remains inside the capture range of the last accepted reference.
There is no guarantee of recovery after large motion or loss of scene overlap.
Finite, strictly increasing update timestamps, valid rigid poses and correctly
shaped depth/IMU vectors are required. Repeated structures and dynamic objects
can still defeat local alignment; robust SLAM and real-data evaluation remain
future work.

## Occupancy evidence

`OccupancyMapper(resolution=.2, origin=(0,0,0), shape=None)` maintains sparse
signed log odds in world voxel indices. Supply a three-integer `shape` to bound
the grid. Otherwise negative indices are supported and storage grows only with
observed cells. `integrate(depth_m, camera, pose)` expects an **accepted** pose;
the replay caller must check `can_integrate` before invoking it.

Each valid depth pixel is backprojected using optical Z and transformed into the
world. Voxel DDA traverses the ray from the camera to that measured surface;
traversed cells receive free evidence, and the endpoint receives occupied
evidence. It never clears cells behind a hit. Bounded maps clip rays at their
bounds even when the camera lies outside. An endpoint beyond the bounds adds no
false occupied cell at the boundary. Cells touched only at an edge/corner do not
receive free evidence. NaN pixels add no observation, including no maximum-range
free ray.

A cell receives at most one update per frame. Occupied observations take
precedence over free rays in that frame. Defaults are +0.85 occupied, −0.4 free,
clamped to ±4; repeated later free measurements can clear an old obstacle.
Positive evidence exports label 2, negative evidence label 1, and absent or
numerically zero evidence remains unknown. Optional `max_rays` deterministically
subsamples pixels for slower CPUs; the default integrates all valid pixels.

`integrate` returns ray/update counts and current map counts. `counts()` reports
observed/free/occupied cells and frames integrated, plus unknown cells for a
bounded grid. `export()` returns the `voxel_map` object compatible with
`ares.report.v1`: finite `resolution`, `origin`, `shape`, and sorted
`cells: [[x,y,z,label], ...]`. It does not contain its own schema-version field
because the enclosing report carries the version. A sparse export shifts its
origin to the minimum known indices so exported indices remain nonnegative;
successive sparse export origins can therefore differ. An empty sparse map has
shape `[1,1,1]` and no cells. No floor plan, rooms or oracle occupancy are invented.

The focused unit tests independently raycast moved cameras inside an analytic
room and check translation/rotation recovery, body-gyro extrinsics, planar
degeneracy, dropout recovery, ray occlusion, evidence accumulation, outside-grid
clipping, sparse export and the ground-truth file firewall:

```bash
.venv/bin/python -m unittest tests.test_recording tests.test_odometry_mapping -v
```

Without NumPy these test modules skip, keeping the standard-library Phase 0/1
suite usable. The implementation makes no onboard-rate or real-place claim.
