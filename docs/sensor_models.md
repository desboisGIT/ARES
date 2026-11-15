# Sensor approximation conventions

These models are reproducible simulation assumptions, not evidence of real sensor
performance. NumPy is optional for the rest of ARES and required for these APIs.
Profiles may contain nominal parameter records or sampled scalar values. Profile
v2 requires every sensor parameter explicitly; only legacy v1 gets defaults.

`SensorSuite(profile, seed)` owns independent SHA-256 named NumPy generators for
IMU, flow, and odometry. No global RNG is read or modified. Calls advance their
own stream. Input vectors are finite three-vectors; dt is finite and positive.
Rotations must be proper orthonormal body-to-world matrices. World Z is up;
gravity is [0, 0, -gravity_mps2] m/s² (v1 default 9.80665). `imu(a_world, omega_body, R, dt)` returns
`accel_mps2 = R.T @ (a_world - gravity) + noise` and `gyro_rps = omega_body + noise`.
Independent Gaussian noise uses profile `imu_accel_noise_std` (m/s²/sample) and
`imu_gyro_noise_std` (rad/s/sample). `imu_rate_hz` is metadata for scheduling (v1
200 Hz); dt does not rescale per-sample standard deviations.

`optical_flow(velocity_body, height_m, dt)` returns body XY velocity in
`velocity_body_mps2d`, `range_m`, and `valid`. Height means perpendicular distance
to a planar, textured horizontal surface; no attitude compensation, image flow,
texture failure, or rate integration is simulated. Velocity noise is Gaussian
`optical_flow_noise_std_mps` (v1 0.02 m/s/sample), range noise
`range_noise_std_m` (v1 0.01 m/sample). Nonpositive, nonfinite, or beyond-range
heights/returns produce NaN values and valid=False. Range uses depth_max_range_m.

`drift(position, dt)` returns a noisy world position, anchored at its first call.
Each subsequent path increment adds `odometry_drift_per_m * distance` times the
average of a fixed random unit bias and a fresh random unit direction. Thus the
error norm is bounded by rate times cumulative sampled path distance, has a
linear bias component, and never grows while stationary. This is a bounded
simulation drift convention, not a covariance or full odometry estimator.

`DepthNoiseModel.from_profile(profile)` maps depth_noise_std_m to base_std_m,
depth_noise_quadratic_per_m2 to quadratic_std_per_m2, depth_max_range_m to
max_range_m, depth_min_range_m to min_range_m (v1 0), depth_edge_width_px to edge_width_px, depth_edge_threshold_m to
edge_threshold_m, and copies max_range_dropout_probability and
material_hole_probability. Legacy defaults respectively are 0.001 m/m², 1 pixel,
0.2 m, 0.1 and [0, 0.15, 0.1, 0.7, 0.9]. Material IDs are normal=0, shiny=1,
black=2, glass=3, mirror=4, no hit=-1. The supplied base/range values remain
required in v1. The dataclass defaults are convenience fixture assumptions.

`apply_depth_noise(depth, materials, model, seed, frame_index)` takes (H,W) or
(B,H,W) optical-axis Z metres and returns float32 with NaN missing returns.
Noise standard deviation is sqrt(base² + (quadratic*z²)²). Material holes are
Bernoulli per material; independent range dropout probability is
max_range_dropout_probability*(z/max_range)². Values outside (min_range,max_range] before
or after noise are missing, not clamped or returned as zero. Input NaNs remain
NaN. A discontinuity above edge_threshold_m or a hit/no-hit boundary invalidates
both adjacent pixels and a further edge_width_px-1 pixels in each image axis.
Width 0 disables edge masking; width is bounded to 32. Image boundaries do not
wrap and batches do not interact. Random streams name Gaussian/material/range,
seed and frame separately; the same seed/frame/input repeats exactly.

`fit_depth_noise(measured, reference, materials, source='synthetic'|'real')`
accepts matched 1D/2D/3D arrays. It fits nonnegative base² and quadratic² to
sample residual variance in 12 equal distance bins, weighted by count-1, using
the mean z⁴ within each bin. Bias is reported separately. It requires 200 returns,
a depth span of at least 0.5 m, max/min at least 1.5, and four populated bins with
20 returns each; otherwise it raises ValueError. Measured nonfinite/nonpositive
values are missing when reference is finite and positive with material 0..4.
Output schema ares.depth_noise_fit.v1 includes counts, bias, RMSE, bin statistics,
source/provenance and material missing fractions (null if material unobserved).
JSON has no NaN/Infinity. Missing fractions conflate edges, range and material
holes and must not be treated as isolated material coefficients. The estimator
cannot remove misregistration, reference error, motion, outliers or correlations;
real pairs require independent registration/quality review. Declaring synthetic
input never labels a fit as real hardware calibration.
