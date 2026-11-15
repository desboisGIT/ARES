# Profile v2 sensor assumptions

`ares/data/configs/drone_sim_v2.json` uses `ares.profile.v2` and retains every
legacy parameter and its nominal/range values. The existing
`ares/data/configs/drone_nominal.json` remains the frozen v1 fixture. Consumers
must select v2 explicitly; v1 rejects the additional v2 fields. All v2 fields are
required. Each parameter records a nominal value and an inclusive randomization
range; these ranges describe simulation assumptions, not confidence intervals or
measured hardware tolerances. No real sensor recording or fitted hardware
calibration is supplied by this profile.

| Added parameter | Unit | Nominal | Range |
| --- | --- | --- | --- |
| gravity_mps2 | m/s² | 9.80665 | [9.78, 9.83] |
| depth_min_range_m | optical-axis m | 0.2 | [0.15, 0.3] |
| depth_noise_quadratic_per_m2 | m/m² (1/m) | 0.001 | [0.0005, 0.002] |
| depth_edge_width_px | integer pixels | 1 | [0, 2] |
| depth_edge_threshold_m | m | 0.2 | [0.1, 0.4] |
| max_range_dropout_probability | probability | 0.15 | [0.05, 0.3] |
| optical_flow_noise_std_mps | m/s per sample | 0.05 | [0.02, 0.1] |
| range_noise_std_m | m per sample | 0.02 | [0.01, 0.04] |
| imu_rate_hz | Hz | 200 | [100, 400] |
| optical_flow_rate_hz | Hz | 100 | [50, 100] |
| material_hole_probability | five probabilities | [0, 0.15, 0.25, 0.65, 0.8] | [[0, 0.01], [0.05, 0.3], [0.1, 0.5], [0.4, 0.85], [0.6, 0.95]] |

Material vector order is normal, shiny, black, glass, mirror. All nominal values
and bounds must be finite numbers; booleans are rejected. Gravity, minimum range,
edge threshold and sensor rates are positive. Noise coefficients and edge width
are nonnegative. Edge width requires integers in [0, 32], including its bounds;
0 disables edge masking. Probabilities and all probability range bounds lie in
[0, 1]. Material vectors and lists of component ranges have exactly five entries.
The maximum randomized minimum range must remain strictly below the minimum
randomized maximum range. Existing FOV requirements remain positive and strictly
below pi radians, including the entire randomization range.

Depth is absolute optical-axis Z in metres, with camera axes right/down/forward.
`depth_max_range_m` is an absolute Z cutoff, not Euclidean ray length, a delta, or
a normalization factor. The noisy sensor accepts Z in (depth_min_range_m,
depth_max_range_m]; an out-of-range measurement is missing (NaN). Existing base
noise `depth_noise_std_m` and the new coefficient give
`std(z) = sqrt(base² + (quadratic*z²)²)`. Independent range dropout probability is
`max_range_dropout_probability * (z/depth_max_range_m)²`; material holes and edge
holes add separate missing-return mechanisms. Width 1 masks both pixels adjacent
to an edge; greater widths extend that band. See [sensor_models.md](sensor_models.md)
for the full simulation and fitting conventions.

World Z points up; gravity acceleration is [0, 0, -gravity_mps2]. IMU reports body
specific force, including positive upward force at rest with aligned axes. IMU,
flow and range noise parameters are per-sample standard deviations; changing dt
or a declared sampling rate does not automatically scale their variance.
`imu_rate_hz` and `optical_flow_rate_hz` are explicit scheduling metadata for
callers; a `SensorSuite` method call produces one sample and does not itself run
a timer. Optical flow assumes a planar textured floor and receives body velocity
and perpendicular height rather than inferring image motion.

The depth frame clock uses existing `depth_fps`: frame i is captured at
`i/depth_fps` seconds and becomes available at capture time plus
`compute_latency_s`. These are deterministic simulated times; a declared 90 Hz
sensor does not assert that the renderer achieves 90 Hz wall-clock throughput.
Latency is not added to the capture timestamp and does not alter ray geometry.

Legacy v1 retains its original schema and file. When a sensor API consumes v1,
its explicitly documented compatibility defaults are gravity 9.80665 m/s²,
minimum range 0 m, quadratic depth coefficient 0.001 m/m², edge width 1 pixel,
edge threshold 0.2 m, maximum-range dropout probability 0.1, material holes
[0, 0.15, 0.1, 0.7, 0.9], optical-flow noise 0.02 m/s/sample, range noise
0.01 m/sample, and IMU rate metadata 200 Hz. V1 has no optical-flow-rate field
or automatic flow timer; the caller determines invocation cadence. V1 depth
FPS, depth maximum range, base depth noise, IMU noise, drift and latency continue
to come from its original required fields. V2 does not use these compatibility
defaults to fill missing profile fields.
