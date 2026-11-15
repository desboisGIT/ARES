# Flight depth sensor integration

`ares.flight_sensors.FlightSensors` provides the forward depth camera for the
Phase 4 full rigid-body environment. It reuses the exact Phase 2 voxel DDA ray
traversal and depth-noise model, extending CUDA kernels to independent episode
parameters for every drone in a batch. It does not use a hidden omnidirectional
range sensor or a ground-truth map as policy input.

```python
sensor = FlightSensors(scene, sampled_profiles, device="cuda", seed=1741)
frame = sensor.observe(position, quaternion, frame_index)
```

`sampled_profiles` is a nonempty list of numeric parameter dictionaries, one per
drone. Full ARES profile documents are also accepted and use their nominal
values. Camera dimensions must agree within the batch; different FOV, range and
noise settings are supported within the same batch. Defaults use a 32×24 image.
`position` has shape `[B,3]` and `quaternion` has shape `[B,4]` in xyzw order,
rotating body coordinates into the z-up world. `frame_index` is a nonnegative
integer identifying a reproducible noise frame. `noise=False` requests ideal
geometry for evaluation and tests.

The returned dictionary contains:

| Field | Shape | Meaning |
| --- | --- | --- |
| `depth` | `[B,2,H,W]` | Policy camera features: optical depth divided by that drone's maximum range, followed by a 0/1 validity mask. Missing depth is zero in the first channel. |
| `raw_depth` | `[B,H,W]` | Measured optical-axis depth in metres, with NaN for holes and out-of-range rays. |
| `material_ids` | `[B,H,W]` | Evaluator-only ideal hit-material labels; −1 means no valid ideal hit. These labels can remain present when noise removes the measured depth. |
| `material_ids_ground_truth_only` | scalar boolean | Always true; explicitly marks material labels as privileged metadata. |

Material IDs and any ideal geometry are excluded from policy features. Sensor
outputs are detached: the policy receives camera observations, while the flight
loss can differentiate through the rigid-body rollout. Voxel visibility and
material dropout themselves are not claimed to be differentiable.

## Camera frame and parameter wiring

The body frame is x forward, y left, z up. Optical camera axes are right, down,
forward. Camera-to-world rotation is body-to-world rotation multiplied by
`ares.depth.OPTICAL_TO_BODY`. Depth is optical Z, not Euclidean ray length: a
frontoparallel wall has constant depth across its image.

| Episode parameter | Consumption |
| --- | --- |
| `depth_width`, `depth_height` | Image size and pixel-center principal point. |
| `depth_horizontal_fov_rad`, `depth_vertical_fov_rad` | Independent horizontal/vertical focal lengths per drone. |
| `depth_min_range_m`, `depth_max_range_m` | Ideal-hit validity and measured-depth clipping; an occluding near object remains an occlusion even if too near for a valid depth. Maximum range also normalizes policy depth. |
| `depth_noise_std_m` | Additive Gaussian noise floor. |
| `depth_noise_quadratic_per_m2` | Depth-dependent noise combined in quadrature: sigma² = base² + quadratic² × Z⁴. |
| `depth_edge_width_px`, `depth_edge_threshold_m` | Missing measurements around validity changes and sufficiently large neighboring depth discontinuities, using the Phase 2 edge rule. |
| `material_hole_probability` | Independent hit-material dropout probabilities for normal, shiny, black, glass and mirror surfaces. |
| `max_range_dropout_probability` | Additional dropout probability multiplied by `(Z / max_range)²`. |

`depth_fps` and `compute_latency_s` are profile fields retained by the reference
renderer clock. `FlightSensors.observe()` does not impose a capture cadence or
delivery schedule. The flight environment owns that timing. In its current
approximation, the environment requests an image at each policy tick and queues
frames until each drone's sampled compute latency has elapsed. Sampled camera
FPS contributes to the effective measurement-age approximation
`1 / depth_fps + compute_latency_s`; this is not an independent asynchronous
camera capture clock. The environment also models sampled IMU/flow/range noise
and drift. It does not run the Phase 2 odometry estimator or calibrated
visual-inertial/optical-flow fusion.

## Execution and reproducibility

CPU execution delegates each episode's calibration and noise to the existing
`GroundTruthDepthRenderer` reference. The immutable scene occupancy/material
volume is shared between episode renderers to avoid copying a scene per drone.

CUDA execution traces a single batch with per-drone camera and noise arrays.
Torch poses and output buffers are shared with CuPy through DLPack. Kernels run
on an external view of Torch's current CUDA stream, including nondefault streams.
An event makes constructor-time device constants available before later calls
on other streams. CUDA does not silently fall back to CPU ray casting.

Noise is reproducible from seed, frame index, environment index and pixel.
Repeated calls with the same arguments reproduce the same measurements;
independent environments have independent streams. CPU and CUDA implement the
same noise distribution, while their random bit streams differ. Ideal geometry
and materials agree across backends. The per-environment CUDA noise extension
also matches the existing Phase 2 CUDA renderer using the same seed, frame and
environment identity.

`FlightSensors` does not update a map. `FlightEnv` integrates delivered measured
depth into its observed map using the recorded capture pose and a noise margin.
The environment derives shield ranges from that known map; unknown space limits
those ranges. These simulation observations, age approximations and noise
assumptions are not a calibrated real camera or onboard estimator validation.

Run `.venv/bin/python -m unittest tests.test_flight_sensors -v` for CPU geometry,
CUDA ideal/noise parity, nondefault-stream interoperation, per-episode material
holes and reproducibility checks.
