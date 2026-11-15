# Phase 2 software and measurement boundaries

Phase 2 preparation has an executable CPU/CUDA check. It does not establish
three real-place maps, a real depth-model fit, or rates on a computer that has not
been supplied. `software_preparation_passed`, `gpu_thousands_gate`,
`hardware_gates`, and `phase_passed` are distinct JSON fields. CPU CI can pass
software checks with the GPU and hardware fields false. Synthetic or user-declared
`source="real"` data alone cannot turn a hardware gate true.

## Depth convention and parity

World/body axes are right-handed, z up, body x forward/y left. Camera optical
axes are x right/y down/z forward. The default optical-to-body rotation is
`[[0,0,1],[-1,0,0],[0,-1,0]]`. Poses map camera coordinates into world coordinates.
The renderer returns **optical-axis Z metres**, not ray length. Intrinsics derive
from profile dimensions/FOV; offline comparison can override them with the
recording's calibrated fx/fy/cx/cy.

`GroundTruthDepthRenderer` is evaluator-only. `DepthBatch.depth_m` is sensor data;
its separately named `ground_truth` block contains ideal depth/material IDs and
must not feed runtime estimation. Voxel DDA returns the first occupied surface.
A too-near obstacle blocks the ray even though its return is invalid; no reading
behind it is invented. No hit, out of range, or missing observations are NaN.
Frame timestamps are index/fps; availability adds the configured compute-latency
assumption. Calling a renderer faster does not alter that simulated clock.

CPU and CUDA ideal raycasts are checked against analytic rooms, occlusions,
outside-grid poses and each other. Full noise distributions use the same model
but backend-specific random algorithms. Within a fixed backend/versions and
explicit seed/frame/environment ID, sensor results repeat; CPU/GPU noise is not
claimed bit-identical. CUDA counters keep identities stable when batches split,
and its uniform sampler excludes 1 so probability-one dropout remains exact.

Noise grows with distance, has per-material holes, invalidates depth edges, and
includes range dropout. See `sensor_models.md` for equations and the limitations
of IMU, planar optical-flow/range and bounded drift approximations. Coefficients
in `drone_sim_v2.json` are assumptions, not D435i measurements.

## Synthetic positive and negative checks

The positive witness is a 20 cm-radius closed handheld-style loop lasting at least
two seconds, with a small yaw oscillation. Sparse frame sampling is declared by
its frame stride. Initial viewing yaw is 0.35 rad so front, side, and floor/ceiling
surface normals are visible. The estimator gets no ground-truth pose, map,
registration, or loop-closure correction. Evaluation aligns the initial origin
and orientation only after runtime estimates are finalized.

A straight-on view with only two plane-normal families is a negative regression.
Noise and mixed depth-edge normals must not manufacture its missing translation
constraint. The estimator rejects it and mapping skips the stale pose. Independent
analytic noisy-room tests also check real motion and keyframe transitions. Passing
one synthetic witness does not prove arbitrary-scene tracking or safe flight.

Runtime replay reads only `metadata.json` and `frames.npz`. Tests poison the
evaluator file and confirm unchanged estimated trajectory/map. Component and
combined execution timings are actual host wall times. Their target checks are
compute-budget checks, not proof of sensor update frequency, real-time scheduling,
IMU-rate pose propagation or onboard performance. Mapper timing uses 64 selected
rays per frame, and the dummy policy is disabled with no actuator interface.
If a supplied profile has zero gyro noise, replay applies a numerical 1e-9 rad/s
floor; the separately documented angle floor still limits the gyro residual.

The GPU benchmark measures one scene with varied camera yaw and nearby positions,
4,096 low-resolution cameras, fixed seed and synchronized completion. It includes
input upload, validation, allocations, raycasting, and on-device noise. Scene
upload, JIT warmup and output download are excluded. These are rendered-camera
frames per second, not control steps, missions or full-stack throughput.

## Registered real-room comparison and fitting

Export reference poses as NPZ with exactly `timestamps_s` and `camera_to_world`
(F,4,4), one calibrated camera pose per recorded depth frame in the imported
scene's coordinate frame. Timestamps must match exactly within 1 ns. Obtain this
registration independently and verify units/extrinsics; estimated poses from the
same system are not an independent accuracy reference.

`ares compare-depth` renders the imported scene with actual recording intrinsics,
reports MAE/RMSE/bias/missing fraction, and can export pairs. `ares calibrate-depth`
requires NPZ arrays `measured_m`, `reference_m`, `material_ids`. Use NaN reference
or material -1 to exclude uncertain regions before fitting. Pose/room/reference
errors, multipath and motion all contaminate residuals; a declared real source is
not independent validation of them. Fitting needs measured distance coverage and
returns explicit provenance/coverage diagnostics. Fit training captures and report
held-out residuals separately. No real pairs have been supplied in this session.

## Output preservation

Ordinary recording writes refuse existing contract files. Comparison pair writes
use exclusive creation on the exact supplied path, including extensionless paths.
Synthetic gate reruns require a matching ownership marker, full byte hashes,
regular files, and consistent synthetic metadata/manifest. Symlinks, changed or
unrelated artifacts are refused. New gate products are built in a temporary
sibling, then replace the owned directory only after successful generation and
evaluation. Previous artifacts remain intact if validation or building fails.

The retained native capture bag is never a gate output and is never replaced by
that workflow. Real captures and actual onboard rate/thermal/latency reports must
still be collected using `handheld_rig.md` before Phase 2 can pass.
