# Phase 2: simulation and handheld-rig preparation

**Software preparation and the GPU parallel-depth gate pass. Phase 2 remains
open for real recordings, calibration and onboard measurements.** The user
confirmed that Stage A hardware and recordings are not yet available and asked
to prepare the software/capture tools. Phase 1's 50-plan approval is recorded
against its gallery/scene hashes in `reports/phase_1.md` and the local review JSON.

## Gate status

| Gate | Status and evidence |
| --- | --- |
| GPU depth simulation for thousands of drones | **Pass:** 4,096 simultaneous 24×32 camera views, measured on the available RTX 5070 Ti. |
| Recorded-data odometry and mapping software | **Synthetic preparation passes:** one runtime reader/estimator/mapper handles both declared sources. A 24-frame noisy closed-loop witness tracks all frames and builds observed free/occupied voxels. |
| Clean maps from at least three real places, loop errors reported | **Pending:** no rig or real recordings supplied. Synthetic accuracy is not substituted for this gate. |
| Real depth-noise fit and imported-room agreement | **Pending:** fitting and registered comparison tools are implemented; current fit evidence is explicitly synthetic. |
| Rates on the actual onboard computer | **Pending:** no onboard ARM board supplied. Current timings describe this desktop only. |

The machine-readable report uses separate `software_preparation_passed`,
`gpu_thousands_gate`, `hardware_gates` and `phase_passed` fields. Current values
are true, true, pending, and **false**, respectively. CPU CI can pass preparation
while correctly leaving GPU/hardware/full-phase gates incomplete.

## Implemented

- CPU reference and batched CUDA voxel-DDA depth rendering, including camera
  FOV/intrinsics, optical-Z depth, frame scheduling and configured latency.
- Distance-dependent depth noise, material holes, thick-edge invalid bands,
  minimum/maximum range and range dropout. CUDA noise runs on device with explicit
  seed, frame and environment identities; split batches retain their streams.
- Profile-driven IMU specific force/gyro noise, planar optical-flow/range proxy,
  and bounded distance-dependent odometry drift. Profile v2 adds explicit sensor
  parameters while preserving all v1 contract behavior.
- A strict sensor-recording format, local CPU depth/gyro odometry, and signed
  log-odds occupancy mapping. Runtime IO never opens evaluator ground truth.
- Registered imported-room depth comparison with actual camera intrinsics, and
  constrained noise fitting with coverage, residual, missing-return and source
  diagnostics.
- An optional-SDK RealSense recorder preserving native stereo/IMU bags alongside
  normalized arrays. Disk spooling and eight-frame conversion avoid retaining a
  minute's depth in RAM. Startup waits for all selected streams' global clocks;
  later clock fallback is fatal. Native capture is still hardware-untested.
- Synchronous component timing, CPU build/container/CI checks, CUDA measurement,
  non-overwriting capture/pair outputs and staged synthetic-gate publication.

The estimator is an initial **depth/gyro baseline**, not full stereo visual-
inertial SLAM. Accelerometer translation/bias fusion, optical-flow fusion,
IMU-rate pose propagation, global relocalization and loop closure are not
implemented. Stereo data is retained in native bags for that work. This gap must
be addressed or explicitly justified by measured real-place results before
claiming the intended onboard tracking architecture.

## Measured GPU workload

Device: NVIDIA GeForce RTX 5070 Ti, approximately 16 GB; Python 3.14.4,
NumPy 2.5.4, CuPy 14.2.0; queried CUDA runtime/driver API version 13020.
Seed 1741; seven synchronized measured repetitions per mode after warmup.

| Metric | Ideal depth | Noisy sensor depth |
| --- | ---: | ---: |
| Batch size | 4,096 | 4,096 |
| Image size | 24×32 | 24×32 |
| Median batch duration | 1.074 ms | 1.101 ms |
| Rendered camera frames/s | 3.81 million | 3.72 million |
| Per-environment render rate | 931 Hz | 909 Hz |
| Valid-return fraction | 98.44% | 90.23% |

This is a **small-scene, low-resolution rendering benchmark**, with varied yaw
and nearby camera positions. It includes host pose validation/upload, allocations,
raycast, optional on-device noise and completion synchronization. Scene upload,
JIT warmup and output download are excluded. These are camera frames, not control
steps, exploration missions or full-stack throughput. More complex scenes and
real image sizes require their own measurements. The simulated camera clock is
still the configured 90 fps; benchmark speed does not change simulated time.

## Synthetic replay and fit

The positive fixture is a 20 cm-radius loop lasting about two seconds, with
40 cm excursion and small yaw changes. Its oblique initial view sees three
surface-normal families. The 24 recorded frames are an explicitly declared
sparse sampling of a 90 fps camera, with simulated 200 Hz IMU and 100 Hz flow.
The estimator receives only sensor arrays; the evaluator subsequently aligns
the initial origin/orientation for comparison. It does not correct estimated
poses or apply loop closure.

| Synthetic metric | Measured result |
| --- | ---: |
| Accepted frames | 24 / 24 |
| Position RMSE | 4.67 mm |
| Maximum position error | 10.08 mm |
| Closed-loop position error | 3.73 mm |
| Maximum orientation error | 0.00389 rad |
| Observed free / occupied voxels | 661 / 156 |

An initially straight-on view was weakly constrained in lateral translation.
Noisy/mixed normals could give a misleading algebraic full rank. The estimator
now checks plane-normal span and rejects that view; tests cover three independent
noise seeds. It also uses plane-fit normals, uncertainty-weighted gyro residuals,
trapezoidal gyro integration and local keyframes. Separate noisy analytic-room
tests verify more than 15 cm of motion within 1.5 cm error. None of these tests
establish real-camera accuracy.

The isolated synthetic fitting check uses 100,000 registered reference samples,
with 63,164 returned measurements. It recovers base standard deviation 0.010039 m
against 0.010 m input and quadratic coefficient 0.00098885 against 0.001 input.
Edge/range dropout is disabled only in that fitting fixture to isolate the model;
the GPU sensor benchmark enables the configured full model. The real noise model
remains uncalibrated, and material missing fractions must not be mistaken for
isolated material coefficients when other dropout mechanisms are present.

Desktop p99 execution times in this replay were approximately 5.19 ms odometry,
11.05 ms mapper (64 selected rays/frame), and 15.43 ms combined. A disabled
timing-only dummy policy has no actuator interface. These observations are
compute-budget checks with a small sample; they do not establish real-time
schedulability, native-resolution throughput or onboard ARM rates.

## Verification and artifacts

The local combined suite has **153 passing tests**, including actual CUDA parity,
noise-stream and uniform-endpoint checks. CPU-only environments skip GPU tests.
`make build` runs all installed tests, the earlier phase checks, paired axial
benchmark, Phase 2 CPU preparation and wheel packaging. The existing 10% paired
benchmark guard passes. Standard-library discovery preserves the Phase 0/1
workflow and skips optional NumPy suites.

Docker builds and its default nonroot CPU run passes. Runtime output uses a
fresh path rather than trying to rename baked artifacts across overlay layers.
An outside-repository wheel installation ran CPU preparation twice against the
same output and replayed all 12 frames while CuPy imports were deliberately
blocked. SDK-dependent capture has no device evidence; only its conversion,
clock-policy, argument/help and failure paths are tested. Remote CI execution
is not claimed.

Reproduce the measured GPU run:

```sh
make build
.venv/bin/python -m pip install -e '.[cuda]'
.venv/bin/python -m ares phase2 --backend cuda --batch-size 4096 --frames 24 --output reports/phase_2.json
```

`reports/phase_2.json` stores the full input scene/profile snapshots and versions,
seed, benchmark, synthetic evaluation and explicit pending gates.
`reports/phase_2_artifacts/` contains replay trajectory/map, noise-fit statistics,
benchmark, evaluator results, and a labelled synthetic sensor recording. Its
`ground_truth.npz` is separate from runtime `frames.npz`. Ownership hashes protect
previous outputs; unrelated, modified and symlinked recordings are refused.

For Stage A setup and real data collection, use `docs/handheld_rig.md` and
`scripts/capture_realsense.py`. Normalized recordings currently decompress into
RAM for replay; disk-backed capture does not imply streaming replay. A default
minute at 640×480/30 fps is roughly 2.2 GB of float32 depth before workspace;
larger captures need memory checks or shorter segments on the actual board.
Record three independently measured closed routes, retain the raw stereo/IMU
bags, fit and evaluate registered depth pairs, inspect all maps, and measure
onboard rates/latency/memory/thermal behavior before closing this phase.

No hardware was ordered, no flight-controller output is enabled, and no external
research code or pretrained model was imported. NumPy is the optional CPU
dependency; CuPy/CUDA are optional desktop simulation dependencies. License
inventory and distribution-review gaps are recorded in `LICENSES.yaml`.

Low-effort agents handled the recorder, bench utility, packaging and build checks;
medium effort handled sensor/calibration models and integration tests; high effort
handled odometry, geometry/parity review and preservation regressions. The main
agent retained GPU kernels, recording/replay orchestration, versioned contracts,
gate ownership and evidence reporting.
