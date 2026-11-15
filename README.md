# ARES

Source is publicly viewable with rights reserved; see [LICENSE.txt](LICENSE.txt).
Third-party terms are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Foundations and procedural worlds for the autonomous indoor mapping project in
the [build plan](docs/build_plan_v3.md). Phase 0 provides deterministic contracts
and scoring. Phase 1 adds furnished rooms, apartments, corridor-based building
floors, a recorded-occupancy-map importer, and a human floor-plan review gallery.
Phase 2 now provides CPU/CUDA depth simulation, sensor/noise models, calibration
tools, recorded-data depth/gyro odometry and voxel mapping, and a handheld rig
recorder. Phase 3 adds a batched point-mass world, a classical exploration
baseline, a local first-person gamepad/keyboard viewer, and log replay.
Real human baseline timings are pending; UI tests do not count as human runs.
Real captures and onboard measurements remain pending.

## Phase 4 flight-policy work

Phase 4 adds a pinned BSD DiffAero full rigid-body model, a 500+ Hz body-rate
loop with motor lag, measured-depth/map observations, differentiable training,
PPO comparison, episode randomization and a conditional stopping-tube shield.
The same controller interface produces thrust in newtons and body rates in
radians/second. The repaired policy uses observed-state adaptive control and
shared `braking_v2` waypoint guidance plus a learned depth/state residual.
BPTT is the primary learner; PPO remains a comparison using the same architecture.

```sh
.venv/bin/python -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -e '.[flight,cuda]'
make control-check
make control-validation
make phase4-smoke
make phase4-validation
make phase4
```

CPU-only users can install Torch from `https://download.pytorch.org/whl/cpu`
instead. Validate the controller before training: `make control-check` runs eight
CPU cases, and `make control-validation` runs the 512-case CUDA development gate.
`make phase4-validation` trains both learners on the selected configuration and
checks 256 validation tasks starting at seed 46000. `make phase4` runs 50 updates
with 64 environments and 128-step windows, residual acceleration 0.5 m/s²,
residual penalty 1, and guidance speed fraction 0.95, then evaluates both frozen
methods on 1,000 identical test tasks starting at seed 80000. The mission speed
limit remains 1.5 m/s. This reproduces the sealed cohort: 80000–80015 is consumed
and must not be used for further tuning. Episodes reset independently; curriculum
stages retain the optimizer. New reports are `reports/phase_4_gate_development.json`
and `reports/phase_4_gate_runtime.json`; prior repair evidence is preserved.
The CPU integration report is `reports/phase_4_repair_smoke.json`. Full runs can
take several minutes on the desktop GPU. Curves, immutable weights, source hashes
and individual scores use distinct run directories beside each output report.

The CPU smoke verifies integration and leaves reliability/hardware gates false.
The 99% reliability target is a measured gate, not assumed from successful
training. The shield's geometric certificate requires acceleration, latency,
tracking and estimation bounds that have not been validated on this aircraft;
its current coupled simulation probe uses uncertified braking. Desktop inference
timings cannot pass the onboard 5 ms requirement. Phase 4b maneuvers are separate.

Frozen weights can be evaluated without training or modifying them:

```sh
.venv/bin/python -m ares evaluate-flight CHECKPOINT.pt --device cuda --episodes 100 --split validation --output reports/flight_validation.json
```

Use `make phase4-container` for a CPU Torch image; its default command remains
the lightweight Phase 3 smoke. Run the Phase 4 check explicitly with
`docker run --rm ares:phase4 python -m ares phase4 --smoke --output reports/phase_4_runtime.json`.
The earlier stages remain usable without Torch. See
[flight training](docs/flight_training.md), [DiffAero integration](docs/diffaero_integration.md)
and [shield assumptions](docs/safety_shield.md).

The portable conditional CPU backup component is available through
`make verified-safety-smoke` and `make verified-safety-validation` (set
`SHIELD_OUTPUT` to a fresh result path if desired). See
[verified backup SIL](docs/verified_safety_sil.md) for installation, bounded-service
assumptions and retained timing failures. Its component pass does not close the
full shielded learned-mission or Phase 4 software gate.

The selected BPTT policy passed the learned waypoint gate with **1000/1000**
successes, zero contacts and zero speed violations on the fresh 80000 cohort.
The same-interface PPO comparison completed **994/1000**, with six contacts and
five speed violations; the untrained reference completed **1000/1000**. These
finite simulation measurements do not close the conditional shield or actual
onboard timing gates. See the [Phase 4 learned gate evidence](reports/phase_4_gate.md).

The [earlier repair evidence](reports/phase_4_repair.md) and
[historical Phase 4 results](reports/phase_4.md) remain preserved with their
original failed candidates and cohorts.

## Phase 3 preparation

Install the simulation extra for CPU checks, or the CUDA extra for the full local
benchmark. `make check` includes a bounded CPU smoke and preserves earlier phases.
The full GPU target evaluates 100 held-out apartment episodes with four workers:

```sh
.venv/bin/python -m pip install -e '.[cuda]'
make phase3
.venv/bin/python -m ares fly --seed 50000 --split test --backend cuda --actor human --log-dir logs/manual
.venv/bin/python -m ares view-log LOG_PATH
.venv/bin/python -m ares human-benchmark logs/manual/*.json.gz --baseline reports/phase_3.json --output reports/phase_3_human.json
```

`make phase3` writes `reports/phase_3.json`. The local browser viewer starts on
loopback and supports keyboard or gamepad input; click Start controls to begin
mission timing. After starting, braking does not pause the clock. Replay includes
a frame slider and supports older logs without depth. See
[manual flight and replay](docs/manual_flight.md). Human runs must be flown and
recorded explicitly; no human benchmark times are fabricated by the smoke test.
Train scenes are available for practice; the human aggregation above accepts
the same test cohort, profile and world settings as the classical benchmark.
The local RTX 5070 Ti run measured 3.00 million individual environment steps/s
with 4,096 environments; the frozen explorer passed 100/100 held-out apartments
with zero collisions. See the [Phase 3 evidence](reports/phase_3.md) for timing,
scope, fixed seeds and the pending human gate.

## Phase 2 preparation

`make build` prepares a local `.venv`, runs CPU checks and builds the wheel.
For the GPU benchmark, install the optional CUDA extra and run:

```sh
.venv/bin/python -m pip install -e '.[cuda]'
make phase2
```

This measures 4,096 parallel 24×32 depth cameras and saves
`reports/phase_2.json`. The GPU renderer uses exact voxel DDA and an on-device
noise/dropout kernel; a CPU reference checks its geometry. Sensor parameters are
explicit in `ares/data/configs/drone_sim_v2.json` and remain unfitted assumptions.

```sh
.venv/bin/python -m ares simulate-recording --frames 24 --output recordings/synthetic-demo
.venv/bin/python -m ares replay recordings/synthetic-demo --output reports/synthetic_replay.json
.venv/bin/python -m ares replay recordings/place01 --output reports/place01_replay.json
```

The same runtime replay handles synthetic and real recordings and never opens
the separate evaluator ground-truth file. Its estimator is a local depth/gyro
baseline, with explicit tracking-loss detection. Full stereo VIO, accelerometer
translation/flow fusion and loop closure are future work. CPU replay needs only
NumPy and does not import CuPy or the camera SDK.

When the Stage A rig is available, install the optional recording extra in a
vendor-supported environment and follow [the capture procedure](docs/handheld_rig.md).
The recorder retains native stereo/IMU bags and writes normalized depth/IMU logs,
using bounded disk spooling for minute-long routes. It has not been tested on
physical hardware. For an imported room with independently registered poses:

```sh
.venv/bin/python -m ares compare-depth recordings/place01 --scene measured_room.json --poses registered_poses.npz --pairs-output reports/pairs.npz --output reports/depth_comparison.json
.venv/bin/python -m ares calibrate-depth reports/pairs.npz --source real --output reports/real_noise_fit.json
```

See [Phase 2 evidence](reports/phase_2.md), [sensor models](docs/sensor_models.md),
[recording format](docs/recording_format.md), [odometry/mapping](docs/odometry_mapping.md),
[profile v2](docs/profile_v2.md), and [comparison/validation](docs/phase2_validation.md).
The preparation command exits successfully for passing software checks;
`--require-complete` also requires the real-data/onboard gates and currently fails.

## Phase 1

Generate a scene, import an annotated recording, or run the complete corpus check:

```sh
python3 -m ares generate --seed 42 --kind apartment --split train --output reports/apartment.json
python3 -m ares generate --seed 42 --kind floor --split test --output reports/floor.json
python3 -m ares import-map ares/data/recorded_maps/synthetic_enclosed_room.json --output reports/imported.json
make phase1 WORLD_WORKERS=8
```

The full check generates and validates 10,000 scenes and writes
`reports/phase_1.json`, a reproducible corpus manifest, and an offline gallery of
50 randomly chosen training scenes at
`reports/phase_1_artifacts/review/index.html`. The corpus spans all three world
types, all five sensing materials, open/closed doors, and lit/dark rooms. Dataset
partitions are assigned by occupancy hash, so changing materials or lighting
cannot move the same collision geometry into a held-out split.

The current 50-plan gallery was approved by the user in chat; Phase 1 passes.
The attestation is bound to its exact scene hashes. A changed gallery needs a new
review. Existing review evidence can be supplied when repeating the corpus check:

```sh
python3 -m ares phase1 --workers 8 --review /absolute/path/floor_plan_review.json --require-complete
```

The review is bound to scene hashes and the exact gallery. No approvals are
preselected. Normal Phase 1 runs exit successfully when software gates pass and
report the human gate separately; `--require-complete` also requires the human
gate. `--smoke --count 30` exercises integration without satisfying the full gate.
See [world generation](docs/world_generation.md), [import format](docs/recorded_maps.md),
[review instructions](docs/floor_plan_review.md), and [Phase 1 evidence](reports/phase_1.md).

## Run

With Python 3.11+ and pip available, build and check everything:

```sh
make build
```

This creates `.venv` when needed, installs the simulation extra there, runs the
tests, Phase 0 gate, paired benchmark, Phase 1 sample, Phase 2 CPU checks and Phase 3 CPU smoke, then
creates `dist/ares-0.6.0-py3-none-any.whl`. NumPy supports Phase 2 and SciPy supports Phase 3; CUDA and the
camera SDK are separate optional extras. Phase 0/1 retain their standard-library
runtime. Internet access may be needed initially. No GPU is needed for this build.
An explicit `PYTHON` override uses the caller's prepared environment.

```sh
python3 -m ares validate
python3 -m ares phase0 --seed 42 --output reports/phase_0.json
python3 -m ares benchmark --check
python3 -m unittest discover -s tests -v
```

The gate writes its input scene/profile, scoring configuration, results, seed,
and artifact hash to `reports/phase_0.json`. Its sibling
`reports/phase_0_artifacts/` contains the mission report, a runtime log, and an
explicitly named evaluator log. Identical input and seed produce identical
mission, log, result, and gate JSON; wall-clock benchmark timings are separate.

For a container build, use `make container`, then
`docker run --rm ares:phase3`. Container execution needs a running Docker daemon.
The container runs as a nonroot user. CI checks Python 3.11 and 3.12 and uploads
reports and the wheel. A separate CI job checks the full 10,000-scene corpus.
The container's default command runs the Phase 3 CPU smoke and writes a fresh
`reports/phase_3_runtime.json` report. CI results
require the workflow to run on a remote repo.

## What passes the gate

The packaged scene has three enclosed rooms and two open doors, with a triangle
collision mesh generated from its 20 cm voxel grid. Both representations are
checked for exact agreement in the fixture tests. Scene/profile readers reject
malformed data with a field-specific reason. All random streams take an explicit
seed and name, independent of Python's global random state and salted hashes.

The scorer requires at least 95% supported voxel coverage, mean free/occupied
floor-plan IoU of at least 0.80, all reachable rooms, zero collisions, an episode
within the timeout, and a safe virtual landing. A successful evaluator witness
proves that the scorer can accept an episode. The four attack policies instant
report, endless hover, leaving the map, and an omniscient forged map must fail.
Dense training reward lives separately and cannot certify success.

The witness traverses evaluator-known free space. Its time is a harness fixture,
not an exploration or flight performance result. The evaluator checks swept
spheres continuously between samples, preventing voxel-wall tunnelling.
Only evaluator-executed observations support a policy's report; self-reported
coverage, room IDs, time, and landing claims cannot establish success. The
in-process library assumes the evaluator owns its objects; it does not sandbox
malicious Python code. Runtime log validation rejects the optional ground-truth
block unless explicitly authorized by evaluator tooling.

See [the contract definitions](docs/contracts.md),
[scoring assumptions](docs/scoring.md), [Phase 0 evidence](reports/phase_0.md),
and [research references](docs/references.md). Later phases need realistic camera
and dynamics models and hardware measurements before their gates can be met.

## Benchmarks and licenses

CI measures a versioned axial-scan workload against its frozen, equivalent
reference on the same runner. A normalized slowdown over 10% fails the check.
Baseline/reference changes require an explanation in the PR; this initial
baseline does not measure GPU throughput or full-stack latency. See
[benchmark policy](benchmarks/README.md).

`LICENSES.yaml` records dependencies, research references, and unresolved license
checks. No project source license has been selected. Unverified components are
marked `commercial_ok: false`; this flag also records the project's permissive
deployment policy, rather than asserting GPL prohibits commercial activity.
No external research code is incorporated into this foundation.
