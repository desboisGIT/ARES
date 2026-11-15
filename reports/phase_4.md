# Phase 4: trained flight-policy prototype

**The software pipeline runs and two policies are trained. Phase 4 is not
complete: primary waypoint success is 91.8%, below 99%; the coupled braking
probe fails, and onboard inference is unmeasured.** These checkpoints are for
simulation research. Earlier real-recording, onboard and human gates remain open.

## Frozen held-out comparison

Both methods use the same depth/state network, nominal geometric controller,
initial seed and 409,600 training transitions. The learned component is a
three-axis acceleration residual; the complete policy outputs physical thrust
and body-rate commands. This is not a comparison of purely learned controllers.

| Policy | Test tasks | Success | Collisions | Timeouts | Median successful time | P90 successful time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Differentiable BPTT | 1,000 | 918 (91.8%) | 82 | 0 | 2.05 s | 2.87 s |
| PPO comparison | 1,000 | 409 (40.9%) | 551 | 40 | 2.73 s | 4.55 s |

Failures stay in the success-rate denominator. Times include successful tasks
only. A task requires reaching both waypoints, with terminal speed below 0.8 m/s
and position within 0.35 m of the second waypoint. Evaluation uses exact continuous
sphere/voxel collision checks; failed contacts are not discarded.

Training cycled train apartments 100–109. The policies were frozen to exclusive
checkpoint files and reloaded before generating test apartments 60000–60015.
Each test scene supplied up to 64 independent two-leg tasks. The two learners
used identical task, profile and measurement-noise seeds. The common cohort
digest is `84a4054d7c41a414e45601689f331e5a1ffb396bd603251f68eb3eb2d5916ec8`.
No policy tuning followed this held-out run.

| Profile probe | BPTT successes | PPO successes |
| --- | ---: | ---: |
| All parameters at their low endpoints | 55 / 64 | 11 / 64 |
| All parameters at their high endpoints | 51 / 64 | 31 / 64 |
| Independently sampled interior parameters | 812 / 872 | 367 / 872 |

All 29 profile parameters and 35 scalar dimensions are sampled. The endpoint
audit covers every declared component; finite probes do not exhaust all joint
combinations or establish real-aircraft robustness. Task routes include turns,
reversals, height changes and furniture, with body-clear segment setup. They
are not a narrow-gap skill or full exploration benchmark.

## Speed and clearance

The requested speed limit was 1.5 m/s. It is a policy objective, not a velocity
clamp on the rigid-body simulator.

| Metric | BPTT | PPO |
| --- | ---: | ---: |
| Median peak speed | 1.475 m/s | 1.310 m/s |
| Maximum speed | 2.020 m/s | 2.195 m/s |
| Runs exceeding the requested speed limit | 454 / 1,000 | 165 / 1,000 |
| Minimum swept clearance | −0.0144 m | −0.0207 m |

Negative clearance belongs to collision failures. Positive distances beyond the
local query cap are explicitly marked lower bounds in individual records. Speed
overshoot and contacts remain material failures to address; the training loss
does not turn either into a hard safety constraint.

## Coupled shield failure

The geometric shield has 15 passing adversarial/unit cases, including unknown
slabs, thin obstacles, diagonal contact, latency growth, estimation uncertainty
and independent bounded braking trajectories. Its certificate is conditional
on map, estimation, acceleration and straight-braking bounds.

Those bounds have not been established for this quadrotor/controller. The
coupled probe deliberately passes `assumptions_valid=False`, so every check
requests active braking and none claims certification. At 100 Hz it produced
3,716 interventions and **three collisions in four distinct runs**, with no
waypoint successes. Maximum speed was 0.191 m/s and minimum clearance −0.000511 m.
The BPTT and PPO probe traces are identical because uncertified braking bypasses
the learned request; they are not eight independent safety experiments.

This is a failed integrated failsafe, not a successful collision-prevention
gate. Nominal hover/acceleration conversion and randomized state errors do not
enforce the assumed minimum braking acceleration. Low-speed drift can therefore
reach a surface despite the abstract geometry tests passing. The next safety
work is trim/integral or equivalent robust low-level control, explicit tracking
error bounds, and repeated adversarial coupled tests. No empirical probe here
establishes a universal guarantee over preventable collisions.

## Dynamics, randomization and training

The BSD DiffAero source is pinned to
`291ea14196aefbebcf7387dd71f7e096c83878b7`, with attribution and patch notes in
`ares/vendor/diffaero/`. Adapter `ares.diffaero.v2` uses full 13-state rigid-body
dynamics, per-episode mass/inertia/drag/gravity/thrust authority, a first-order
thrust actuator, and a rate loop at 500+ Hz. The policy runs at 60 Hz.

The rate controller uses fixed nominal inertia while physical inertia is
randomized. This avoids an ideal calibrated servo canceling the very uncertainty
under test. CPU/CUDA tests verify distinct low/high-inertia responses and finite
gradients. Compiled CUDA dynamics match eager states and action derivatives;
CPU uses eager execution. A local dynamics-only B64 timing improved about 4.95×;
that is not a whole-training or real-flight throughput claim.

Episodes start level and at exact hover, with actuator thrust equilibrated to
the sampled weight. The initial previous-command observation reflects that
setup trim. Takeoff and hover-trim identification are not tested here.

BPTT used 200 optimizer updates, 32-step truncation windows and batch size 64.
PPO used 200 rollout updates and 6,400 minibatch optimizer updates, with four
epochs per rollout. Both saw 409,600 transitions; compute budgets differ.
Recorded optimizer-loop times were 191.62 s and 107.12 s respectively, excluding
outer scene preparation/evaluation. Both have finite, nonzero sensor-encoder
gradients. Full loss/gradient/reward curves are retained alongside weights.

The current trainer resets the entire batch when any drone terminates. PPO
bootstraps surviving and truncated traces correctly, but later turns receive
less complete-trajectory experience. Longer-horizon training and per-drone reset
support are concrete next training improvements. They should use validation
scenes, followed by a fresh held-out cohort rather than tuning on this one.

Depth is 24×32 with per-episode exact-DDA/noise parameters. The range ring uses
measured-map memory and treats unknown space as blocked. Policies receive no
hidden occupancy, material labels or sampled physical parameters. Sensors/map
updates are detached; BPTT differentiates through physical dynamics and state
feedback. Estimated state, capture cadence and battery consumption remain
explicit approximations, not calibrated Stage A hardware models. The complete
parameter wiring and limits are in `docs/flight_workflow.md` and
`docs/flight_sensors.md`.

## Inference and verification

On the desktop RTX 5070 Ti, single-observation inference after warmup measured:

| Policy | Median | P90 | P99 |
| --- | ---: | ---: | ---: |
| BPTT | 0.688 ms | 0.713 ms | 0.881 ms |
| PPO | 0.690 ms | 0.711 ms | 0.950 ms |

Each row has 100 synchronized measurements and includes the network plus
geometric action conversion. It excludes sensing, estimation, shield, transport
and scheduling. Python 3.14.4, Torch 2.11.0+cu130 and the available desktop GPU
were used. These measurements cannot pass the onboard 5 ms gate.

The local suite passes 276 tests, including real CUDA dynamics/sensor/collision
checks, finite differences, compiled/eager gradients, delayed-action causality,
PPO time-limit bootstrapping and frozen-artifact validation. Seven included
pipeline checks verify that the primary BPTT reliability gate is
separate from the PPO comparison and hardware gates. The earlier `make build`
checks and paired 10% performance guard pass. CPU training, frozen evaluation,
vendor-license resources and CuPy independence were verified from an installed
0.5.0 wheel outside the checkout. The CPU-flight Docker build ran 273 tests
(nine skipped), all previous phase checks and the Phase 4 smoke; its nonroot
runtime smoke passed. Optional CPU-flight CI is configured; remote CI execution
is not claimed. Final hashes and checks are in
`reports/phase_4_artifacts/verification.json`.

The source/dependency inventory is updated in `LICENSES.yaml`. Unresolved
third-party GPU binary redistribution audits are explicitly marked false;
commercial deployment clearance is not claimed.

## Artifacts and reproduction

The machine-readable result is `reports/phase_4.json`; it keeps the failed
reliability and shield gates distinct from pending onboard timing. The frozen
artifacts are under
`reports/phase_4_artifacts/runs/8aa0795886404a1299dc0d5f6cd678ca/`:

- `bptt.pt`: SHA-256 `19694a4fd15348bfbb714a6faa51f672b626331185035ce423280264b8efd54e`.
- `ppo.pt`: SHA-256 `1945e6a05b2736c2801acde3e2e63219f9c951124bcb12dd0ac17c6dfdc6a70b`.
- Per-method training curves and full per-task results in `result.json`.

Checkpoints contain profile/config/runtime-source hashes and are loaded with
`weights_only=True`. Repeated invocations use new directories and do not replace
previous weights. The old ideal-inertia pilot is marked superseded and is not
used as evidence for this corrected runtime.

```sh
make phase4-smoke
make phase4
.venv/bin/python -m ares evaluate-flight CHECKPOINT.pt --device cuda --split validation --episodes 100 --output reports/flight_validation.json
```

`make phase4` exits nonzero while the 99% primary reliability gate fails;
`--smoke` checks software integration, and `--require-complete` requires all
phase gates. Phase 4b maneuver skills have not been started.
