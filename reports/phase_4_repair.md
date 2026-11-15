# Phase 4 controller and training repair

This report tracks the repair of the failed controller and training setup in
`phase_4.md`. The historical report, its two checkpoints and its test cohort are
preserved. Phase 4 remains a simulation research prototype: the 99% learned-flight
gate, conditional shield bounds and onboard timing are separate requirements.

## Frozen final comparison

**The controller repair passes the measured simulation regression, but the learned
flight gate remains failed.** Both frozen policies were evaluated once on the
same 1,000 new test tasks in apartments 70000–70015, after validation. No further
source or weight tuning followed that test. The common dataset digest is
`4c8a435894d16c025ecf71493a81e8e7723001ec5c5f49d23554d2a8aeca7334`.

| Policy | Success | Collisions | Timeouts | Median successful time | P90 successful time | Runs over 1.5 m/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untrained repaired controller | 1000/1000 (100%) | 0 | 0 | 3.883 s | 4.668 s | 0 |
| BPTT learned residual | 858/1000 (85.8%) | 1 | 141 | 4.600 s | 5.822 s | 1 |

The reference has zero training transitions. Its result does not satisfy the
**learned** primary policy requirement. BPTT used 50 updates × 128 steps × 64 lanes
(409,600 transitions), with nonzero depth-encoder gradients. It reached 118/128
prospective validation tasks versus 128/128 for the reference; test performance
confirms the residual currently reduces reliability. Times exclude failures;
failures remain in the denominator. The untrained reference should remain the
simulation reference while the learning objective/residual authority is improved.

Maximum speed was 1.4894 m/s for the reference and 1.5205 m/s for BPTT. Minimum swept
clearance was 0.000271 m for the reference and −0.0000914 m for BPTT. The reference's
near-contact minimum is a material limitation despite zero measured collisions;
this does not establish a robust clearance reserve or a physical flight guarantee.
Endpoint and random-interior profile records, all failures and per-task metrics
are retained. This cohort is now consumed: further tuning must use validation and
then another fresh test cohort. The historical 60000–60015 cohort is unchanged.

The final BPTT checkpoint SHA-256 is
`9e686e2baac3825f0123837111469384c4a69d583505eb9910e83ca99b4bf3a1`. The reference checkpoint SHA-256 is
`4a321a607ee32541cb02c164806ecffc57a3e3626022d4657986792ab3408231`. Both use the 51-file frozen source manifest.
The combined machine-readable result is `phase_4_repair.json`.

## What changed

The old nominal .5 kg acceleration mapper could request almost exact hover thrust
when asked to brake a light vehicle downward at 1 m/s². Zero acceleration also
caused persistent vertical drift across randomized mass/gravity. The new
controller estimates a bounded acceleration disturbance from measured velocity,
orientation and issued commands. It uses fixed nominal constants; sampled plant
parameters are not controller inputs. Observer snapshots are explicit, detached
observation values, so repeated PPO evaluation cannot advance controller state.

Smooth braking transitions to a captured position hold with hysteresis. A separate
terminal-stop radius contract permits the small corrective reversals required by
hold; the old complete-stop geometry remains the default for existing callers.
This contract is conditional, rather than a certificate for this vehicle.

Each environment now resets independently, including its own observer, actuator
queue, sensors, map and episode clock. Surviving trajectories and random streams
continue. BPTT uses 128-step windows; declared scene stages retain one Adam
optimizer. PPO keeps independent terminal/truncation advantage boundaries.

The shared guidance function slows before corners using an explicit braking and
reaction budget. It retains the incoming route direction until handoff, avoiding
shortcuts through obstacles. The first waypoint tolerance is tightened from .35 m
to .10 m; the final .35 m / .8 m/s completion criterion is unchanged. Thus the old
and new controller validation cohorts have identical tasks and profiles, but the
handoff protocol is stricter. Existing checkpoints retain their original nominal
conversion and legacy guidance configuration; runtime hashes reject mismatched
end-to-end re-evaluation.

The adaptive mapper caps the world heading contribution at 1 rad/s before
transforming it to body rates. Tilt feedback and thrust keep their authority.
Paired validation replays showed that faster heading rotations could produce
lateral excursions near walls despite a counteracting acceleration request.
The cap is a development control choice, not a bound on all lateral transients.

Training clearance now uses exact free-space point-to-voxel-box distance minus
the physical body radius, with a piecewise position gradient. Nearest-box
selection is detached to keep the rollout graph small. The evaluator-only
obstacle reserve is fixed per episode to the smaller of .10 m and half the
minimum physical clearance of the two route segments. It is never a policy
input. Continuous swept collision evaluation is unchanged.

This fixes a concrete loss conflict: the old conservative distance field could
assign negative clearance to physically clear waypoints, while the fixed .25 m
reserve exceeded the route's available space. Among 640 sampled training routes,
442 first waypoints had negative surrogate clearance, but all had positive
physical clearance. Five routes had less than .10 m clearance; the minimum was
.01586 m. Tasks are retained, and nonpositive route clearance raises an error.

## Development evidence

The initial adaptive-controller experiment, before guidance repair, completed
104/128 validation tasks. Its runtime wheel is archived beside its experiment.
Its B64 evaluation cohort differs from the B32 controller comparison below.

On validation apartments 42000–42003, B32, the earlier untrained adaptive
controller completed 75/128 tasks with 53 collisions. Braking-aware guidance and
faster trim acquisition completed 125/128 with three collisions, no timeouts and
no speed-limit violations. These are untrained controller references, not learned
policy results. The dataset digest and per-task outcomes are in
`control_tracking_guidance_v2.json`. Remaining failures were replayed with all
sensors and exact CUDA dynamics; the diagnostic report preserves full traces.

The stronger trim observer reduced maximum acquisition drift from about .20 m
to .09 m in the bounded controller stress tests. Measurements are finite and
specific to the recorded noise, delay, attitude and profile populations.

With the yaw limit, the untrained controller completed 128/128 on that same
validation cohort, plus 128/128 on prospective validation apartments 45000–45001
(B64). Both had zero collisions and zero speed-limit violations. A preceding
100-update BPTT trial exposed the loss conflict above: 28/128 successes, five
collisions and 95 timeouts. That trial and its runtime wheel are preserved;
its weights are not the repaired candidate.

| Coupled controller regression | Initially valid | Valid failures | Outside envelope, retained | Outside-envelope collisions |
| --- | ---: | ---: | ---: | ---: |
| Level, 512 cases, seed 71 | 375 | 0 | 137 | 40 |
| 10° attitude extension, 128 cases, seed 89 | 99 | 0 | 29 | 9 |

Both runs used componentwise bounded position noise .01 m and velocity noise
.025 m/s, 30 ms extra command transport, 100 Hz checks, four-second braking/hold
trajectories and two-second acquisition. The invalid initial populations include
overlap/clearance, overspeed and insufficient stopping space. Their failures are
retained in full and do not establish collision prevention outside the envelope.
No valid case exceeded the predeclared .2 m terminal radius. Maximum observed
terminal displacement was .16337 m. Minimum observed active braking after the
fixed .4 s reaction cutoff was .88127 m/s², below the nominal 1 m/s² assumption.
Maximum post-cutoff tracking error was 1.13692 m/s². None of these finite extrema
is a universal bound; runtime shield assumptions remain invalid and no flight
certificate is issued.

The analytic stopping suite uses independent development geometry and bounded
state errors. The full flight environment also has Gaussian errors, drift, noisy
depth and tight clutter. A passing analytic regression therefore does not imply
the waypoint mission gate passes.

## Verification

The local suite passes 325 tests, including real CUDA dynamics, selective reset
graph continuity, immutable PPO observer snapshots, legacy checkpoint migration,
shared guidance loss and the terminal-envelope geometry cases. The 0.6.0 wheel
was installed outside the checkout with CuPy imports blocked: both CPU learners
trained, frozen checkpoints reloaded/evaluated, and the controller smoke passed.
The CPU-flight Docker build passed 322 tests (11 skips), and nonroot runtime
controller/training checks passed. The historical image tag is preserved.
Packaging evidence is in `phase_4_repair_packaging.json`.

The final yaw-limited control suite preserved all 640 earlier stopping trajectories
exactly. Its fixed +x gaze does not exercise large heading changes; those are
covered by the separate waypoint replays and rigid-body yaw/tilt regression.
The final full-environment probe ran 16 ten-second stop/hold episodes with noisy
depth, Gaussian state errors and map drift. It had zero collisions, minimum
clearance .14178 m and maximum speed .26256 m/s. Every one of 16,000 shield checks
intervened and zero commands were certified. It achieved no waypoint completions;
this probe measures failsafe holding, not exploration progress.

The final runner hashes every package Python file, including curriculum,
evaluation, rendering and generation code, and checks the manifest after training
and evaluation. Standalone evaluation also verifies checkpoint bytes before and
after running. Earlier experiments stored hashes of selected runtime files; they
are archived with their own runtime wheels.

## Open gates

The controller's minimum braking, tracking error, estimator error, latency and
long-term terminal hold still need enforceable bounds before enabling shield
certification. Real recordings and calibration, Stage A hardware, actual onboard
inference/latency measurements and human benchmark timings remain pending.
Phase 4b agile skills have not started.
