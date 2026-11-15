# Phase 4 gate status

**The learned waypoint reliability gate passes. The shield gate remains open.**
Actual onboard inference is deferred until hardware is available, as requested.
The project is not yet ready to mark all remaining Phase 4 requirements complete.

## Frozen held-out comparison

All policies were frozen before generating 1,000 test tasks in apartments
80000–80015. The cohort digest is
`9f502fe69698e76662a7f07868364aeb500de64815f2b27c567a1231cc2e0371`.
The all-low and all-high profile batches each contain 64 tasks, followed by 872
random-interior tasks. Failures remain in every denominator.

| Policy | Successes | Collisions | Timeouts | Median successful time | P90 successful time | Speed-exceeding runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BPTT learned residual | 1000/1000 (100%) | 0 | 0 | 3.950 s | 4.852 s | 0 |
| PPO comparison | 994/1000 (99.4%) | 6 | 0 | 3.600 s | 4.517 s | 5 |
| Untrained controller reference | 1000/1000 (100%) | 0 | 0 | 3.867 s | 4.667 s | 0 |

The primary BPTT result exceeds the required 99% measured success rate. Its peak
speed was 1.45583 m/s against a 1.5 m/s mission limit; minimum swept sphere
clearance was .01545 m. PPO peaked at 1.52430 m/s with minimum clearance
−.00220 m. The reference peaked at 1.45429 m/s with minimum clearance .01413 m.
Times include successes only. The primary does not outperform the strong
untrained reference, and no learned superiority is claimed.

The conditional independent-Bernoulli 95% interval for 1000/1000 is approximately
[99.63%, 100%]. Layouts are shared within batches; this interval is not an
unseen-layout or continuous-parameter guarantee. Finite low/high/interior trials
cannot exhaust every joint parameter combination.

## Changes and validation

The selected policy retains the .5 m/s² per-body-axis residual bound. BPTT adds
an explicit coefficient-1 squared-residual penalty to its differentiable physical
loss. PPO applies the same cost rate to sampled residuals, integrated using the
public timestep before GAE. Existing command-smoothness cost remains separate:
smoothness alone did not penalize sustained sideways bias. No privileged route
axis enters the policy or regularization calculation.

Shared guidance uses a .95 cruise-speed fraction, while the requested mission
limit and evaluator remain at 1.5 m/s. Policy and training labels use the same
configuration. Explicit braking authority is unchanged. Defaults preserve earlier
experiments, and each checkpoint serializes its selected settings.

Each learner received 409,600 training transitions: 50 updates × 128 steps × 64
environments. BPTT used 50 optimizer updates; PPO used 6,400 minibatch optimizer
updates. Architecture, initial seed, observation interface and dynamics match;
compute budgets differ. Both have finite, nonzero depth-encoder gradients.

Before testing, the primary passed 128/128 development tasks and 256/256
prospective validation tasks without contacts or speed excursions. A second
independent BPTT seed also passed 256/256. Weights, all 51 flight-runtime Python
sources, profile snapshots and the evaluation protocol were frozen before test
generation. The 80000 cohort is now sealed, alongside 60000 and 70000. No tuning
followed the new held-out evaluation.

## Conditional backup progress and open shield gate

The conditional full nonlinear CPU backup mathematics is now closed in
`reports/phase_4_shield_artifacts/cpu_rpi_certificate_v1/certificate.json`
(SHA-256 `7a90aad6cf0cdc9f92148be849f8bbf6d18609bf768aa5dc84970da4609407af`).
Its bounded-service contract, exact current quaternion abstraction, numerical
backend, queues and finite reservation horizon remain explicit assumptions.
This does not certify the original Gaussian observation runtime or a real sensor.

The selected bounded-service component run, `coupled_reanchoring_v1/retry_003`,
completed 8 × 1,000 steps: 8,000 positive conditional decisions, 6,181 anchor
accepts, 1,819 cached fallbacks, and zero membership, collision, unknown-entry or
queue misses. Decision-plus-acknowledgement time averaged 2.304 ms and peaked at
4.271 ms; waypoint-distance progress was .191–.297 m. These desktop timings
exclude sensing, plant simulation and the oracle, and are not universal bounds.

Three adversarial scenarios each completed 1,000 functionally valid steps and
refused 139 reserves touching unknown/occupied space. Minimum physical clearance
was .1586 m. One 12.128 ms corner timing miss remains retained, so the adversarial
timing gate did not pass. Fifteen bad-input preflight cases were refused with no
bad commands applied.

The new `adversarial_gc_v2/attempt_002` repeated those exact three scenarios
under bounded worker GC discipline: 3,000 safe steps, zero calls over 10 ms,
maximum 3.832 ms, no control-time collections, and exact physical command-word
and non-timing trace equivalence. Its raw wrapper false flags remain preserved:
the wrapper compared JSON lists with in-memory tuples. A separate read-only
serialized-equivalence audit resolves that reporting defect; no physical rerun
or criterion change was used. The derived verdict is recorded in
`adversarial_gc_v2/attempt_002/derived_summary_v1.json`. The original timing miss
and the new attempt's zero-step preflight predecessor remain retained.

The learned-command bridge retained the original Gaussian/noisy-depth policy
observations and added an independent bounded SIL monitor. It completed 3,000
conditionally certified physical steps and 1,800 neural frames with no enclosure,
collision, unknown-entry or queue misses. Low and random-interior cases reached
the first waypoint, but **0/3 complete two-waypoint missions finished within the
10-second reservation**. All 3,000 issued commands counted as interventions;
learned requests influenced anchor intent rather than being issued directly.
Subsequent development added proved public trim/energy renewal. A streaming run
renewed three times and completed 2,287 safe steps, then exhausted its reservation
while still .419 m from the current waypoint; it also retained 469 control calls
over 10 ms. A later coverage/hold experiment also failed completion and timing.
These are development failures, not held-out validation passes.

The next positive integration protocol uses an explicitly preflight-observed
static local map. All free labels must still come from bounded first-hit ray
witnesses, and occluded or unresolved cells remain unknown. Crossing the earlier
deliberately unscanned slab required active discovery, which belongs to Phase 5;
those unknown-space failures remain retained. No new positive cohort has passed.
The new prior-map development mission then completed both waypoints in 18.48 s:
1,848 certified physical steps, four successful renewals, zero safety/unknown/
queue failures and zero all-in shield calls over 10 ms (maximum 6.712 ms).
Peak speed was .08837 m/s, minimum physical clearance lower bound .260 m, and all
1,848 issued words counted as interventions. The original Gaussian policy path
remained unchanged. Source hashes were stable; sensing, NN inference, simulated
plant and independent oracle timings are recorded separately from shield work.
This is development evidence, archived as
`learned_prior_map_v4/development_prior_low4500.json`, not a fresh validation pass.
Frozen prospective validation and portable integrated verification remain open.

Therefore `shield_gate.passed`, `nonhardware_ready` and `phase_passed` remain
false. A component pass does not upgrade the full software status. Actual onboard
timing remains deferred; physical deployment is not claimed.

Selected summaries, all under `reports/phase_4_gate_artifacts/shield_research/`:

| Evidence | Relative artifact | SHA-256 |
| --- | --- | --- |
| Component | `coupled_reanchoring_v1/retry_003/summary.json` | `f67f1daa464edaa3a4522e2b402c6ee6cc4df0f0cf8ab4b9c10aed4530535b7d` |
| Adversarial | `adversarial_reanchoring_v1/summary.json` | `da6b4039e747e34c4b60b7b68e78818a11f06631916a6084f58fda58ebd2eccd` |
| Learned bridge | `learned_command_bridge_v1/development_summary.json` | `dc1c0327be2238d3c1ac7dae989b27090a8d67423d54b3cdfd61e6ac8c237675` |

## Artifacts and verification

The machine-readable index is `phase_4_gate.json`. The pre-test freeze record is
`phase_4_gate_artifacts/test_freeze.json`; full per-task results are in
`phase_4_gate_bptt_test.json`, `phase_4_gate_ppo_test.json` and
`phase_4_gate_reference_test.json`. Training curves and immutable checkpoints are
under `phase_4_regularized_validation_artifacts/`.

The frozen 51-source runtime wheel is archived under
`phase_4_gate_artifacts/frozen_flight_runtime/`, SHA-256
`9e7850db055ab1cbea454a45e1cd0662571b09eb3d5cccf52d436b8557e6042e`.
That frozen 51-source flight runtime passed 344 tests. Outside-checkout CPU
training/reload/evaluation with CuPy blocked verifies the selected residual,
penalty and guidance settings; details are in `phase_4_gate_packaging.json`.

The earlier 52-source tree added `safety_contract.py` while preserving those
51 sources and passed the full 351-test suite. That wheel's evidence remains in
`phase_4_safety_contract_packaging.json`. The new portable `ares.verified_safety`
component now has seven passing focused tests. The current 55-source package
passed all 358 tests in 79.164 s, with all source hashes unchanged during testing;
its log and source manifest are in
`phase_4_verified_safety_artifacts/full_suite_revision02.*`. The preceding
357-test revision and its log remain retained separately.

The portable CLI and its scope are documented in
[verified safety SIL](../docs/verified_safety_sil.md). Its wheel passed 20 CPU
steps outside the checkout with CuPy imports blocked. Packaging evidence is
`reports/phase_4_verified_safety_artifacts/outside_checkout_packaging.json`
(SHA-256 `f84bc0c388f7ecdb829e693f377af21869a2945a5fe0d9b72621f75eda6c981a`);
wheel SHA-256 is
`e90870979baacf781d4955c47595cb6ef3e032ec153ae9da25d544642eee76d1`.
Hardware timing is not inferred from desktop or packaging checks. Previous
reports, frozen wheels, checkpoints and sealed cohorts are retained.
