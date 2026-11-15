# Phase 4 completion plan

The learned waypoint gate has passed; the conditional shield gate remains open.
Actual onboard timing is deferred until hardware is available. This is a work
plan, not a whole-phase completion claim. Requirements come from Phase 4 and
section 4 of [the build plan](docs/build_plan_v3.md). The current
[gate report](../reports/phase_4_gate.md) and
[machine-readable index](../reports/phase_4_gate.json) preserve the evidence.

The frozen 80000–80015 cohort is consumed, alongside 60000 and 70000, and must
not be used for tuning. All 51 benchmark flight sources remain frozen. The
current 55-source tree adds the separate conditional input module
`safety_contract` and portable toolkit. All 358 tests pass in 79.164 s, including
seven archive/CLI tests, with unchanged source hashes during testing. Earlier
344-, 351- and 357-test evidence remains tied to its own archived revision.

| Requirement | Current evidence | Acceptance gap | Dependency |
| --- | --- | --- | --- |
| BPTT through full DiffAero dynamics with the declared low-resolution sensor/mission interface and episode randomization | Selected BPTT trained for 409,600 transitions, with finite nonzero depth-encoder gradients; 500+ Hz rate loop, selective resets and frozen profile/source snapshots | Preserve provenance and declared interface; any later learning changes need validation and a fresh test cohort | Immutable checkpoint, 51-source manifest and archived wheel |
| At least 99% learned waypoint-pair success | BPTT 1000/1000, zero collisions/timeouts/speed violations on sealed 80000 tasks; second training seed passed 256/256 validation | Measured gate passed; finite endpoint/interior coverage is not a universal continuous-range proof | Retain all tasks, failures, parameter strata and frozen evaluation protocol |
| Report speed, clearance and shield interventions | BPTT peak 1.45583 m/s, minimum swept clearance .01545 m; reference 1000/1000, minimum .01413 m | These unshielded clearances are not a certified uncertainty/tracking reserve; do not claim learned superiority over the reference | Enforceable state/map and backup-tube bounds |
| PPO with the same interface for comparison | PPO 994/1000, six collisions and five speed violations; same architecture, initialization seed, transitions and test tasks; optimizer/compute budgets reported separately | Comparison delivered; no separate 99% PPO gate is stated in the build plan | Preserve full comparison evidence and failures |
| Shield prevents every collision preventable under its assumptions; no entry into unknown space | Conditional nonlinear CPU backup mathematics is closed; bounded-service component passed 8,000 steps with 8,000 positive decisions. Adversarial GC retest passed 3,000 functional/timing steps; old 12.128 ms timing miss retained | New prior-map development completed both waypoints in 18.48 s with four renewals and zero timing/safety misses; frozen prospective validation and portable integrated verification remain open. Default Gaussian inputs alone remain uncertified | Proved public renewal, additional observed-map witnesses, operational timing and real supplier guarantees |
| Under 5 ms inference on the actual onboard computer | Desktop and packaging evidence only | Deferred actual-device inference and end-to-end timing; not discharged by desktop results | Onboard device and deployed timestamped stack |

## Remaining sequence

1. **Preserve the passed learning gate.** Keep checkpoint hashes, all per-task
   outcomes, source manifests and the 51-source wheel sealed. BPTT and PPO each
   used 50 × 128 × 64 transitions with residual authority .5 m/s², penalty 1 and
   guidance fraction .95; the mission speed limit stayed 1.5 m/s. Future learning
   experiments use validation only, followed by another predeclared fresh test
   cohort if a replacement candidate is selected. The current gate needs no
   additional tuning against its test set.
2. **Complete learned-command missions under the conditional contract.** The
   bridge completed 3,000 certified physical steps and 1,800 neural frames using
   original Gaussian policy observations plus an independent bounded monitor.
   Low/interior cases reached the first waypoint, but none completed both within
   10 seconds; every issued word was an intervention. Public trim/energy renewal
   now works in development, including three renewals during a longer learned
   mission, but completion and timing still failed. The next positive protocol
   explicitly supplies a genuinely preflight-observed static local map through
   bounded first-hit rays; unresolved cells remain unknown. The all-direction
   external prior-map service is separate from the unchanged forward-facing NN
   depth interface. Active discovery is Phase 5. Preserve earlier unknown-space
   challenges, freeze the new protocol and validate complete cluttered missions
   with requested/issued commands and every failure retained.
3. **Retain the closed backup proof and its domain.** The exact/outward nonlinear
   CPU backup argument is archived in `cpu_rpi_certificate_v1`; the selected
   bounded-service component completed 8 × 1,000 steps with zero membership,
   collision, unknown-entry or queue misses. Its 6,181 anchor accepts and 1,819
   fallbacks demonstrate component operation, not full mission completion. Keep
   the deterministic supplier, exact-current-quaternion abstraction, guarded
   numerical backend and finite horizon enforced. Default Gaussian inputs do not
   satisfy that contract; physical supplier guarantees remain separate.
4. **Close timing and portable integration evidence.** Selected component guard
   timing was 2.304 ms mean / 4.271 ms maximum, but adversarial tests retain a
   12.128 ms corner miss despite functional success and 139 reserve refusals per
   scenario. Preserve that failure and address scheduling/latency before claiming
   operational 100 Hz coverage. A bounded cyclic-GC worker diagnostic has since
   completed 1,000 Docker steps with no collections or timing misses. The exact
   three-scenario adversarial set also completed 3,000 steps with zero timing
   misses, maximum 3.832 ms, and unchanged physical commands. Its separate
   serialized-equivalence audit preserves and explains the wrapper's list/tuple
   comparator false flags. The portable
   component CLI now passes 20 CPU
   steps outside the checkout with CuPy blocked; use its documented commands for
   reproducible smoke and validation. Include synchronous map/proof updates in
   control timing and observation helpers in the source manifest, then verify the
   integrated package outside the checkout and run its final full suite.
5. **Measure onboard timing when hardware is available.** Measure the frozen
   deployed policy under realistic load, including latency tails, against the
   under-5-ms requirement. Timestamp sensing, estimation/map updates, policy,
   100 Hz shield, command transport and the 500+ Hz flight controller. Feed
   measured end-to-end bounds back into the shield and repeat coupled validation
   when those assumptions change.

Finite low/high/interior trials support the reported measured success rate;
they do not prove every joint parameter combination. The earlier reference's
.000271 m minimum clearance was insufficient evidence of a robust reserve, and
the current larger measured minima also need an uncertainty/backup argument.
Whole-phase completion remains blocked by the open shield and deferred actual
onboard timing obligations, not by an unpassed learned waypoint gate.

Current artifact paths and SHA-256 values are indexed in the
[gate report](../reports/phase_4_gate.md) and its JSON. Portable commands and
conditional scope are in [verified safety SIL](verified_safety_sil.md); packaging
evidence is `reports/phase_4_verified_safety_artifacts/outside_checkout_packaging.json`.
The full software status remains false while learned-command mission integration
is open; conditional component success is not whole-phase completion.
