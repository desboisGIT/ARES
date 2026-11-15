# Adversarial reanchoring SIL v1

These new tests import the frozen coupled runner's public packet, initial
admission and guarded physical-word pipeline. Shared guard, membership, explicit
norm/point kernels, ARES sources and retry_003 are not modified.

The physical scenarios are predeclared, each using admitted case0, its full low
profile/seed9100,1,000 .01 s macrosteps, fresh bounded measurements and identical
(.0003,.0002,.0001) m anchor increments:

- `unknown_plane`: forget public free cells with x index>=26 (x>=6.5 m). The
  original physical room and x=[9,9.25] slab remain unchanged.
- `occupied_plane`: replace the physical x9 slab with x=[6.5,6.75]. Rebuild the
  known-free map from new first-hit rays in that changed world.
- `occupied_corner`: use both x=[6.5,6.75] and y=[6.5,6.75] occupied slabs and
  rebuild all ray witnesses. This tests a corner world; the fixed anisotropic
  intent need not make both planes simultaneously limiting.

New scans use the same bounded supplier at the fixed preflight snapshot
(6.125,6.125,2.125). No old depths are reused for changed physical geometry and
no ground-truth occupancy is copied into the guard map. The independent oracle
checks every free label against physical geometry without altering the map.

The guard must retain a cached safe anchor when a candidate still fits C but its
complete future backup reservation touches unavailable geometry. Each scenario
requires at least one `proposal_reserve_not_known_free`, positive cached fallback
and genuine goal-distance progress. Actual C, both queues and the complete body-
inflated chord AABB are checked after every physical step; misses fail the case.
Timing is strict max10 ms for decide+acknowledge, excluding supplier, plant,
energy update and oracle. Every timing sample and GC delta is retained.

`run_all.py` runs three single-thread workers pinned to CPUs4,5,6, on separate
physical cores. Affinity is not an exclusive reservation. Functional failures
stop the other workers, retaining their completed/partial outcomes. Reports
remain conditional software SIL results with `flight_certificate=false`.

```bash
OMP_NUM_THREADS=1 .venv/bin/python reports/phase_4_gate_artifacts/shield_research/adversarial_reanchoring_v1/input_refusals.py
OMP_NUM_THREADS=1 .venv/bin/python reports/phase_4_gate_artifacts/shield_research/adversarial_reanchoring_v1/run_all.py
```

The refusal suite exercises stale/future/Gaussian/oversized measurements, invalid
orientation/history, missing reserve/lease, non-SPD shapes, transport mismatches
and lost arithmetic thread mode. Its outcomes are explicitly `preflight_refused`
with no applied commands and no executed-flight claim. The acknowledgement
mismatch test separately records that a correct word had previously received a
conditional plan certificate, but neither it nor the wrong word was applied.
A refused malformed input is not evidence of surviving permanent service loss.

All current shared inputs/helpers and new test scripts are hashed before/after
execution. No overall phase-pass or physical/Gaussian-runtime certificate is set.

## Recorded outcomes

All three scenarios completed1,000 steps with zero actual-C/queue/contact/unknown
misses. Each produced686 accepted new anchors and314 cached fallbacks. Of those
fallbacks,139 were specifically `proposal_reserve_not_known_free` despite the
candidate's complete public state box still fitting C. The first occurred at
tick773, proposed anchor (6.2082658,6.1352414,2.0486379); its largest C-channel
value was .979731. The same physical trajectory progressed (+.200656,+.133168,
+.042908)m, reducing goal distance by .241550m. Plane/corner physical clearance
remained at least .158603m. Independent map oracles verified all1352 plane free
cells and832 corner free cells; corner motion was first limited by its x plane.

The timing result is mixed and remains a **failure** of the combined strict gate:
unknown/occupied plane maxima were3.890/3.996ms, but the occupied corner had one
12.128ms decide+acknowledge call at tick82 (decide12.065ms,ack.0633ms). GC collection
deltas were zero for that call; its cause cannot be inferred from these records.
All other2999 calls fit10ms. No replacement run was performed to remove the miss.
`all_scenarios_passed=false` is retained. This is not live scheduling or a universal
worst-case execution-time proof.

All15 negative inputs were explicitly refused in preflight/isolated transaction
tests, with zero applied commands. Exact reasons are in `input_refusals.json`.
The lost-thread-mode test left physical state unchanged and latched the backend
blocked. All55 ARES Python sources and shared/new helpers remained hash-stable;
scenario workers had identical hash maps. `summary.json` and `all_scenarios.json`
retain the failures, scope and source-report provenance. No overall phase-pass
flag is set and no physical or Gaussian-runtime certificate is claimed.
