# Prepared bounded-GC adversarial wrapper v2

Status: **prepared, not run**. No timing flights have started. The default command
only writes `prepared.json`; timing requires explicit `--run` after the separate
root/assessor timing-window clearance.

The three scenarios, low-profile case0 restore,1,000-step horizons, noise,
model/proof, anchor intent and strict10ms criteria are exactly those in the
byte-preserved `adversarial_reanchoring_v1/run_scenario.py`. The only control-worker
policy addition is the frozen `ControlWorkerGC` context with maximum1,000 steps.
The sealed loop reserves its declared bound and reports successful completed
counts afterward; it is not mislabeled as per-tick instrumentation.

Each scenario runs in a fresh child pinned to one of physical cores4,5,6. Parent
GC-enabled state/thresholds are checked unchanged. Calls retain original
per-tick decide/ack/total wall times and GC deltas, with preflight collection cost,
context stats/counts and current/lifetime peak RSS. OS preemption is not removed
from timers. Remaining misses remain failures; no replacement timing samples are
selected.

## Output isolation and provenance

The original runner always writes its ray map relative to its module `HERE`.
With root authorization, this wrapper temporarily rebinds **only** that global in
each child to a NEW output directory, restoring it in `finally`. `ROOT`,
`RESEARCH`, imports, function bytecode, actual inputs and model state are unchanged.
No source file is edited.

The inner runner's source inventory changes location with that output binding,
so its `source_hashes_stable` flag is explicitly insufficient by itself. The
outer wrapper hashes the original scenario/shared sources before and after the
rebinding, binds the GC helper and new wrapper, verifies every original protocol
hash against its previous scenario report before starting, and independently
hashes all old source/evidence files before and after. It compares every original
trace field except `guard_timing` with the new trace, retaining all differences.
Every old summary, ray map, scenario report, log and failed timing observation
remains untouched.

```bash
# Prepare only; launches no physics.
.venv/bin/python reports/phase_4_gate_artifacts/shield_research/adversarial_gc_v2/run_wrapper.py

# ONLY after an explicitly coordinated timing window:
.venv/bin/python reports/phase_4_gate_artifacts/shield_research/adversarial_gc_v2/run_wrapper.py --run --run-id attempt_001
```

Each new run identifier must name a nonexistent directory. Functional failures
stop other workers and retain their outcomes; timing failures remain reported.
No new scenario, proof, motion criterion, phase-pass or hardware/WCET claim is
introduced. `flight_certificate` remains false.
