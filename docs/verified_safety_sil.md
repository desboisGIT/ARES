# Verified backup software in the loop

The conditional CPU backup is available as a portable, checksum-verified tool:

```sh
python -m pip install '.[flight]'
python -m ares.verified_safety smoke --output /tmp/ares-backup-smoke.json
python -m ares.verified_safety validate --output /tmp/ares-backup-validation.json
```

Use fresh output paths. The tool preserves existing results and logs. `smoke`
defaults to one case and ten steps; `validate` fixes eight cases and 1,000 steps
per case. A C compiler and headers are required for the worker's arithmetic-mode
probe. The tested backend is eager Float32 on x86-64 CPU, with one Torch thread.

`smoke` checks functional integration and reports timing separately, so a shared
CI host is not mistaken for a real-time deployment test. `validate` also requires
every measured guard call to meet the 10 ms budget. Both modes retain timing
failures in their reports; `conditional_SIL_passed` is false whenever timing
fails, even when the functional smoke exits successfully.

The installed package includes the exact source/evidence closure. The launcher
checks its archive and every member, extracts an isolated temporary workspace,
binds the unchanged installed ARES physics, and runs a separate process. Results
and logs survive outside that workspace. CUDA and compiled kernels are outside
this numerical contract.

## What this tool establishes

The initial set comes from bounded public position, velocity and rate packets,
issued-command history and a public trim calculation. The guard receives no
sampled mass, gravity, motor state or actual battery energy. Simulated sensors
and the independent test oracle can access physical state; the guard cannot.

The backup uses the full 13-state rigid-body model, motor lag, rate loop, battery
depletion and delayed command queue. Exact and outward arithmetic verifies its
nonlinear invariant set across the full physical parameter box. Known-free
voxels require bounded first-hit ray witnesses. Every accepted anchor reserves
the whole remaining backup volume, including the body and a 5 mm per-axis
intermediate-motion allowance. Rejected proposals retain the previous backup.

The declared service supplies fresh deterministic position/velocity/rate bounds
and exact current raw quaternion feedback, matching the current Phase 4 attitude
abstraction. Its lease and static-map witnesses must cover the complete backup
deadline. Gaussian, RMS or confidence bounds cannot replace these obligations.
The finite reservation is ten seconds; it is not an infinite battery guarantee.

## Current evidence and limits

The frozen eight-case, 8,000-step component run passed all membership, collision,
unknown-space and queue checks. It accepted 6,181 anchor changes and used the
cached backup 1,819 times. Waypoint-distance progress was 0.191–0.297 m. Desktop
guard time averaged 2.304 ms and peaked at 4.271 ms, measured over decision and
command acknowledgement. Sensor rendering, plant simulation and the independent
oracle are excluded from that timing. These are finite measurements on pinned,
non-exclusive desktop cores, not a universal execution-time bound.

Three additional boundary scenarios each rejected 139 proposals whose backup
reserve touched unknown or occupied space, while completing their physical runs
without collision, unknown entry or enclosure misses. Their timing evidence
retains one 12.128 ms corner-case outlier. The failed earlier timing experiments
are also retained; they are not substituted for the selected run.

This is a conditional backup component. The original randomized Gaussian flight
benchmark remains unchanged and uncertified by this input contract. Connection
from the frozen learned thrust/body-rate commands to certified cluttered missions
is being validated separately. Passing this command alone does not mark all
Phase 4 software requirements complete. Real sensor guarantees, onboard timing
and physical deployment are not inferred from the SIL fixture.

The source report is [the current gate report](../reports/phase_4_gate.md).
