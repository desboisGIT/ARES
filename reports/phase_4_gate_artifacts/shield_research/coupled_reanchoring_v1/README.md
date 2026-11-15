# Coupled public-only reanchoring SIL v1

This research runner couples the SHA-bound public-history initial admission,
`ReanchoringShield`, the bounded-service witness map and the guarded explicit-
norm eager-CPU full rigid-body backend. No ARES sources, sealed checkpoints,
held-out cohorts or previous experimental reports are changed.

The authoritative completed experiment is `retry_002/full_sweep.json`: eight
independent cases, each completing 1,000 .01 s macrosteps. Every case, command,
proposal, fallback and independent oracle result is retained. Earlier timing-
failed smokes and the stopped first sweep remain development history.

## Result and timing limitation

All 8,000 simulated macrosteps received positive **conditional** guard decisions.
6,181 new anchors were accepted; 1,819 rejected proposals used the cached backup.
There were no actual-C membership misses, queue mismatches, physical collisions
or unknown-map entries. Every case moved positively along x/y/z and reduced its
distance to the initial public anchor + (.3,.2,.1) m. Goal-distance progress ranged
from .19090 to .29690 m, maximum speed was .03724 m/s, and minimum physical body
clearance was 1.55517 m.

The strict 100 Hz timing gate **failed**: one of 8,000 measured decide+acknowledge
calls took 10.542 ms; the remaining 7,999 were at most 10 ms. Per-case mean guard
time ranged from 6.154 to 6.618 ms. Individual duration samples were not retained,
so the precise tick of that one miss is unavailable. No replacement run was used
to discard the miss. The report marks `all_timing_gates_passed=false` and
`all_cases_passed=false`. Positive conditional decision flags are not evidence
that live wall-clock scheduling assumptions held for every command.

Guard timing excludes the supplier, physical plant, energy update and independent
oracle. Workers were pinned to CPUs0–7, one logical CPU per physical core;
SMT siblings8–15 were not used by these workers. Affinity is not exclusive core
reservation. This is software SIL evidence, not a hardware or Gaussian-runtime
certificate. No overall phase-pass flag is set.

## Reproduction

From the repository root, an endpoint smoke can be run with:

```bash
OMP_NUM_THREADS=1 taskset -c 7 .venv/bin/python reports/phase_4_gate_artifacts/shield_research/coupled_reanchoring_v1/run_coupled.py --start-case 1 --cases 1 --steps 100 --output my_new_smoke.json
```

The full sweep uses eight separate single-thread processes and requires eight
distinct physical cores. Give each attempt a new identifier; existing attempt
directories are rejected:

```bash
OMP_NUM_THREADS=1 .venv/bin/python reports/phase_4_gate_artifacts/shield_research/coupled_reanchoring_v1/run_sweep.py --run-id my_new_attempt
```

Any unexpected actual-C/queue/collision/unknown failure triggers a cross-worker
stop. Other workers finish their current transaction and retain partial records;
they are not relabeled invalid initial conditions. Timing misses remain failures
but do not erase other cases. Before/after hashes bind executing helpers, proof
manifest/certificate, public history, restore input, bounded map, fixture and
ARES sources. Every completed retry worker had the same stable hash map.

## Public and private boundaries

The guard receives only recomputed `InitialAdmission` public-history rows,
measured p/v/q/w packets, the public hold/trim/motor-residual families, two issued
command words, the witness-derived map and explicit backend/service attestations.
The public energy upper bound is 4,338 J and the guaranteed fresh-service and
backup horizons are 1,000 ticks. Packet q has the explicit exact raw-orientation
service tag. The initial packet exactly continues its admitted public history
at rebased mission tick zero; subsequent packets are freshly captured per tick.

`PlantAndOracle` alone restores `simulation_truth_for_replay_only` state13D,
motor, last two command words and energy words. Sampled profile/mass/gravity,
actual motor, battery capacity and actual energy never enter the guard. Physical
stepping uses verified motor decay, explicit verified norm/sqrt, guarded CPU
arithmetic and the verified battery authority/energy primitives. The guard
acknowledges the actual transmitted Float32 command words.

The constructor's high-delay binary64 boundary is bound downward only when it
exceeds exact1/50; low .005 stays unchanged. Exact `struct.pack('>f',...)`
equality proves this changes no physical Float32 delay or normalized queue frame.
Raw, bound and stored delay values are reported per case. No backend/proof-domain
inequality is relaxed.

The independent oracle checks each complete actual chord AABB plus physical
body radius against both analytic physical free voxels and the public known-free
map. It checks actual-C membership at every endpoint with zero measurement
errors, projective point quaternion, exact product of actual mass/gravity words,
end-energy battery authority, actual motor residual and both actual issued queues.
An actual-C miss fails the case. The oracle is excluded from guard latency and
cannot change the guard's proposal, command or reserve.
