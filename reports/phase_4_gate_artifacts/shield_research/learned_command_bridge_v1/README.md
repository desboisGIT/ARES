# Frozen learned command / verified shield bridge

This is a **prospective development integration**, not a Phase4 acceptance report.
It consumes the frozen BPTT checkpoint's actual thrust/body-rate proposals, maps
them approximately into fixed-heading backup-anchor intent, and issues only the
existing shield's exact admitted backup command. Every binary32 physical word
change is counted as an intervention, including incompatible learned yaw.

`adapter.py` uses public bounded p/v/q for inversion and untrusted floating-point
candidate screening. It cannot authorize motion. `ReanchoringShield.decide`
rechecks the current augmented uncertainty set and the complete future reserve
exactly; otherwise the previous fixed-anchor backup remains selected.

The network receives `FlightEnv._observe` and `FlightSensors` outputs with the
original full randomized Gaussian/state and depth-noise models. Its 24×32 depth,
43 ordinary features, next two waypoints, gaze, measured map ring, previous
issued command and adaptive-controller memory retain the original interface.
No bounded-monitor value, physical mass/motor state, or occupancy label replaces
a network feature. Physical state is visible to the sensor boundary and separate
oracle, as in the original simulator. Network weights and the old runtime51
sources remain unchanged.

The rational scheduler has 60 Hz policy/sensor events and 100 Hz physical/shield
events. A policy event uses the most recent completed 100 Hz physical endpoint,
never the future endpoint that has been computed for the next macro boundary.
The recorded capture age is below10ms. Controller memory uses the latest Gaussian
observation and actually issued command at100Hz. These scheduling changes are
explicit multirate adaptation, not a claim of bit-identical old60Hz rollout.

The new furniture fixture has the same exact physical geometry on the original
.2m sensor lattice and the shield's .25m lattice. Its fresh ray witnesses come
from an independently enforced bounded SIL supplier with a30s static-world
lease. The eight existing public trim histories may initialize physically
identical parameter cases; no sampled hidden value enters guard admission.
These first development cases use the same fixture geometry and deliberately
nontrivial two-leg missions (.634m and1.086m), with the original .1m first/.35m
second waypoint conditions. Seeds90000+ are prospective; no sealed80000 NN
cohort is inspected or reused.

The initial guard remains capped at10s. Reaching the first waypoint is not a
successful two-waypoint mission. Longer missions need explicit proved public
trim refresh, sufficient remaining energy and independently valid future
service/map leases. Extending a timestamp without those obligations is invalid.

Run from the repository root:

```sh
.venv/bin/python reports/phase_4_gate_artifacts/shield_research/learned_command_bridge_v1/test_adapter.py
.venv/bin/python reports/phase_4_gate_artifacts/shield_research/learned_command_bridge_v1/run_integrated.py --case 0 --steps 1000 --task-seed 90000 --output development_low_90000.json
```

Reports retain every attempted step, requested/issued commands, proposal/anchor,
actual interventions, Gaussian observation summaries, policy capture timestamps,
C/queue/collision/unknown checks, timing and source hashes. Clearance uses the
original swept-sphere evaluator; capped local-distance values are lower bounds,
not precise distances to a distant obstacle. The first low-profile driver is
archived in `driver_snapshot_low_90000/`; later reports embed their driver sources.
