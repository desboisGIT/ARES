# Phase 3: fast world and exploration baselines

**The GPU throughput and classical apartment gates pass. Actual human timings
remain pending, so `phase_passed` is false.** Manual flight, logging, replay and
human-result verification are ready. Phase 2's hardware gates remain pending.

## Held-out results

| Actor | Runs | Median mission time | P90 mission time | Pass rate | Collision segments |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen classical explorer | 100 | 287.80 s | 415.54 s | 100 / 100 (100%) | 0 |
| Human | 0 | Pending | Pending | Pending | Pending |

Times describe passing missions in simulated seconds; failures always remain in
the pass-rate denominator. The 95% Wilson interval for the observed classical
pass rate is 96.30–100%. This is a result on the generated apartment family,
not a general real-building success claim. All 389 required room instances
were matched, all runs landed safely, minimum coverage was 99.6662%, and minimum
floor-plan IoU was 0.81381. Evaluation took 331.84 s wall time with four workers.

Development used only train seeds 100–109. The policy was frozen at SHA-256
`8ddea82cd2971492daee617959b0894280cbcc37e86f29331eda47fdcdffbfca`
before the single held-out evaluation on test seeds 50000–50099. No policy
tuning followed these results. Generator `ares.worldgen.v1` partitions occupied
geometry across splits. The measured dataset digest is
`faf551fdb9813af80d6efb4aac010b51873f15ad035cb125cb2630c507bf8c48`.
`reports/phase_3.json` preserves scene hashes, profile/config snapshots,
individual results, policy settings and separate gate statuses.

## GPU measurement

The RTX 5070 Ti measured **3,004,412 individual environment steps/s** with
4,096 active environments, exceeding the 100,000 requirement. The timed 25
batches contain 102,400 actual steps over 0.03408 s, with no terminal/frozen
environments. Median synchronized batch duration was 1.386 ms; p90 was 1.410 ms.
Five longer repetitions of 250 batches each measured 3.130–3.165 million steps/s,
again with every environment active. These repetitions are retained separately
in `reports/phase_3_artifacts/throughput_repetitions.json`.

This measures one small training apartment shared by independent environments,
each with a 32×24 cone. It includes capped point-mass/yaw dynamics, continuous
sphere/voxel collision, DDA rays and atomic observed-map updates. Warmup,
initialization, policy planning, scoring, frame logging and observation downloads
are excluded. Python 3.14.4, NumPy 2.5.4 and CuPy 14.2.0 were used. Larger scenes,
native camera sizes and full policy-stack throughput need separate measurements.

## Implementation and verification

The observed-map classical explorer uses frontier-adjacent viewpoints, ray-based
expected gain, hull-aware known-free routing and a bounded return/landing policy.
It never reads scene metadata or true coverage. Room inference has an explicit
rectangular-generator prior at the above-door layer, documented in
`docs/classical_explorer.md`. The evaluator retains a policy-independent target
and rejects unsupported map claims. The model and scoring boundary are described
in `docs/fast_world.md`.

The combined local suite passed **214 tests**, including actual CUDA parity,
continuous grazing collisions, occlusion, reset/isolation, immutable evidence,
scorer identity binding, spawn-safe workers, human-log verification and manual
clock/watchdog behavior. Browser QA under actor `ui_test` verified yaw, braking,
landing, reset and replay seeking to a scored final frame. It found a lost
one-shot Land request; the session now latches it until a physics tick consumes
it, with stale/early-request regressions. Keyboard fallback was exercised;
physical gamepad input remains untested because no controller was connected.

A final audit corrected launch-hull evidence for imported scenes with off-center
spawns. Exact sphere/voxel intersection now seeds only the actual physical hull.
CPU/CUDA regressions cover hidden cells behind the cone and diagonal hull cells.
Old/new launch evidence was identical for all 110 development/evaluation scenes,
so this fix preserves the recorded benchmark results and frozen policy.

`make build` passed all 214 local tests, the earlier phase smoke checks, the
paired 10% performance guard and wheel packaging. The Docker build passed its
CPU suite (211 tests, three skipped GPU tests) and phase checks. An installation
of the 0.4.0 wheel outside the checkout, with CuPy imports prohibited, passed
the CPU exploration smoke, loaded the packaged viewer and replayed its log to a
passing final score. Standard-library-only foundation imports still work.
The nonroot container's default CPU smoke also passed, retaining false GPU,
held-out and full-phase gates. Build/package checks and final source hashes are
retained in `reports/phase_3_artifacts/verification.json`.

`reports/phase_3_replay_smoke_artifacts/logs/classical-train-100.json.gz` is a
completed CPU example with depth, observed map, actions and score. New logs also
record policy configuration and version metadata. Replay accepts exploration
JSON/gzip, Phase 0 flight logs, Phase 2 estimated trajectories and normalized
sensor directories. Pose-only logs display their trajectory without invented
occupancy; raw depth explicitly has no estimated pose. Runtime readers do not
open evaluator ground-truth files.

## Recording the human row

Practice on train scenes using the commands in `docs/manual_flight.md`. To
collect a run matching this benchmark, use a test seed in 50000–50099:

```sh
.venv/bin/python -m ares fly --seed 50000 --split test --backend cuda --actor human --log-dir logs/manual
```

Open the printed URL, click Start controls, explore, return to the launch marker,
brake for one second and click Land. The independent wall-clock scheduler keeps
time running while thinking or braking. Reset and close retain partial audit
logs. A completed collision or timeout is a legitimate failed human attempt;
partial runs cannot qualify.

After actual human runs, aggregate them without rerunning the classical cohort:

```sh
.venv/bin/python -m ares human-benchmark logs/manual/*.json.gz --baseline reports/phase_3.json --output reports/phase_3_human.json
```

The command recomputes metrics by CPU action replay, checks cohort/profile/world
configuration and real-time duration, and rejects duplicates, UI tests, partials
and incompatible records. Human identity and wall duration are unsigned operator
attestations. The benchmark reports this limit explicitly. Rejected attempts
have reasons; completed valid failures stay in the denominator. The original
classical result is preserved by the default output path. Human timing gate
completion requires actual eligible records; no sample count beyond that is
specified by the plan, so the recorded count must accompany any comparison.

All three browser-QA audit logs were rejected by the real human aggregation
command. `reports/phase_3_ui_rejection.json` retains zero human runs and null
human times. Automated runs have not been substituted for human measurements.

Reproduce the software checks with `make build`, and the full GPU/classical run
with `make phase3`. CPU smoke checks exercise integration without certifying the
GPU, held-out or human gates. The model uses perfect pose, ideal voxel sensing
and point-mass motion; it does not establish real-flight behavior.
