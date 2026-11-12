# Phase 0 report

Date: 12 November 2026. Scope: Phase 0 foundations, explicitly confirmed by the
user. The supplied build plan is retained as reference in `docs/build_plan_v3.md`.
The later simulation, training, hardware, and flight phases have not been executed.

## Gate outcome

The Phase 0 software gate passes. The default command is `make build`.

| Gate | Evidence |
| --- | --- |
| One-command build from clean source | A source-only temporary checkout, with no preexisting build/dist/report artifacts, passed `make build` using a fresh virtual environment without setuptools. Build isolation installs its backend. The final source also passes the local build. |
| Containers and CI | Docker image builds, runs checks during build, and completes the default mission as nonroot. GitHub Actions is configured for Python 3.11/3.12. No remote CI run is claimed. |
| Hand-built three-room scene validates | Packaged 6.4 × 2.4 × 2.6 m grid; 20 cm voxels, three enclosed rooms, two 0.8 × 2.0 m open doors, exact exposed-voxel collision mesh. All three rooms are body-reachable. |
| Broken scenes rejected for the right reason | Gate compares exact expected diagnostics for out-of-bounds occupancy, an occupied spawn, and a missing referenced room. Unit tests additionally cover mesh indices, polygons, doors, materials, lighting, nonfinite numbers, duplicate JSON keys, and physical-profile ranges. |
| Attack policies fail | Instant reporting, endless hover, leaving the map, and reporting an omniscient forged map all fail. Each has an independent tested failure reason. |
| Same seed gives identical results | Gate runs seed 42 twice and compares complete report/result/evaluator-log content. Separate processes with Python hash seeds 1 and 999 also produced byte-identical artifacts. |

The full final unit suite has **48 passing tests**, including six independent
adversarial scorer tests and regressions for continuous collision checking,
unsupported map claims, arbitrary mapper room IDs, invalid polygons, evidence
bound to another scene/profile, and mutable-input isolation. Dense training reward
is tested separately and has no authority over terminal pass/fail.

The wheel includes all packaged scene/profile/benchmark assets. Installation into
a temporary directory and execution outside the repository passed validation and
the Phase 0 gate, with the same deterministic mission artifact digest as source.
Local host Python: 3.14.4; container Python: 3.12. Python 3.11 is configured in CI
but has not been separately executed here.

## Measured witness

This is an evaluator-only scripted traversal of known space, not a classical
exploration policy or a real flight result.

| Metric | Result |
| --- | ---: |
| Body-reachable camera cells | 1,680 |
| Fixed observable target voxels | 3,972 |
| Coverage | 100% |
| Mean free/occupied floor-plan IoU | 1.00 |
| Rooms found / required | 3 / 3 |
| Collision segments | 0 |
| Minimum surface clearance | 0.15 m |
| Virtual landing | Pass |
| Harness elapsed time | 1,344.2 s |

The witness enumerates reachable voxel centers at a scripted 0.5 m/s, then dwells
at the virtual home station. Its elapsed time is only a reproducible fixture;
it cannot establish mapping speed or physical safety.

Deterministic mission/result/evaluator-log digest:
`f954b4fe376c6c22f7723e17e9738650147353639c578c5711d3769e8f1fa093`.
Scene/profile identity:
`b84a2aee5de9056feb231f02f0bf8c869f1f3e8c12224fe0b2ace2862b82843a`.

## Evidence and interpretation

Regenerate machine-readable evidence with:

```sh
make build
```

Outputs are `reports/phase_0.json`, `reports/benchmark.json`,
`reports/phase_0_artifacts/{mission_report,witness.runtime,witness.evaluator}.json`,
and `dist/ares-0.1.0-py3-none-any.whl`. The gate JSON stores full input snapshots,
versions, scorer rules, seed, invalid-scene diagnostics, attack results, and hash.
Generated JSON/logs and build products are ignored by git; this report is retained.

The CPU axial-scan benchmark compares 11 paired repeats of 1,200 scans against a
frozen equivalent reference. The final local check passes the 10% normalized
slowdown limit; exact timings and ratio are in `reports/benchmark.json`. Docker
also passes this check. This is an initial CPU-primitive baseline, not a GPU or
full-stack throughput measurement.

The plan's ambiguous coverage, IoU, room-discovery, and landing meanings are
resolved explicitly in `docs/scoring.md`: a trajectory-independent observable
voxel target, separate free/occupied class IoUs, evidence-backed one-to-one room
polygon matching, and a virtual home-station dwell. The Phase 0 sensor is an
ideal six-axis voxel scan with no physical scan cost. Camera FOV, yaw, depth
noise/dropout, dynamics, actual touchdown, and motor control remain later work.

The evaluator owns trusted scene/telemetry data; policy reports cannot assert
their own success. This in-process boundary is not a sandbox for hostile Python
code. Runtime log readers reject evaluator ground truth by default. Raw camera,
depth, IMU, command, shield, and map-snapshot log fields are honestly null in this
phase. Arbitrary scene validation checks mesh structure, while exact mesh/voxel
equivalence is separately proven for the packaged fixture.

`LICENSES.yaml` records the zero-third-party-package runtime, build dependencies,
container audit gaps, and verified research-reference licenses. ARES source and
generated-asset licenses await owner selection (`NOASSERTION` for the fixture).
Unverified entries are marked `commercial_ok: false`. External research code has
not been imported. FUEL's GPL license is incompatible with the plan's chosen
permissive deployment policy; GPL itself permits commercial use.

## Delegation

Low-effort agents handled build tooling, reference lookup, log serialization,
straightforward tests, and packaging/determinism checks. A medium-effort agent
handled strict contracts and an independent adversarial review. The main agent
retained scorer design, evidence ownership, integration, defect fixes, and the
overall Phase 0 gate. All delegated changes were integrated and checked together.
