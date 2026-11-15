# Phase 1: world generator

Date: 12 November 2026. Base repository commit: `b357232`.

**Phase 1 passes.** The user approved all 50 displayed plans in chat on 12 November
2026: “alright i've check they all seems nice. we can move on to the next phase”. This user attestation is recorded against the exact gallery
and scene hashes; it is not an agent assessment or a claimed browser export.

## Gate evidence

| Gate | Result |
| --- | --- |
| 10,000 valid scenes, no individual hand fixes | **10,000 / 10,000**, zero failed cases. Includes 3,334 single rooms, 3,333 apartments, and 3,333 corridor-based floors. |
| At least one scene per second | **67.30 scenes/s**, 148.59 s wall time with eight worker processes on this host. Includes generation, schema/geometry validation, and hashing. |
| Human finds 50 random floor plans plausible | **Passed by user chat attestation**, bound to the 50-scene gallery below. |
| Recorded-map importer | Implemented and tested end to end on a labelled synthetic occupancy export. No real capture was supplied, and none is claimed. |

The corpus includes all five sensing materials, open and closed doors, lit and
dark rooms, furniture volumes, and multiple difficulty levels. Every case passes
the scene contract, closed-shell and spawn-clearance checks, and independent
mesh surface-area and signed-volume comparisons against its occupied voxels.
Material and lighting fields are preserved for Phase 2; they do not simulate
optical effects yet.

The corpus uses 8,000 training cases, 1,000 validation cases, and 1,000 test cases.
Occupancy hashes assign geometry to one partition, irrespective of changes to
lighting or materials. There are **zero cross-split occupancy duplicates**.
The human gallery samples only training cases. No policy training or tuning
occurs in this phase.

Corpus ID:
`ce5dd4b5c79f306e725790fa5070c438bdb68e7b6a6c45edefb778248159b871`

Gallery ID:
`239daaf2be41c2e4e398f8a6496320490c4c0bec5f2032b458169a5c0c536062`

The throughput figure excludes gallery construction and writing whole scene JSON
files. Exact environment, worker count, elapsed time, rates, failures, gate flags,
and importer provenance are stored in `reports/phase_1.json`. It is an initial
CPU world-preparation result; it makes no GPU or flight-performance claim.

## Implementation and validation

`ares.worldgen.v1` creates rectangular furnished rooms, partitioned apartments,
and 13–25-space floors with independent units around a common corridor. A
spanning tree of room adjacency provides architectural connectivity; door state
determines current physical reachability. Door aprons and internal routes are
reserved before furnishing. Separate units have their own corridor entrance and
cannot open directly into each other's private rooms.

`ares.geometry.voxel_mesh` merges exact exposed voxel surfaces into coplanar
rectangles, keeping large collision meshes compact. It rejects unrepresentable
coordinates. Analytic geometry tests cover surface area, outward winding,
internal-face removal, concavities, disconnected solids and translated origins.
The importer keeps unknown and missing occupancy cells blocked, validates
annotations, preserves the recording origin, and emits conversion provenance.
It accepts annotated occupancy exports, not raw camera/point-cloud streams.

The combined suite has **77 passing tests**. `make build` passes validation,
Phase 0's successful witness and four attack policies, the existing paired
benchmark (below its 10% slowdown limit), a 30-scene Phase 1 smoke check, and wheel
packaging. Tests also exercise cross-process determinism, geometry-based split
assignment, complete scene hashes, actual generation/import CLI paths, review
binding, and refusal to mark a small corpus or unreviewed gallery complete.

Docker builds and runs the checks using Python 3.12; the image runs as nonroot.
The Python 3.11/3.12 CI matrix remains, with a separate 10,000-scene CI job and
artifact upload. Remote CI has not been executed by this task. Runtime remains
standard-library-only; the wheel includes the synthetic recorder fixture.

Browser inspection verified all 50 plans render, the expand control works, no
review decisions are selected, and export is blocked when decisions or reviewer
fields are incomplete. That inspection is UI verification, not the human
plausibility review.

## Review and reproduce

Open [the 50-plan review gallery](phase_1_artifacts/review/index.html) in a browser.
Expand plans as needed, mark each **Plausible** or **Needs fix**, supply the actual
reviewer/date, and export `floor_plan_review.json`. Instructions are in
`docs/floor_plan_review.md`. The supplied user attestation has been validated against this gallery;
`phase_passed` is now true. The same workflow applies to future changed galleries.

```sh
make build
make phase1 WORLD_WORKERS=8
python3 -m ares phase1 --workers 8 --review /absolute/path/floor_plan_review.json --require-complete
```

Generated evidence is under `reports/phase_1_artifacts/`: the corpus manifest
contains regeneration arguments and canonical hashes for every scene; `review/`
contains the gallery and its manifest. Generated artifacts are git-ignored, while
this report and the source are retained. Re-running identical generation options
regenerates the matching gallery. Changing the generator or scene content
invalidates previous reviews rather than silently reusing them.

The new code and assets retain the project's pending-license status. No third-
party source, assets, or runtime dependency was added. Synthetic geometry is
bounded and simplified; it is not an architectural or building-code model.
Recorded-map conversion has synthetic test evidence only. Photorealism, realistic
depth noise, flight dynamics, and real sensor calibration remain later phases.

Routine gallery, build and packaging work used low-effort agents. Import and
independent geometry/contract review used medium-effort agents. The main agent
retained layout generation, mesh construction, corpus orchestration, partition
integrity, integration, and gate reporting.
