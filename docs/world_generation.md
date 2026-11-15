# World generator v1

`generate_scene(seed, kind="apartment", split="train")` returns an independent
`ares.scene.v1` dictionary. All randomness comes from a named, explicitly seeded
stream. `ares.worldgen.v1` identifies this generator's initial algorithm. Changes
to generation semantics require a new generator version and corpus manifests.
The scene/profile contracts and Phase 0 fixture remain compatible.

## Geometry and contents

- `single_room`: a furnished rectangular room, with a side length of roughly
  3–6 m, labelled as a bedroom, living room, or study.
- `apartment`: a variable rectangular footprint partitioned into four to seven
  rooms when the minimum dimensions permit. Four-room plans combine living and
  kitchen space; larger plans separate them and add bedrooms or a study. Every
  apartment has a bathroom, an entrance room, and sleeping/living space.
- `floor`: two to four unit pairs around a shared 1.6–2.0 m corridor. Each unit
  has living/kitchen space, a bedroom and bathroom. Private units connect to the
  corridor through their own entrance; doors never join different units directly.
  Floors contain 13–25 labelled spaces including the corridor.

Dimensions lie on a 0.2 m lattice. Floors, ceilings, walls, door leaves, and solid
furniture boxes are occupied voxels. Door openings are 1 m wide and 2 m high.
Ceilings range from 2.6 to 3.2 m above the interior floor. Furniture represents
beds, wardrobes, sofas, tables, counters, and bathroom fixtures with simple solid
volumes rather than detailed meshes. Full door aprons and connecting routes are
reserved before furnishing; the corridor is kept unfurnished. All building shells
are enclosed, and the nominal 0.15 m drone has clearance at the spawn.

A seeded spanning tree of adjacent rooms establishes architectural connectivity;
some additional links are included. Some room doors are closed, so physical
reachability depends on door state. Glass, mirror, shiny, and black material
patches are physical collision surfaces. Materials and lit/dark room states are
metadata for the later sensor model; no optical simulation happens here.

`voxel_mesh` extracts the exact union boundary and merges adjacent coplanar voxel
faces into rectangles with outward winding. There are no internal collision
faces. Merging keeps large scenes compact without changing collision geometry.
Coplanar rectangle triangulation can have T-junctions; downstream code requiring
a topologically conforming mesh must retessellate. Voxel collision geometry is
authoritative in the existing evaluator. Surface area and signed volume are
checked independently against voxel counts in every corpus case; analytic tests
cover winding, concavities, disconnected solids, origins, and coordinate precision.

These are bounded synthetic layouts, not a building-code model or an architectural
dataset. The generator does not claim all unusual arrangements are plausible;
the human review gate exists to inspect that limitation. Scene licensing remains
`NOASSERTION`, `commercial_ok: false`, pending owner selection; no assets or new
runtime packages were imported from third parties.

## Dataset separation and reproducibility

Candidates are drawn from `ares.worldgen.v1:{kind}:{split}:{attempt}` streams.
The SHA-256 digest of canonical grid occupancy determines a split bucket:
bucket 0 is test, 1 validation, and 2–9 training. A call uses deterministic
rejection sampling until the requested bucket is reached (a bounded 512 attempts).
The collision mesh is built only for the selected candidate. This assigns all
identical occupied geometry to one partition even if materials or lighting differ;
it adds no split marker to the scene. Rejection is partition selection among
constructed scenes, not repair of invalid scenes. Failed construction or exceeded
limits are reported; individual scenes are never edited to make a corpus pass.

The full corpus uses seed 1741 and 10,000 sequential explicit case seeds.
Kinds alternate evenly; 80% of requests use training, 10% validation, 10% test.
The corpus manifest records each case's kind, split, seed, room/voxel/triangle
counts, full-scene hash, identity-independent content hash, and occupancy hash.
Its own deterministic hash excludes wall-clock timings. Parallel worker count
changes throughput rather than scene selection or corpus identity.

Schema/geometry integrity checks may inspect every partition. Learned model
training or tuning must consume only training scenes; validation is for model
selection and the test partition is reserved for final held-out evaluation.
The human gallery samples training cases only. No policy training occurs in
Phase 1. Reproduce a manifest entry with `ares generate` and its stored arguments,
then compare the canonical scene hash.

## Gate and throughput measurement

`ares phase1` checks full schema validity, enclosing shell, spawn clearance,
voxel/mesh surface area and volume, all difficulty levels/materials/door states/
lighting states, and cross-split geometry duplication. It records failures with
their explicit seed rather than skipping or replacing them. Passing requires
10,000 valid cases and at least one scene per second, plus the importer fixture.

Timing includes candidate generation, mesh extraction, validation and hashing;
it excludes gallery construction and writing full scene JSON files. Worker count,
Python/platform details, elapsed wall time, and rate are reported. One CPU worker
is the default; use `--workers` for explicitly reported process parallelism.
This benchmark concerns world preparation, not GPU simulation. The existing
paired axial benchmark retains its 10% regression gate; the new corpus rate is
an initial measurement and a separate minimum-rate CI gate.

The recorded-map adapter is tested using a clearly synthetic fixture. It supports
annotated exports of real occupancy maps but cannot validate a real capture that
has not been supplied. It preserves unknown space conservatively as occupied.

`software_passed` covers the automatic checks. `phase_passed` also requires a
matching, complete human attestation of at least 50 plausible plans. Neither a
test fixture nor an agent's visual inspection supplies that human attestation.
