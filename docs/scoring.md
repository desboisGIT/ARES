# Scorer and harness contract v1

The supplied plan leaves the observable-space denominator, floor-plan classes,
room matching, and landing tolerances open. The definitions below freeze one
testable Phase 0 interpretation. Changes to these meanings require a new scorer
version and corresponding contract changes. They do not establish later gates.

## Authoritative evidence

`GroundTruthWorld` validates the scene/profile and constructs their SHA-256
identity from canonical JSON. `GroundTruthWorld.record` constructs immutable
`GroundTruthEpisode` telemetry and sensor evidence. The scorer refuses episodes
bound to a different scene/profile identity. Policy-owned input is only a report.
An application must keep the evaluator outside policy control; these Python
objects are not a security sandbox or authenticated remote telemetry protocol.

The harness is kinematic. It checks a spherical drone against occupied voxel
AABBs with an exact minimum segment-to-box distance, including the whole path
between samples. Touching an obstacle counts as a collision. The collision count
is the number of sample segments with contact, including an initial invalid
pose; it is not a count of distinct physical impacts. Body departure from the
grid bounds fails independently. Minimum clearance is surface-to-surface in
metres, and can be negative on collision. The collision mesh is retained in the
scene; the Phase 0 evaluator's authoritative collider is its voxel grid.

## Observable space

Before a policy runs, six-connected BFS finds all body-clear voxel centers
reachable from the spawn. For each center, six axial center rays extend to the
first occupied voxel or the profile's nominal depth range, whichever comes
first. The union of their traversed free voxels and first occupied voxels is
the fixed target set. Coverage is the number of correctly classified target
voxels that also occur in the trusted observation ledger, divided by this set's
size. Free and occupied target voxels have equal weight. Incorrect or unsupported
claimed voxels fail the episode independently of the coverage threshold.

The trace's scans use the same ideal sensor and only cell-centered, collision-free
poses. Six axial rays model a virtual orientation sweep; they have no scan-time
cost, camera FOV, yaw dynamics, depth image, noise, or material dropout. They are
a foundation fixture, not a RealSense simulation. Transparent/shiny materials and
lighting are preserved as scene data for Phase 2. Closed or body-inaccessible
spaces cannot expand the denominator through a policy's choices.

## Floor plan and rooms

The 20 cm floor-plan grid is the horizontal slice at the spawn voxel's z index.
Values are 0 unknown, 1 free, and 2 occupied; the reference slice is projected
from the fixed observable set. Storage order is `cells[y * nx + x]`. IoU is
computed separately for free and occupied cells over the entire declared grid,
then averaged. An empty class on both sides has IoU 1. Unknown claims give no
intersection; invented classes outside the reference set enlarge the union.
Every claimed known floor cell additionally needs trusted observation support.
Map origin, dimensions, and resolution must match the scene exactly. This
single-height definition must be revised for stairs or multistory scenes.

A room is reachable if its polygon contains at least one body-reachable camera
center. Discovery requires an observed free cell inside it and a corresponding
reported polygon with at least 0.80 IoU on the same XY grid. A deterministic
maximum bipartite matching pairs report polygons to scene polygons, so mapper
room IDs are arbitrary and one polygon cannot satisfy multiple rooms. All
reachable rooms must match, and unmatched claimed rooms fail. Report polygons
must be simple, have nonzero area, and lie within the grid's world bounds.

## Time, landing, and reward

Elapsed time comes from evaluator timestamps, starting at zero at the spawn and
strictly increasing. A report's end time must match. Zero-time and over-3600 s
episodes fail. The evaluator-only witness travels at a scripted 0.5 m/s; time
here cannot rank learned or classical physical flight policies.

The virtual landing station is centered on the spawn at spawn height. An explicit
evaluator land event plus a terminal dwell lasting at least 1 s with every
sample within 10 cm of that station, no collisions, and no bounds violations
establishes virtual safe landing. This is a harness event, not a descent,
touchdown, disarm, or flight-controller failsafe. Real landing belongs to the
later dynamics/hardware gates.

`ares.reward.dense_reward` accepts progress/time/collision training signals. Its
value has no path to the terminal scorer and cannot override any pass criterion.

## JSON outputs

All formats use SI units, right-handed z-up coordinates, finite JSON numbers,
and explicit schema versions. Scene/profile fields are specified in
`contracts.md`; report/log implementations also provide strict validators.

- `ares.report.v1`: `voxel_map` is `{resolution, origin[3], shape[3],
  cells:[[x,y,z,value],...]}` with unique bounded indices and values 1/2;
  omitted voxels are unknown. `floor_plan` is the equivalent dense 2D grid.
  `rooms` holds `{room_id,polygon}` objects; `person_markers` is reserved as `[]`;
  `end_time_s` is a finite nonnegative number.
- `ares.episode_result.v1`: pass/fail and reasons, authoritative time, coverage,
  floor-plan IoU, required/found room IDs, collisions, minimum clearance, virtual
  landing status, unsupported map cells, scorer/scene/profile versions and IDs,
  scene/profile digest, seed, and the exact `ScoringRules` values. A malformed
  report is a contract error, not a scoreable episode.
- `ares.flight_log.v1`: seed, scene/profile IDs, and timestamped pose-estimate
  samples. Poses are idealized harness poses with identity orientation. Depth,
  camera, IMU, commands, policy output, shield intervention, and map snapshot
  fields are present as null because Phase 0 produces none. The optional
  `ground_truth` block is rejected by default and contains separately copied
  evaluator poses and observed voxels. It must never be sent to onboard code.
- `ares.phase0.v1`: input snapshots, seed, all gate results, attacks, witness,
  invalid-scene rejection reasons, artifact digest, and explicit limitations.

Canonical artifact serialization sorts object keys, sorts voxel indices and
matched room IDs, and rejects NaN/Infinity. The deterministic gate runs the same
seed twice and compares complete mission/result/evaluator-log content.
