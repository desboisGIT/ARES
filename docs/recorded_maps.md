# Annotated recorded-map import

`ares.importer.load_recorded_map(path)` reads an `ares.recorded_map.v1` JSON
export and returns a validated, strict `ares.scene.v1` object.
`import_recorded_map(recording)` accepts the decoded object. Both reject malformed
input with `ContractError` and leave inputs unchanged. The importer uses only the
Python standard library. It does not ingest raw depth frames, ROS bags, or SLAM
logs: a recorder/export adapter must first produce the format below.

The bundled `ares/data/recorded_maps/synthetic_enclosed_room.json` is a **synthetic
test fixture, not a real capture**. No sensor rig or recorded sensor logs were
available. It exercises a translated, enclosed room without asserting real-world
sensor fidelity. No real capture provenance is claimed.

## Export contract

All listed keys are required; extra keys are rejected. JSON duplicate keys,
nonfinite numbers, duplicate voxel entries, and out-of-bounds indices are errors.

| Key | Value |
| --- | --- |
| `schema_version` | `"ares.recorded_map.v1"` |
| `recording_id` | Nonempty source identifier; becomes `scene_id` |
| `seed` | Integer, excluding booleans |
| `units`, `frame` | `"m"`, `"right_handed_z_up"` |
| `license` | `{"spdx": "…", "commercial_ok": true/false}` from source |
| `grid` | `resolution`, `origin`, `shape`, `cells` described below |
| `rooms` | Array of `{room_id, label, polygon}`; at least one room |
| `doors` | Array of `{door_id, room_ids, state, voxel_indices}`; explicit `[]` when none |
| `lighting` | One `{room_id, state}` per room; state `"lit"` or `"dark"` |
| `spawn` | Three finite world coordinates, in recorded free space inside a room |
| `default_material` | `"normal"`, `"shiny"`, `"black"`, `"glass"`, or `"mirror"` |

`grid.resolution` must be exactly `0.2` metres; resampling is unsupported.
`grid.origin` is a finite three-vector locating voxel index `[0,0,0]`'s lower
corner in world coordinates. `grid.shape` contains three positive integers.
The grid is limited to 1,000,000 voxels, checked before occupancy expansion.
Extents must be finite and the origin must permit a representable 0.2 metre step.
`grid.cells` is an array of `[i,j,k,value]` integer rows. Indices must be inside
the shape. Values are `0` unknown, `1` free, and `2` occupied. Missing entries
are unknown. Coordinates are not transformed: a cell occupies
`origin + 0.2 * [i,j,k]` through `origin + 0.2 * [i+1,j+1,k+1]`.

Room polygons are world XY metre coordinates, counterclockwise, simple, inside
the grid bounds. Door indices are grid indices; `room_ids` names two distinct
existing rooms bordering the doorway through horizontal free neighbours.
Annotations and spawn must be supplied from the recording/export process:
the importer does not infer room geometry, doorways, lighting, or spawn.

## Conservative scene conversion

Scene v1 has no unknown state. Every unknown voxel, including omitted cells,
becomes occupied collision geometry. Only explicitly free cells remain free.
Doors retain their original identifiers, room associations, state, and indices.
An open door requires explicitly free voxels; unknown voxels cannot be cleared
by an open annotation. A closed door requires occupied scene voxels (recorded
occupied or unknown); a recorded-free closed door is rejected as contradictory.
Door annotations never overwrite occupancy evidence.

The collision mesh consists of exact exposed voxel faces, triangulated, including
unknown geometry. Every occupied voxel receives the declared `default_material`.
This is a recorded **material assumption**, not a measured surface classification.
License and lighting metadata are retained, not certified by the adapter. A map
with no collision geometry cannot satisfy scene v1 and is rejected.

`import_recorded_map_with_provenance(recording)` returns `(scene, provenance)`.
The separate provenance dictionary includes the source schema and identifier,
source entry count, grid size, explicit and missing unknown counts, total unknown
count, free and recorded occupied counts, resulting occupied count, the
`"occupied"` unknown policy, and the default material assumption. Store it beside
the exported scene if needed; no provenance keys are added to scene v1.

```python
from ares.contracts import load_json
from ares.importer import import_recorded_map_with_provenance

scene, provenance = import_recorded_map_with_provenance(load_json("map.json"))
```

Import is deterministic for the same input and does not mutate source objects.
The output passes the existing `validate_scene` contract, including room, door,
material, lighting, collision-mesh, and spawn checks.
