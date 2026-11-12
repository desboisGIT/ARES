# Phase 0 data contracts

Python 3.11 or later and the standard library are sufficient. The authoritative
validators are `ares.contracts.validate_scene` and `validate_profile`. Each accepts
a Python dictionary, returns `None` on success, and raises `ContractError` (a
`ValueError`) containing a field path and reason on failure. `load_json(path)` reads
UTF-8 JSON objects and rejects duplicate object keys, NaN, Infinity and malformed
JSON. Validators reject missing and unknown keys at every defined object level.
Numbers must be finite; booleans are never numbers. Integer fields require JSON
integers, not floats with integral values. No implicit conversion is performed.

Version strings are frozen: `ares.scene.v1` and `ares.profile.v1`. Every breaking
change to fields, types, units, coordinate semantics, or validation rules requires
a new version and explicit validator support. IDs and labels are nonempty strings.
Room labels are descriptive, scene-provided strings, not a fixed taxonomy.

## Scene v1

Every scene has exactly these top-level fields:

| Field | Type and meaning |
| --- | --- |
| `schema_version` | `"ares.scene.v1"` |
| `scene_id` | Nonempty scene identifier |
| `seed` | Explicit integer generator seed |
| `units` | `"m"` |
| `frame` | `"right_handed_z_up"`; x/y horizontal, z vertical |
| `license` | `{spdx: string, commercial_ok: boolean}` |
| `grid` | `{resolution: 0.2, origin: [x,y,z], shape: [nx,ny,nz], occupied: [[i,j,k],...]}` |
| `collision_mesh` | `{vertices: [[x,y,z],...], triangles: [[i,j,k],...]}` |
| `rooms` | `[{room_id: string, label: string, polygon: [[x,y],...]}]` |
| `doors` | `[{door_id: string, room_ids: [id,id], state: "open" or "closed", voxel_indices: [[i,j,k],...]}]` |
| `surfaces` | `[{material: string, voxel_indices: [[i,j,k],...]}]` |
| `lighting` | `[{room_id: string, state: "lit" or "dark"}]` |
| `spawn` | World position `[x,y,z]` |

The origin is the minimum corner of cell `[0,0,0]`. Cell `[i,j,k]` spans
`origin + 0.2*[i,j,k]` to `origin + 0.2*[i+1,j+1,k+1]`; point ownership is
half-open at the upper face. Shape dimensions are positive integers. Occupied
cells are unique in-bounds integer triples. Unlisted in-grid cells are free;
outside-grid space is outside the scene, never implicitly free.

Mesh vertices are unique finite world coordinates within the grid's closed bounds.
Triangles use zero-based vertex indices, must be nondegenerate and unique without
regard to winding, and vertices/triangles must be nonempty. The structural validator
does not prove watertightness, winding, or mesh/voxel geometric equivalence. The
original fixture generator emits precisely the exposed faces of occupied voxels,
with outward winding and two triangles per face; its test verifies that equality.

Rooms are nonempty with unique IDs. Polygons contain at least three unique
vertices, omit a repeated closing vertex, have counterclockwise positive area,
stay within XY grid bounds, and have no crossing or touching nonadjacent edges.
Polygons describe horizontal room interiors; wall thickness and door passages may
lie between them. Polygon membership includes boundary points. There is no fixed
room-height field; 3D occupancy determines free height.

Door IDs are unique. Each door references two distinct existing rooms and contains
nonempty unique in-bounds voxel indices. Different doors cannot share cells. Open
door cells are free; closed door cells are occupied. A horizontal neighbor of at
least one door cell must be free and have its XY center inside each referenced
room. This is a local adjacency check; body-size clearance and global room
reachability belong to the evidence/scoring layer.

Allowed depth materials are `normal`, `shiny`, `black`, `glass`, and `mirror`.
Every occupied voxel has exactly one material, and free voxels cannot have one.
Every room has exactly one lighting state. Spawn must lie in-grid, in a free cell,
and within a room polygon. Structural validation checks point occupancy; the
evidence layer checks the drone body's clearance.

License metadata records the supplied SPDX string and commercial flag; the
validator checks their types, not legal correctness. The hand-built scene is an
original fixture, not imported third-party geometry. The owner has not selected
a project or asset license. Its metadata is therefore `spdx: "NOASSERTION"` and
`commercial_ok: false`, pending owner selection; generation does not assign a license.

## Profile v1

The profile is exactly `{schema_version: "ares.profile.v1", profile_id: string,
units: "SI", parameters: {...}}`. Every parameter has exactly `nominal` and `range`.
Scalars use `range: [minimum, maximum]`. Inertia uses three nominal principal
moments and three corresponding `[minimum, maximum]` ranges in body x/y/z order.
Ranges are closed, ordered, and include the nominal value. Every required
parameter is present and no additional names are accepted.

| Parameter | Units / restrictions |
| --- | --- |
| `mass_kg` | kg; strictly positive |
| `inertia_kg_m2` | Three principal moments, kg m²; strictly positive |
| `max_thrust_n` | N, total collective thrust; strictly positive |
| `motor_lag_s` | s; strictly positive |
| `linear_drag` | kg/s, isotropic linear drag coefficient; nonnegative |
| `battery_energy_j` | J; strictly positive |
| `depth_horizontal_fov_rad`, `depth_vertical_fov_rad` | Radians, strictly between 0 and π |
| `depth_width`, `depth_height` | Positive integer pixels, including range endpoints |
| `depth_fps` | Frames per second; strictly positive |
| `depth_max_range_m` | m; strictly positive |
| `depth_noise_std_m` | m; nonnegative |
| `imu_accel_noise_std` | m/s² per simulated sample; nonnegative |
| `imu_gyro_noise_std` | rad/s per simulated sample; nonnegative |
| `odometry_drift_per_m` | m/m; nonnegative |
| `compute_latency_s` | s; nonnegative |
| `body_radius_m` | m; strictly positive spherical collision envelope |

Nominal inertia and every possible independent sample within its ranges satisfy
the principal-moment triangle inequalities. This requires each axis's upper bound
to be no greater than the sum of the other two lower bounds. Zero-width ranges
are permitted. v1 does not impose universal upper bounds on mass, latency, thrust,
or noise, or certify hardware feasibility. The nominal fixture describes a
0.5 kg, 0.15 m radius drone with 24.5 N collective thrust and low-resolution depth.
Sensor values are Phase 0 assumptions, not measured calibration or device claims.

The packaged authoritative files are `ares/data/scenes/three_room.json` and
`ares/data/configs/drone_nominal.json`. Regenerate deterministically with
`python3 scripts/generate_fixture.py`; no random library or external assets are used.
The scene is a 6.4 m × 2.4 m × 2.6 m grid with enclosed floor, walls and ceiling,
three rooms, and two 0.8 m × 2.0 m open doorways. The scene does not introduce
procedural generation, depth simulation, or a flight model.
