# Classical observed-map explorer

`ares.exploration.ClassicalExplorer` is an independent classical baseline. It
uses frontier-adjacent viewpoints, expected information gain and global grid
routing, ideas common to frontier planners such as FUEL and FALCON. It contains
no code from those projects and adds only NumPy and SciPy (BSD) runtime
requirements through the simulation extra.

The research references are [FUEL](https://arxiv.org/abs/2010.11561)
(incremental frontier structure and hierarchical planning) and
[FALCON](https://arxiv.org/abs/2407.00577) (coverage-path guidance).
This baseline uses their broad exploration-planning ideas. It does not
reproduce their trajectory optimizers or establish a comparison to their
published performance.

Construct it with `ClassicalExplorer()` or an `ExplorationConfig`. Supply one
public observation dictionary to `act`; it returns a NumPy vector
`[world_vx, world_vy, world_vz, yaw_rate, land_flag]`.

| Observation | Meaning |
| --- | --- |
| `known_grid` | XYZ uint8 map: unknown 0, observed free 1, observed occupied 2 |
| `resolution`, `origin` | Grid geometry in metres |
| `position`, `velocity`, `yaw`, `time_s` | Estimated world state and mission clock |
| `home` | Public launch/landing position |
| `body_radius_m`, `max_speed`, `max_yaw_rate`, `dt` | Public vehicle/control limits |
| `camera` (optional) | `hfov_rad`, `vfov_rad`, `max_range_m`; defaults 87°, 58°, 4 m |

The constructor receives no simulator, scene or evaluator. The policy does not
read hidden occupancy, true coverage, room/door labels, layout seed, rendering
objects or scorer output. Its only map is the supplied observed map. Unit tests
use an observation dictionary that raises on accesses outside this interface.
The rectangular map bounds are metadata, not evidence of free space.

An initial stationary full yaw sweep gathers the forward-facing camera map.
The planner erodes observed free space by the spherical vehicle hull, labels
reachable center positions using six-connected search, and samples spatially
distributed viewpoints near unknown space at multiple heights. For each
candidate it casts exact voxel-DDA rays through the observed map. Rays stop at known occupied
cells; unknown cells before those stops contribute expected gain. Horizontal
ray sectors are combined into the camera field of view to choose gaze. The
utility divides gain by travel cost and discounts previously visited views.
This is an expectation for selecting a view, never occupancy evidence for
navigation or reports. No unobserved cell becomes traversable.

Global routes use equal-cost six-connected search, retaining centers at all
turns and compressing straight runs. This prevents diagonal corner cutting.
At 20 cm resolution and 15 cm body radius, configuration-space inflation
includes the center, six face neighbors and twelve edge neighbors: 19 cells.
A further exact segment-versus-voxel swept-sphere check validates each action;
unknown cells and out-of-bounds space block motion. The action caps speed and
yaw rate using the public limits, slows when turning, and points the camera
along horizontal travel. Each reached view gets another stationary full sweep.
The model has yaw only, so height changes and horizontal travel are both needed
to observe space above/below the launch altitude.

The planner returns over a known-safe route when no eligible observed-map view
has sufficient expected gain, after 16 completed viewpoints each add fewer
than three observed voxels, or when its configured mission time budget expires. Neither condition claims a particular true coverage. It holds still
at home for at least 1.1 s before requesting landing. The default 3300 s policy
budget reserves time inside the harness's 3600 s timeout. Viewpoint revisit
limits and the budget guarantee bounded attempts, but they can leave unseen
space; only the independent evaluator determines success.

`extract_rooms(known_grid, resolution, origin)` returns dictionaries with
`room_id` and counterclockwise XY `polygon`. It uses an explicit generator
family prior: the layer 2.3 m above grid origin lies above doors/furniture and
below ceilings. Each sufficiently large observed-free component at that layer
gives an inferred rectangular footprint. The prior is public and independent
of any sampled scene metadata. This is suitable for the generated rectangular
room family, not a general real-building floor-plan estimator. Partial
observations can fragment or underestimate rooms. Polygons are geometric
inferences; the function never changes or fills the supplied voxel map, and a
report must preserve every unobserved voxel as unknown.

Run the focused checks with
`.venv/bin/python -m unittest tests.test_exploration -v`. They cover the
information boundary, exact hull geometry, occluded gain, command safety,
landing dwell, and observed-room extraction. These unit checks alone do not
establish the apartment pass-rate gate; that requires the separate frozen-policy
held-out benchmark, with no test layouts used for development.

## Development validation (train only)

These checks used generated apartment scenes with train seeds 100–109, the
default profile, CUDA FastWorld, 50 ms steps, and 2 m/s / 1.5 rad/s public
limits. No held-out layout was inspected or evaluated during development.
All ten runs passed the independent scorer, found every required room, had
zero collisions, and landed safely. These are development results, not the
held-out gate.

| Train seed | Time (s) | Coverage | Floor-plan IoU | Rooms found / required |
| --- | ---: | ---: | ---: | ---: |
| 100 | 159.25 | 1.000000 | 0.925731 | 1 / 1 |
| 101 | 299.35 | 0.999945 | 0.930756 | 4 / 4 |
| 102 | 278.60 | 0.999794 | 0.925592 | 4 / 4 |
| 103 | 442.35 | 0.999954 | 0.913154 | 5 / 5 |
| 104 | 205.35 | 1.000000 | 0.900601 | 1 / 1 |
| 105 | 211.60 | 0.999188 | 0.940196 | 3 / 3 |
| 106 | 408.20 | 0.999633 | 0.919449 | 5 / 5 |
| 107 | 343.90 | 1.000000 | 0.926473 | 4 / 4 |
| 108 | 215.90 | 0.999155 | 0.905917 | 3 / 3 |
| 109 | 279.60 | 0.999953 | 0.933488 | 4 / 4 |

Median time: 279.10 s. 90th percentile: 411.62 s.
Minimum coverage: 0.999155. Minimum floor-plan IoU: 0.900601.
