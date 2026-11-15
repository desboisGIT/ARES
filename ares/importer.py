"""Conservative, dependency-free occupancy export importer.

This adapter accepts annotated recorder exports, not raw sensor streams. Unknown
space remains collision geometry because scene v1 has no unknown state.
"""

from copy import deepcopy
import math

from .contracts import (ContractError, _fail, _list, _number, _object, _string,
                        _vector, load_json, validate_scene)
from .geometry import voxel_mesh

MAX_GRID_VOXELS = 1_000_000
_KEYS = {"schema_version", "recording_id", "seed", "units", "frame", "license",
         "grid", "rooms", "doors", "lighting", "spawn", "default_material"}
_MATERIALS = ("normal", "shiny", "black", "glass", "mirror")


def import_recorded_map_with_provenance(recording: dict) -> tuple[dict, dict]:
    """Return a validated scene and conversion provenance without mutating input.

    Cell indices use the supplied grid origin; annotation polygons and spawn
    already use world metres. No transform, room inference, or hole filling is
    performed. Explicitly free cells alone are traversable.
    """
    _object(recording, _KEYS, "recording")
    for key, expected in (("schema_version", "ares.recorded_map.v1"),
                          ("units", "m"), ("frame", "right_handed_z_up")):
        if recording[key] != expected:
            _fail(key, f"expected {expected!r}")
    _string(recording["recording_id"], "recording_id")
    _number(recording["seed"], "seed", True)
    if recording["default_material"] not in _MATERIALS:
        _fail("default_material", "unknown depth material")
    grid = recording["grid"]
    _object(grid, {"resolution", "origin", "shape", "cells"}, "grid")
    resolution = _number(grid["resolution"], "grid.resolution")
    if resolution != 0.2:
        _fail("grid.resolution", "expected 0.2 metres; resampling is not supported")
    origin = _vector(grid["origin"], 3, "grid.origin")
    shape = _vector(grid["shape"], 3, "grid.shape", True)
    if any(n <= 0 for n in shape):
        _fail("grid.shape", "dimensions must be positive")
    total = math.prod(shape)
    if total > MAX_GRID_VOXELS:
        _fail("grid.shape", f"exceeds resource limit of {MAX_GRID_VOXELS} voxels")
    for axis in range(3):
        upper = origin[axis] + shape[axis] * resolution
        if not math.isfinite(upper) or origin[axis] + resolution == origin[axis]:
            _fail(f"grid.origin[{axis}]", "cannot represent finite 0.2 metre voxel coordinates")
    spawn = _vector(recording["spawn"], 3, "spawn")
    if any(not origin[a] <= spawn[a] < origin[a] + shape[a] * resolution
           for a in range(3)):
        _fail("spawn", "outside grid bounds")
    cells = _list(grid["cells"], "grid.cells")
    if len(cells) > total:
        _fail("grid.cells", "more entries than grid voxels")
    values = {}
    for n, row in enumerate(cells):
        path = f"grid.cells[{n}]"
        _list(row, path, 4)
        index = _vector(row[:3], 3, path, True)
        if any(not 0 <= index[a] < shape[a] for a in range(3)):
            _fail(path, "voxel index out of bounds")
        if index in values:
            _fail(path, "duplicate voxel index")
        value = _number(row[3], f"{path}[3]", True)
        if value not in (0, 1, 2):
            _fail(f"{path}[3]", "expected 0 (unknown), 1 (free), or 2 (occupied)")
        values[index] = value
    occupied = {(x, y, z) for x in range(shape[0]) for y in range(shape[1])
                for z in range(shape[2]) if values.get((x, y, z), 0) != 1}
    if not occupied:
        _fail("grid.cells", "scene v1 requires nonempty collision geometry")
    # Annotation semantics, including door state versus occupancy, are checked by
    # the scene contract. An open door cannot silently clear an unknown voxel.
    scene = {
        "schema_version": "ares.scene.v1", "scene_id": recording["recording_id"],
        "seed": recording["seed"], "units": "m", "frame": "right_handed_z_up",
        "license": deepcopy(recording["license"]),
        "grid": {"resolution": resolution, "origin": list(origin),
                 "shape": list(shape), "occupied": [list(c) for c in sorted(occupied)]},
        "collision_mesh": voxel_mesh(occupied, resolution=resolution, origin=origin),
        "rooms": deepcopy(recording["rooms"]), "doors": deepcopy(recording["doors"]),
        "surfaces": [{"material": recording["default_material"],
                      "voxel_indices": [list(c) for c in sorted(occupied)]}],
        "lighting": deepcopy(recording["lighting"]), "spawn": deepcopy(recording["spawn"]),
    }
    validate_scene(scene)
    counts = {value: sum(v == value for v in values.values()) for value in (0, 1, 2)}
    provenance = {
        "source_schema_version": recording["schema_version"],
        "recording_id": recording["recording_id"],
        "source_cell_count": len(values), "grid_voxel_count": total,
        "explicit_unknown_count": counts[0], "missing_cell_count": total - len(values),
        "unknown_cell_count": counts[0] + total - len(values),
        "free_cell_count": counts[1], "recorded_occupied_count": counts[2],
        "scene_occupied_count": len(occupied), "unknown_policy": "occupied",
        "default_material_assumption": recording["default_material"],
    }
    return scene, provenance


def import_recorded_map(recording: dict) -> dict:
    """Convert an annotated ``ares.recorded_map.v1`` object to strict scene v1."""
    return import_recorded_map_with_provenance(recording)[0]


def load_recorded_map(path) -> dict:
    """Read a JSON recorder export (rejecting duplicate keys) and import it."""
    return import_recorded_map(load_json(path))
