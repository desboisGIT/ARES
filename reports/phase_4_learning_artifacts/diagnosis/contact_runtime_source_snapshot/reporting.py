"""Versioned report contract. A report contains claims, never scoring authority."""

import math

from .contracts import ContractError, _intersects


def _keys(value, expected, path):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ContractError(f"{path}: expected keys {sorted(expected)}")


def _number(value, path, minimum=None):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (float, int))
                 and math.isfinite(value) and (minimum is None or value >= minimum))
    except OverflowError:
        valid = False
    if not valid:
        raise ContractError(f"{path}: expected finite number" + (f" >= {minimum}" if minimum is not None else ""))


def _vector(value, length, path, integer=False, minimum=None):
    if not isinstance(value, list) or len(value) != length:
        raise ContractError(f"{path}: expected array of length {length}")
    for item in value:
        _number(item, path, minimum)
        if integer and type(item) is not int:
            raise ContractError(f"{path}: expected integers")


def validate_report(report):
    _keys(report, {"schema_version", "voxel_map", "floor_plan", "rooms", "person_markers", "end_time_s"}, "report")
    if report["schema_version"] != "ares.report.v1":
        raise ContractError("report.schema_version: unsupported version")
    _number(report["end_time_s"], "report.end_time_s", 0)
    for name, dimension in (("voxel_map", 3), ("floor_plan", 2)):
        grid = report[name]
        _keys(grid, {"resolution", "origin", "shape", "cells"}, name)
        _number(grid["resolution"], f"{name}.resolution", 1e-9)
        _vector(grid["origin"], dimension, f"{name}.origin")
        _vector(grid["shape"], dimension, f"{name}.shape", True, 1)
        if not isinstance(grid["cells"], list):
            raise ContractError(f"{name}.cells: expected array")
        if name == "voxel_map":
            seen = set()
            for cell in grid["cells"]:
                _vector(cell, 4, "voxel_map.cells", True, 0)
                index = tuple(cell[:3])
                if any(index[i] >= grid["shape"][i] for i in range(3)) or index in seen:
                    raise ContractError("voxel_map.cells: out of bounds or duplicate cell")
                if cell[3] not in (1, 2):
                    raise ContractError("voxel_map.cells: value must be 1 free or 2 occupied")
                seen.add(index)
        else:
            if len(grid["cells"]) != math.prod(grid["shape"]):
                raise ContractError("floor_plan.cells: length must match shape")
            if any(type(v) is not int or v not in (0, 1, 2) for v in grid["cells"]):
                raise ContractError("floor_plan.cells: expected 0 unknown, 1 free, 2 occupied")
    if not isinstance(report["rooms"], list):
        raise ContractError("report.rooms: expected array")
    ids = set()
    for room in report["rooms"]:
        _keys(room, {"room_id", "polygon"}, "report.rooms")
        if not isinstance(room["room_id"], str) or not room["room_id"] or room["room_id"] in ids:
            raise ContractError("report.rooms: room ids must be nonempty and unique")
        ids.add(room["room_id"])
        if not isinstance(room["polygon"], list) or len(room["polygon"]) < 3:
            raise ContractError("report.rooms.polygon: expected at least 3 vertices")
        for vertex in room["polygon"]:
            _vector(vertex, 2, "report.rooms.polygon")
        polygon = room["polygon"]
        if len({tuple(p) for p in polygon}) != len(polygon):
            raise ContractError("report.rooms.polygon: duplicate vertices")
        floor = report["floor_plan"]
        if any(not floor["origin"][i] <= p[i] <= floor["origin"][i] + floor["shape"][i] * floor["resolution"]
               for p in polygon for i in range(2)):
            raise ContractError("report.rooms.polygon: outside floor-plan bounds")
        edges = list(zip(polygon, polygon[1:] + polygon[:1]))
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in edges) / 2
        if abs(area) < 1e-12:
            raise ContractError("report.rooms.polygon: zero area")
        for i, (a, b) in enumerate(edges):
            for j, (c, d) in enumerate(edges[i + 1:], i + 1):
                if j == i + 1 or (i == 0 and j == len(edges) - 1):
                    continue
                if _intersects(a, b, c, d):
                    raise ContractError("report.rooms.polygon: self intersection")
    # Reserved extension; accepting arbitrary markers would freeze an undefined
    # people contract before Phase 10. v1 intentionally supports only [].
    if report["person_markers"] != []:
        raise ContractError("report.person_markers: v1 requires an empty array")


def make_report(world, observations, end_time_s, rooms=()):
    grid = world.scene["grid"]
    nx, ny, _ = grid["shape"]
    floor = [0] * (nx * ny)
    height = world.index(world.spawn)[2]
    for (x, y, z), value in observations.items():
        if z == height:
            floor[y * nx + x] = value
    return {
        "schema_version": "ares.report.v1",
        "voxel_map": {"resolution": grid["resolution"], "origin": list(grid["origin"]),
                      "shape": list(grid["shape"]),
                      "cells": [[*cell, value] for cell, value in sorted(observations.items())]},
        "floor_plan": {"resolution": grid["resolution"], "origin": list(grid["origin"][:2]),
                       "shape": list(grid["shape"][:2]), "cells": floor},
        "rooms": [{"room_id": room["room_id"], "polygon": [list(p) for p in room["polygon"]]}
                  for room in rooms],
        "person_markers": [], "end_time_s": end_time_s,
    }
