"""Strict, dependency-free readers for the versioned Phase 0 contracts."""

import json
import math
from pathlib import Path


class ContractError(ValueError):
    """A document violates a supported ARES contract."""


def _fail(path, reason):
    raise ContractError(f"{path}: {reason}")


def _object(value, keys, path):
    if not isinstance(value, dict):
        _fail(path, "expected object")
    missing, extra = set(keys) - value.keys(), value.keys() - set(keys)
    if missing:
        _fail(path, f"missing keys: {', '.join(sorted(missing))}")
    if extra:
        _fail(path, f"unknown keys: {', '.join(sorted(extra))}")


def _number(value, path, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "expected integer" if integer else "expected finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        _fail(path, "expected finite number")
    if integer and not isinstance(value, int):
        _fail(path, "expected integer")
    return value


def _string(value, path):
    if not isinstance(value, str) or not value.strip():
        _fail(path, "expected nonempty string")


def _list(value, path, length=None):
    if not isinstance(value, list) or (length is not None and len(value) != length):
        _fail(path, "expected list" if length is None else f"expected list of length {length}")
    return value


def _vector(value, n, path, integer=False):
    _list(value, path, n)
    for i, item in enumerate(value):
        _number(item, f"{path}[{i}]", integer)
    return tuple(value)


def load_json(path) -> dict:
    """Read an object, rejecting duplicate keys and nonstandard NaN/Infinity."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail("json", f"duplicate key: {key}")
            result[key] = value
        return result

    def constant(value):
        _fail("json", f"nonfinite constant: {value}")

    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"),
                            object_pairs_hook=pairs, parse_constant=constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"json: {exc}") from exc
    if not isinstance(result, dict):
        _fail("json", "expected object")
    return result


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(p, a, b):
    return abs(_cross(a, b, p)) < 1e-10 and all(
        min(a[i], b[i]) - 1e-10 <= p[i] <= max(a[i], b[i]) + 1e-10 for i in range(2))


def _intersects(a, b, c, d):
    x, y, z, w = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    return ((x * y < 0 and z * w < 0) or _on_segment(c, a, b)
            or _on_segment(d, a, b) or _on_segment(a, c, d) or _on_segment(b, c, d))


def point_in_polygon(point, polygon):
    """Return whether an XY point lies in or on a simple polygon."""
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if _on_segment(point, a, b):
            return True
        if (a[1] > point[1]) != (b[1] > point[1]):
            if point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]:
                inside = not inside
    return inside


def validate_scene(scene) -> None:
    _object(scene, {"schema_version", "scene_id", "seed", "units", "frame", "license", "grid",
                    "collision_mesh", "rooms", "doors", "surfaces", "lighting", "spawn"}, "scene")
    for key, expected in (("schema_version", "ares.scene.v1"), ("units", "m"), ("frame", "right_handed_z_up")):
        if scene[key] != expected:
            _fail(key, f"expected {expected!r}")
    _string(scene["scene_id"], "scene_id")
    _number(scene["seed"], "seed", True)
    _object(scene["license"], {"spdx", "commercial_ok"}, "license")
    _string(scene["license"]["spdx"], "license.spdx")
    if not isinstance(scene["license"]["commercial_ok"], bool):
        _fail("license.commercial_ok", "expected boolean")
    grid = scene["grid"]
    _object(grid, {"resolution", "origin", "shape", "occupied"}, "grid")
    resolution = _number(grid["resolution"], "grid.resolution")
    if resolution != 0.2:
        _fail("grid.resolution", "expected 0.2 metres")
    origin = _vector(grid["origin"], 3, "grid.origin")
    shape = _vector(grid["shape"], 3, "grid.shape", True)
    if any(n <= 0 for n in shape):
        _fail("grid.shape", "dimensions must be positive")

    def cells(values, path):
        _list(values, path)
        result = set()
        for n, value in enumerate(values):
            cell = _vector(value, 3, f"{path}[{n}]", True)
            if any(not 0 <= cell[i] < shape[i] for i in range(3)):
                _fail(path, "voxel index out of bounds")
            if cell in result:
                _fail(path, "duplicate voxel index")
            result.add(cell)
        return result

    occupied = cells(grid["occupied"], "grid.occupied")
    mesh = scene["collision_mesh"]
    _object(mesh, {"vertices", "triangles"}, "collision_mesh")
    _list(mesh["vertices"], "collision_mesh.vertices")
    vertices = [_vector(v, 3, f"collision_mesh.vertices[{i}]") for i, v in enumerate(mesh["vertices"])]
    if not vertices or len(set(vertices)) != len(vertices):
        _fail("collision_mesh.vertices", "must be nonempty and unique")
    upper = [origin[i] + shape[i] * resolution for i in range(3)]
    if any(any(v[i] < origin[i] - 1e-9 or v[i] > upper[i] + 1e-9 for i in range(3)) for v in vertices):
        _fail("collision_mesh.vertices", "vertex outside grid bounds")
    _list(mesh["triangles"], "collision_mesh.triangles")
    seen = set()
    for i, value in enumerate(mesh["triangles"]):
        t = _vector(value, 3, f"collision_mesh.triangles[{i}]", True)
        if any(not 0 <= index < len(vertices) for index in t):
            _fail("collision_mesh.triangles", "vertex index out of bounds")
        if len(set(t)) != 3 or tuple(sorted(t)) in seen:
            _fail("collision_mesh.triangles", "duplicate or degenerate triangle")
        seen.add(tuple(sorted(t)))
        a, b, c = (vertices[index] for index in t)
        ab, ac = [b[j] - a[j] for j in range(3)], [c[j] - a[j] for j in range(3)]
        if sum((ab[(j+1)%3]*ac[(j+2)%3] - ab[(j+2)%3]*ac[(j+1)%3])**2 for j in range(3)) < 1e-20:
            _fail("collision_mesh.triangles", "degenerate triangle")
    if not seen:
        _fail("collision_mesh.triangles", "must be nonempty")
    _list(scene["rooms"], "rooms")
    rooms = {}
    for i, room in enumerate(scene["rooms"]):
        path = f"rooms[{i}]"
        _object(room, {"room_id", "label", "polygon"}, path)
        for key in ("room_id", "label"):
            _string(room[key], f"{path}.{key}")
        if room["room_id"] in rooms:
            _fail("rooms", "duplicate room_id")
        polygon = _list(room["polygon"], f"{path}.polygon")
        if len(polygon) < 3:
            _fail(f"{path}.polygon", "requires at least three vertices")
        points = [_vector(v, 2, f"{path}.polygon[{j}]") for j, v in enumerate(polygon)]
        if len(set(points)) != len(points):
            _fail(f"{path}.polygon", "duplicate polygon vertex")
        if any(any(not origin[j] <= p[j] <= upper[j] for j in range(2)) for p in points):
            _fail(f"{path}.polygon", "vertex outside grid bounds")
        edges = list(zip(points, points[1:] + points[:1]))
        for j, (a, b) in enumerate(edges):
            for k, (c, d) in enumerate(edges[j+1:], j+1):
                if k == j + 1 or (j == 0 and k == len(edges) - 1):
                    continue
                if _intersects(a, b, c, d):
                    _fail(f"{path}.polygon", "self-intersecting polygon")
        area2 = sum(a[0]*b[1] - a[1]*b[0] for a, b in edges)
        if area2 <= 1e-10:
            _fail(f"{path}.polygon", "requires positive counterclockwise area")
        rooms[room["room_id"]] = room
    if not rooms:
        _fail("rooms", "must be nonempty")
    _list(scene["doors"], "doors")
    door_ids = set()
    door_cells = set()
    for i, door in enumerate(scene["doors"]):
        path = f"doors[{i}]"
        _object(door, {"door_id", "room_ids", "state", "voxel_indices"}, path)
        _string(door["door_id"], f"{path}.door_id")
        if door["door_id"] in door_ids:
            _fail("doors", "duplicate door_id")
        door_ids.add(door["door_id"])
        ids = _list(door["room_ids"], f"{path}.room_ids", 2)
        for rid in ids:
            _string(rid, f"{path}.room_ids")
        if ids[0] == ids[1] or any(rid not in rooms for rid in ids):
            _fail(f"{path}.room_ids", "requires two distinct existing rooms")
        if door["state"] not in ("open", "closed"):
            _fail(f"{path}.state", "expected open or closed")
        indices = cells(door["voxel_indices"], f"{path}.voxel_indices")
        if not indices:
            _fail(f"{path}.voxel_indices", "must be nonempty")
        if door_cells & indices:
            _fail("doors", "door voxel assignments overlap")
        door_cells |= indices
        if door["state"] == "open" and indices & occupied:
            _fail(path, "open door voxels must be free")
        if door["state"] == "closed" and not indices <= occupied:
            _fail(path, "closed door voxels must be occupied")
        # Every listed room must border the doorway through a horizontal face.
        for rid in ids:
            if not any(point_in_polygon([origin[0]+(x+dx+0.5)*resolution,
                                         origin[1]+(y+dy+0.5)*resolution], rooms[rid]["polygon"])
                       and 0 <= x+dx < shape[0] and 0 <= y+dy < shape[1]
                       and (x+dx, y+dy, z) not in occupied
                       for x, y, z in indices for dx, dy in ((1,0),(-1,0),(0,1),(0,-1))):
                _fail(path, "door must border both referenced rooms")
    _list(scene["surfaces"], "surfaces")
    assigned = set()
    for i, surface in enumerate(scene["surfaces"]):
        path = f"surfaces[{i}]"
        _object(surface, {"material", "voxel_indices"}, path)
        if surface["material"] not in ("normal", "shiny", "black", "glass", "mirror"):
            _fail(f"{path}.material", "unknown depth material")
        indices = cells(surface["voxel_indices"], f"{path}.voxel_indices")
        if not indices <= occupied:
            _fail(path, "surface voxels must be occupied")
        if assigned & indices:
            _fail("surfaces", "duplicate surface assignment")
        assigned |= indices
    if assigned != occupied:
        _fail("surfaces", "every occupied voxel requires a material")
    _list(scene["lighting"], "lighting")
    lit_ids = set()
    for i, light in enumerate(scene["lighting"]):
        path = f"lighting[{i}]"
        _object(light, {"room_id", "state"}, path)
        if not isinstance(light["room_id"], str) or light["room_id"] not in rooms:
            _fail(path, "unknown room_id")
        if light["room_id"] in lit_ids:
            _fail("lighting", "duplicate room_id")
        lit_ids.add(light["room_id"])
        if light["state"] not in ("lit", "dark"):
            _fail(f"{path}.state", "expected lit or dark")
    if lit_ids != rooms.keys():
        _fail("lighting", "every room requires lighting state")
    spawn = _vector(scene["spawn"], 3, "spawn")
    cell = tuple(math.floor((spawn[i] - origin[i]) / resolution) for i in range(3))
    if any(not 0 <= cell[i] < shape[i] for i in range(3)):
        _fail("spawn", "outside grid bounds")
    if cell in occupied:
        _fail("spawn", "must lie in free space")
    if not any(point_in_polygon(spawn[:2], room["polygon"]) for room in rooms.values()):
        _fail("spawn", "must lie in a room polygon")


_POSITIVE = {"mass_kg", "max_thrust_n", "motor_lag_s", "battery_energy_j", "depth_horizontal_fov_rad",
             "depth_vertical_fov_rad", "depth_width", "depth_height", "depth_fps", "depth_max_range_m", "body_radius_m"}
_NONNEGATIVE = {"linear_drag", "depth_noise_std_m", "imu_accel_noise_std", "imu_gyro_noise_std",
                "odometry_drift_per_m", "compute_latency_s"}
_PIXELS = {"depth_width", "depth_height"}
_V2_POSITIVE = {"gravity_mps2", "depth_min_range_m", "depth_edge_threshold_m", "imu_rate_hz", "optical_flow_rate_hz"}
_V2_NONNEGATIVE = {"depth_noise_quadratic_per_m2", "depth_edge_width_px", "max_range_dropout_probability",
                   "optical_flow_noise_std_mps", "range_noise_std_m"}


def validate_profile(profile) -> None:
    _object(profile, {"schema_version", "profile_id", "units", "parameters"}, "profile")
    if profile["schema_version"] not in ("ares.profile.v1", "ares.profile.v2"):
        _fail("schema_version", "expected 'ares.profile.v1' or 'ares.profile.v2'")
    v2 = profile["schema_version"] == "ares.profile.v2"
    if profile["units"] != "SI":
        _fail("units", "expected 'SI'")
    _string(profile["profile_id"], "profile_id")
    params = profile["parameters"]
    extra = _V2_POSITIVE | _V2_NONNEGATIVE | {"material_hole_probability"} if v2 else set()
    _object(params, _POSITIVE | _NONNEGATIVE | {"inertia_kg_m2"} | extra, "parameters")
    for name, param in params.items():
        path = f"parameters.{name}"
        _object(param, {"nominal", "range"}, path)
        vector = name in ("inertia_kg_m2", "material_hole_probability")
        if vector:
            size = 3 if name == "inertia_kg_m2" else 5
            nominal = _vector(param["nominal"], size, f"{path}.nominal")
            ranges = _list(param["range"], f"{path}.range", size)
        else:
            nominal, ranges = [param["nominal"]], [param["range"]]
        for i, (value, bounds) in enumerate(zip(nominal, ranges)):
            integer = name in _PIXELS or name == "depth_edge_width_px"
            low, high = _vector(bounds, 2, f"{path}.range" + (f"[{i}]" if vector else ""), integer)
            _number(value, f"{path}.nominal", integer)
            if low > high:
                _fail(f"{path}.range", "lower bound exceeds upper bound")
            if not low <= value <= high:
                _fail(f"{path}.nominal", "outside randomization range")
            if (name == "inertia_kg_m2" or name in _POSITIVE | _V2_POSITIVE) and low <= 0:
                _fail(f"{path}.range", "values must be positive")
            if name in _NONNEGATIVE | _V2_NONNEGATIVE and low < 0:
                _fail(f"{path}.range", "values must be nonnegative")
            if name in ("material_hole_probability", "max_range_dropout_probability") and not 0 <= low <= high <= 1:
                _fail(f"{path}.range", "probabilities must lie in [0,1]")
            if name == "depth_edge_width_px" and high > 32:
                _fail(f"{path}.range", "edge width must stay in [0,32]")
            if name.endswith("fov_rad") and high >= math.pi:
                _fail(f"{path}.range", "field of view must be below pi radians")
        if name == "inertia_kg_m2":
            if any(nominal[i] > sum(nominal) - nominal[i] for i in range(3)):
                _fail(path, "inertia violates triangle inequalities")
            if any(ranges[i][1] > sum(ranges[j][0] for j in range(3) if j != i) for i in range(3)):
                _fail(path, "randomized inertia can violate triangle inequalities")
    if v2 and params["depth_min_range_m"]["range"][1] >= params["depth_max_range_m"]["range"][0]:
        _fail("parameters.depth_min_range_m", "minimum range must stay below maximum range")
