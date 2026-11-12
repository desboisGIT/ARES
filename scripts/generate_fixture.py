"""Regenerate original fixtures; asset licensing awaits the owner's selection."""

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_scene():
    shape = (32, 12, 13)
    occupied = set()
    door_cells = {x: {(x, y, z) for y in range(4, 8) for z in range(1, 11)} for x in (10, 21)}
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if (x in (0, 31) or y in (0, 11) or z in (0, 12)
                        or (x in door_cells and (x, y, z) not in door_cells[x])):
                    occupied.add((x, y, z))
    # Counterclockwise face corners viewed from outside the occupied cube.
    faces = [((-1,0,0), ((0,0,0),(0,0,1),(0,1,1),(0,1,0))),
             ((1,0,0), ((1,0,0),(1,1,0),(1,1,1),(1,0,1))),
             ((0,-1,0), ((0,0,0),(1,0,0),(1,0,1),(0,0,1))),
             ((0,1,0), ((0,1,0),(0,1,1),(1,1,1),(1,1,0))),
             ((0,0,-1), ((0,0,0),(0,1,0),(1,1,0),(1,0,0))),
             ((0,0,1), ((0,0,1),(1,0,1),(1,1,1),(0,1,1)))]
    vertices, vertex_ids, triangles = [], {}, []
    for cell in sorted(occupied):
        for direction, corners in faces:
            neighbor = tuple(cell[i] + direction[i] for i in range(3))
            if neighbor in occupied:
                continue
            ids = []
            for corner in corners:
                lattice = tuple(cell[i] + corner[i] for i in range(3))
                if lattice not in vertex_ids:
                    vertex_ids[lattice] = len(vertices)
                    vertices.append([round(v * 0.2, 10) for v in lattice])
                ids.append(vertex_ids[lattice])
            triangles.extend([[ids[0], ids[1], ids[2]], [ids[0], ids[2], ids[3]]])
    rooms = [{"room_id": f"room_{i+1}", "label": label,
              "polygon": [[left,0.2],[right,0.2],[right,2.2],[left,2.2]]}
             for i, (left, right, label) in enumerate(((0.2,2.0,"entrance"),
                                                     (2.2,4.2,"living_room"),
                                                     (4.4,6.2,"bedroom")))]
    return {"schema_version": "ares.scene.v1", "scene_id": "three_room", "seed": 0,
            "units": "m", "frame": "right_handed_z_up",
            "license": {"spdx": "NOASSERTION", "commercial_ok": False},
            "grid": {"resolution": 0.2, "origin": [0,0,0], "shape": list(shape),
                     "occupied": [list(c) for c in sorted(occupied)]},
            "collision_mesh": {"vertices": vertices, "triangles": triangles},
            "rooms": rooms,
            "doors": [{"door_id": f"door_{i+1}", "room_ids": [f"room_{i+1}", f"room_{i+2}"],
                       "state": "open", "voxel_indices": [list(c) for c in sorted(door_cells[x])]}
                      for i, x in enumerate((10,21))],
            "surfaces": [{"material": "normal", "voxel_indices": [list(c) for c in sorted(occupied)]}],
            "lighting": [{"room_id": r["room_id"], "state": "lit"} for r in rooms],
            "spawn": [1.1,1.1,1.1]}


def build_profile():
    values = {
        "mass_kg": (0.5, 0.45, 0.55), "max_thrust_n": (24.5, 22.0, 27.0),
        "motor_lag_s": (0.025, 0.015, 0.04), "linear_drag": (0.1, 0.05, 0.2),
        "battery_energy_j": (54000.0, 43200.0, 64800.0),
        "depth_horizontal_fov_rad": tuple(math.radians(v) for v in (87.0, 85.0, 89.0)),
        "depth_vertical_fov_rad": tuple(math.radians(v) for v in (58.0, 56.0, 60.0)), "depth_width": (32,32,32),
        "depth_height": (24,24,24), "depth_fps": (90.0,60.0,90.0),
        "depth_max_range_m": (4.0,3.0,4.0), "depth_noise_std_m": (0.01,0.005,0.025),
        "imu_accel_noise_std": (0.02,0.01,0.04), "imu_gyro_noise_std": (0.002,0.001,0.005),
        "odometry_drift_per_m": (0.01,0.0,0.03), "compute_latency_s": (0.01,0.005,0.02),
        "body_radius_m": (0.15,0.14,0.17)}
    parameters = {name: {"nominal": n, "range": [low,high]} for name, (n,low,high) in values.items()}
    parameters["inertia_kg_m2"] = {"nominal": [0.002,0.002,0.003],
                                  "range": [[0.0018,0.0022],[0.0018,0.0022],[0.0027,0.0033]]}
    return {"schema_version": "ares.profile.v1", "profile_id": "drone_nominal",
            "units": "SI", "parameters": parameters}


if __name__ == "__main__":
    for relative, data in (("scenes/three_room.json", build_scene()),
                           ("configs/drone_nominal.json", build_profile())):
        path = ROOT / "ares" / "data" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
