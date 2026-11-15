"""Seeded, constructive indoor layouts with exact voxel/mesh collision geometry.

Generated geometry is synthetic, not an architectural or building-code model.
All dimensions lie on the frozen 20 cm lattice. Layouts are constructive;
geometry-hash rejection sampling assigns disjoint dataset partitions.
"""

from collections import defaultdict
import hashlib
import json

from .geometry import voxel_mesh
from .randomness import seeded_rng

GENERATOR_VERSION = "ares.worldgen.v1"
KINDS = ("single_room", "apartment", "floor")
SPLITS = ("train", "validation", "test")
RESOLUTION = 0.2


def _partition(rect, count, rng):
    leaves = [rect]
    while len(leaves) < count:
        choices = [(i, r) for i, r in enumerate(leaves) if max(r[2] - r[0], r[3] - r[1]) >= 25]
        if not choices:
            break
        i, r = max(choices, key=lambda pair: (pair[1][2] - pair[1][0]) * (pair[1][3] - pair[1][1]))
        x0, y0, x1, y1 = r
        axis = 0 if x1 - x0 >= y1 - y0 else 1
        if min(x1 - x0, y1 - y0) >= 25 and rng.random() < 0.3:
            axis = 1 - axis
        low, high = (x0, x1) if axis == 0 else (y0, y1)
        cut = rng.randint(low + 12, high - 13)
        parts = ((x0, y0, cut, y1), (cut + 1, y0, x1, y1)) if axis == 0 else (
            (x0, y0, x1, cut), (x0, cut + 1, x1, y1))
        leaves[i:i + 1] = parts
    return leaves


def _layout(kind, rng):
    if kind == "single_room":
        nx, ny = rng.randint(17, 30), rng.randint(17, 27)
        return nx, ny, [(1, 1, nx - 1, ny - 1)], [rng.choice(("bedroom", "living_room", "study"))]
    if kind == "apartment":
        nx, ny = rng.randint(36, 52), rng.randint(30, 44)
        rects = _partition((1, 1, nx - 1, ny - 1), rng.randint(4, 7), rng)
        labels = (["entrance", "living_kitchen", "bedroom", "bathroom"] if len(rects) == 4 else
                  ["entrance", "living_room", "kitchen", "bedroom", "bathroom", "bedroom", "study"][:len(rects)])
        return nx, ny, rects, labels
    segments = rng.randint(2, 4)
    widths = [rng.randint(21, 27) for _ in range(segments)]
    depth = rng.randint(28, 34)
    hall = rng.randint(8, 10)
    nx, ny = sum(widths) + segments + 1, 2 * depth + hall + 4
    hall_y = depth + 2
    rects, labels = [(1, hall_y, nx - 1, hall_y + hall)], ["corridor"]
    x = 1
    for i, width in enumerate(widths):
        for side in (0, 1):
            front = rng.randint(12, 15)
            bath = rng.randint(7, 9)
            if side == 0:
                rear_y0, rear_y1 = 1, depth - front
                living_y0, living_y1 = rear_y1 + 1, depth + 1
            else:
                living_y0 = hall_y + hall + 1
                living_y1 = living_y0 + front
                rear_y0, rear_y1 = living_y1 + 1, ny - 1
            rects.extend([(x, living_y0, x + width, living_y1),
                          (x, rear_y0, x + bath, rear_y1),
                          (x + bath + 1, rear_y0, x + width, rear_y1)])
            labels.extend([f"unit_{2 * i + side + 1}/living_kitchen",
                           f"unit_{2 * i + side + 1}/bathroom", f"unit_{2 * i + side + 1}/bedroom"])
        x += width + 1
    return nx, ny, rects, labels


def _adjacencies(rects):
    edges = []
    for a, first in enumerate(rects):
        for b in range(a + 1, len(rects)):
            other = rects[b]
            for axis in (0, 1):
                for left, right in ((first, other), (other, first)):
                    if left[axis + 2] + 1 != right[axis]:
                        continue
                    orthogonal = 1 - axis
                    low = max(left[orthogonal], right[orthogonal])
                    high = min(left[orthogonal + 2], right[orthogonal + 2])
                    if high - low >= 5:
                        edges.append((a, b, axis, left[axis + 2], low, high))
    return edges


def _candidate(seed, kind, split, attempt):
    rng = seeded_rng(seed, f"{GENERATOR_VERSION}:{kind}:{split}:{attempt}")
    nx, ny, rects, labels = _layout(kind, rng)
    nz = rng.randint(14, 17)
    rooms = [{"room_id": f"room_{i + 1}", "label": label,
              "polygon": [[round(x0 * .2, 10), round(y0 * .2, 10)],
                          [round(x1 * .2, 10), round(y0 * .2, 10)],
                          [round(x1 * .2, 10), round(y1 * .2, 10)],
                          [round(x0 * .2, 10), round(y1 * .2, 10)]]}
             for i, ((x0, y0, x1, y1), label) in enumerate(zip(rects, labels))]
    free_xy = {(x, y) for x0, y0, x1, y1 in rects for x in range(x0, x1) for y in range(y0, y1)}
    occupied = {(x, y, z) for x in range(nx) for y in range(ny)
                for z in ((0, nz - 1) if (x, y) in free_xy else range(nz))}
    # Choose a randomized spanning tree before adding some extra links. The
    # graph remains connected as architecture even when leaf doors are closed.
    edges = _adjacencies(rects)
    if kind == "floor":
        # Separate apartments open onto common circulation, never directly
        # into another unit's bathroom or bedroom through a party wall.
        edges = [edge for edge in edges if
                 labels[edge[0]].split("/")[0] == labels[edge[1]].split("/")[0]
                 or ("corridor" in (labels[edge[0]], labels[edge[1]])
                     and any("living_kitchen" in labels[i] for i in edge[:2]))]
    rng.shuffle(edges)
    parent = list(range(len(rects)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    selected = []
    for edge in edges:
        a, b = edge[:2]
        if find(a) != find(b):
            parent[find(a)] = find(b)
            selected.append(edge)
        elif rng.random() < .18:
            selected.append(edge)
    if len({find(i) for i in range(len(rects))}) != 1:
        raise RuntimeError("constructive layout has disconnected room adjacency")
    doors, reserved = [], set()
    for edge in selected:
        a, b, axis, plane, low, high = edge
        start = rng.randint(low, high - 5)
        indices = [[plane if axis == 0 else t, t if axis == 0 else plane, z]
                   for t in range(start, start + 5) for z in range(1, 11)]
        # Keep the building's main circulation open; occasional room doors
        # remain closed and are intentionally excluded by reachable scoring.
        state = "closed" if rng.random() < .15 and "corridor" not in (labels[a], labels[b]) else "open"
        if state == "open":
            occupied.difference_update(map(tuple, indices))
        doors.append({"door_id": f"door_{len(doors) + 1}", "room_ids": [f"room_{a + 1}", f"room_{b + 1}"],
                      "state": state, "voxel_indices": indices})
        center = (plane, start + 2) if axis == 0 else (start + 2, plane)
        for room_index in (a, b):
            x0, y0, x1, y1 = rects[room_index]
            middle = ((x0 + x1) // 2, (y0 + y1) // 2)
            # A one-metre-wide orthogonal route from every door to room center.
            for x in range(min(center[0], middle[0]), max(center[0], middle[0]) + 1):
                for y in range(center[1] - 2, center[1] + 3):
                    reserved.add((x, y))
            for y in range(min(center[1], middle[1]), max(center[1], middle[1]) + 1):
                for x in range(middle[0] - 2, middle[0] + 3):
                    reserved.add((x, y))
            # Reserve the full doorway apron even for a turn near a corner.
            for t in range(start, start + 5):
                for distance in (-2, -1, 0, 1, 2):
                    reserved.add((plane + distance, t) if axis == 0 else (t, plane + distance))
    first = rects[0]
    spawn_cell = ((first[0] + first[2]) // 2, (first[1] + first[3]) // 2, 5)
    reserved.update((x, y) for x in range(spawn_cell[0] - 3, spawn_cell[0] + 4)
                    for y in range(spawn_cell[1] - 3, spawn_cell[1] + 4))
    material = {}
    for rect, label in zip(rects, labels):
        if label == "corridor":
            continue
        x0, y0, x1, y1 = rect
        style = label.rsplit("/", 1)[-1]
        pieces = ([(4, 4, 4), (4, 4, 8)] if style == "bathroom" else
                  [(8, 10, 3), (3, 7, 10)] if style == "bedroom" else
                  [(9, 4, 4), (4, 5, 4), (3, 8, 5)])
        for width, depth, height in pieces:
            candidates = []
            for w, d in ((width, depth), (depth, width)):
                if w <= x1 - x0 - 1 and d <= y1 - y0 - 1:
                    candidates.extend((x, y, w, d) for x in (x0, x1 - w) for y in (y0, y1 - d))
            rng.shuffle(candidates)
            for x, y, w, d in candidates:
                footprint = {(a, b) for a in range(x, x + w) for b in range(y, y + d)}
                if footprint & reserved or any((a, b, 1) in occupied for a, b in footprint):
                    continue
                cells = {(a, b, z) for a, b in footprint for z in range(1, min(height + 1, nz - 1))}
                occupied.update(cells)
                finish = rng.choice(("normal", "normal", "black", "shiny"))
                material.update((cell, finish) for cell in cells)
                break
    # Surface patches keep glass/mirror as physical obstacles; lighting and
    # optical failure response are consumed by later depth simulation.
    for finish in ("glass", "mirror", "black", "shiny"):
        candidates = sorted(cell for cell in occupied if cell[2] in range(4, 10)
                            and (cell[0] in (0, nx - 1) or cell[1] in (0, ny - 1)))
        anchor = rng.choice(candidates)
        for cell in candidates:
            if abs(cell[0] - anchor[0]) <= 2 and abs(cell[1] - anchor[1]) <= 2:
                material[cell] = finish
    surfaces = defaultdict(list)
    for cell in sorted(occupied):
        surfaces[material.get(cell, "normal")].append(list(cell))
    return {"schema_version": "ares.scene.v1", "scene_id": f"world-v1:{split}:{kind}:{seed}",
            "seed": seed, "units": "m", "frame": "right_handed_z_up",
            "license": {"spdx": "NOASSERTION", "commercial_ok": False},
            "grid": {"resolution": .2, "origin": [0, 0, 0], "shape": [nx, ny, nz],
                     "occupied": [list(cell) for cell in sorted(occupied)]},
            "collision_mesh": {"vertices": [], "triangles": []}, "rooms": rooms, "doors": doors,
            "surfaces": [{"material": finish, "voxel_indices": cells} for finish, cells in sorted(surfaces.items())],
            "lighting": [{"room_id": room["room_id"], "state": "dark" if rng.random() < .25 else "lit"} for room in rooms],
            "spawn": [round((index + .5) * .2, 10) for index in spawn_cell]}


def geometry_digest(scene):
    """Ignore lighting/material changes when partitioning identical geometry."""
    return hashlib.sha256(json.dumps(scene["grid"], sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def generate_scene(seed, kind="apartment", split="train"):
    if type(seed) is not int or kind not in KINDS or split not in SPLITS:
        raise ValueError("seed must be integer; kind single_room/apartment/floor; split train/validation/test")
    for attempt in range(512):
        scene = _candidate(seed, kind, split, attempt)
        bucket = int(geometry_digest(scene)[:8], 16) % 10
        assigned = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
        if assigned == split:
            scene["collision_mesh"] = voxel_mesh(scene["grid"]["occupied"])
            return scene
    raise RuntimeError("geometry partition rejection limit reached; use a different explicit seed")
