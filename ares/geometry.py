"""Exact voxel-union surface meshes, with coplanar faces merged into rectangles."""

from collections import defaultdict
import math

from .contracts import ContractError


def voxel_mesh(occupied, resolution=0.2, origin=(0, 0, 0)):
    if (type(resolution) not in (int, float) or not math.isfinite(resolution) or resolution <= 0
            or not isinstance(origin, (tuple, list)) or len(origin) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in origin)):
        raise ContractError("mesh: resolution must be positive and origin finite XYZ")
    voxels = set()
    for cell in occupied:
        if (not isinstance(cell, (tuple, list)) or len(cell) != 3
                or any(type(v) is not int or v < 0 for v in cell)):
            raise ContractError("mesh: occupied indices must be nonnegative integer triples")
        voxels.add(tuple(cell))
    planes = defaultdict(set)
    for cell in voxels:
        for axis in range(3):
            uv = tuple(cell[i] for i in range(3) if i != axis)
            for sign in (-1, 1):
                neighbor = list(cell)
                neighbor[axis] += sign
                if tuple(neighbor) not in voxels:
                    planes[(axis, sign, cell[axis] + (sign == 1))].add(uv)
    vertices, triangles, ids = [], [], {}
    for (axis, sign, plane), cells in sorted(planes.items()):
        axes = [i for i in range(3) if i != axis]
        for u, v in sorted(cells):
            if (u, v) not in cells:
                continue
            width = 1
            while (u + width, v) in cells:
                width += 1
            height = 1
            while all((x, v + height) in cells for x in range(u, u + width)):
                height += 1
            for x in range(u, u + width):
                for y in range(v, v + height):
                    cells.remove((x, y))
            corners = [(u, v), (u + width, v), (u + width, v + height), (u, v + height)]
            if sign != (-1 if axis == 1 else 1):
                corners.reverse()
            face = []
            for a, b in corners:
                lattice = [0, 0, 0]
                lattice[axis], lattice[axes[0]], lattice[axes[1]] = plane, a, b
                key = tuple(lattice)
                if key not in ids:
                    ids[key] = len(vertices)
                    try:
                        position = [origin[i] + key[i] * resolution for i in range(3)]
                    except OverflowError as exc:
                        raise ContractError("mesh: voxel coordinates cannot be represented") from exc
                    if not all(math.isfinite(v) for v in position):
                        raise ContractError("mesh: voxel coordinates must be finite")
                    vertices.append(position)
                face.append(ids[key])
            triangles.extend([[face[0], face[1], face[2]], [face[0], face[2], face[3]]])
    if len({tuple(v) for v in vertices}) != len(vertices):
        raise ContractError("mesh: resolution/origin cannot represent distinct voxel corners")
    return {"vertices": vertices, "triangles": triangles}
