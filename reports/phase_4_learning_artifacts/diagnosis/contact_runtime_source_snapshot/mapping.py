"""Sparse signed-log-odds voxel occupancy mapping on CPU."""

import math
import numpy as np

from .odometry import depth_points, validate_pose


class OccupancyMapper:
    """Integrate camera Z depth with voxel DDA, stopping at each measured hit.

    Unknown cells have no evidence. Optional shape bounds the world grid; an
    unbounded sparse grid is shifted on export so report indices are nonnegative.
    No map cells behind a hit receive either free or occupied evidence.
    """

    def __init__(self, resolution=.2, origin=(0, 0, 0), shape=None, *, max_rays=None,
                 hit_log_odds=.85, miss_log_odds=-.4, clamp_log_odds=4.):
        if isinstance(resolution, bool) or not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be finite and positive")
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float)
        if self.origin.shape != (3,) or not np.isfinite(self.origin).all():
            raise ValueError("origin must contain three finite numbers")
        if shape is not None and (len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape)):
            raise ValueError("shape must contain three positive integers")
        self.shape = None if shape is None else tuple(shape)
        if max_rays is not None and (type(max_rays) is not int or max_rays < 1):
            raise ValueError("max_rays must be a positive integer or None")
        if not (math.isfinite(hit_log_odds) and hit_log_odds > 0 and
                math.isfinite(miss_log_odds) and miss_log_odds < 0 and
                math.isfinite(clamp_log_odds) and clamp_log_odds > 0):
            raise ValueError("require positive hit/clamp and negative miss log odds")
        self.max_rays = max_rays
        self.hit_log_odds, self.miss_log_odds, self.clamp_log_odds = hit_log_odds, miss_log_odds, clamp_log_odds
        self._evidence = {}
        self.frames_integrated = 0

    def _inside(self, cell):
        return self.shape is None or all(0 <= cell[i] < self.shape[i] for i in range(3))

    def _ray(self, start, end):
        """Yield cells intersecting a segment, with optional bounded-grid clipping."""
        start = (start - self.origin) / self.resolution
        end = (end - self.origin) / self.resolution
        direction = end - start
        enter, leave = 0., 1.
        if self.shape is not None:
            for axis in range(3):
                if abs(direction[axis]) < 1e-14:
                    if not 0 <= start[axis] < self.shape[axis]:
                        return
                else:
                    t0, t1 = -start[axis] / direction[axis], (self.shape[axis] - start[axis]) / direction[axis]
                    enter, leave = max(enter, min(t0, t1)), min(leave, max(t0, t1))
            if enter >= leave or leave < 0 or enter > 1:
                return
        # Nudge into clipped region; endpoint itself remains exact for hit indexing.
        first = start + direction * min(leave, enter + 1e-10)
        last = start + direction * max(enter, leave - 1e-10)
        cell, last_cell = np.floor(first).astype(np.int64), np.floor(last).astype(np.int64)
        step = np.sign(direction).astype(np.int64)
        delta = np.full(3, np.inf)
        crossing = np.full(3, np.inf)
        for axis in range(3):
            if step[axis]:
                delta[axis] = abs(1 / direction[axis])
                edge = cell[axis] + (1 if step[axis] > 0 else 0)
                crossing[axis] = (edge - start[axis]) / direction[axis]
        # A segment crosses at most the Manhattan distance + 1 distinct voxels.
        limit = int(np.abs(last_cell - cell).sum()) + 4
        for _ in range(limit):
            value = tuple(int(v) for v in cell)
            if self._inside(value):
                yield value
            if np.array_equal(cell, last_cell):
                break
            next_t = float(np.min(crossing))
            if next_t > leave + 1e-10:
                break
            # Advance every tied axis: cells touched only at an edge have no volume.
            axes = np.isclose(crossing, next_t, atol=1e-12, rtol=0)
            cell[axes] += step[axes]
            crossing[axes] += delta[axes]

    def integrate(self, depth_m, camera, pose):
        """Integrate an accepted camera-to-world pose; callers exclude lost poses."""
        transform = validate_pose(pose)
        points = depth_points(depth_m, camera).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        valid_count = len(points)
        if self.max_rays is not None and len(points) > self.max_rays:
            points = points[np.linspace(0, len(points) - 1, self.max_rays, dtype=int)]
        points = points @ transform[:3, :3].T + transform[:3, 3]
        start = transform[:3, 3]
        free, occupied = set(), set()
        for endpoint in points:
            hit = tuple(int(v) for v in np.floor((endpoint - self.origin) / self.resolution))
            cells = list(self._ray(start, endpoint))
            if self._inside(hit):
                occupied.add(hit)
            free.update(cell for cell in cells if cell != hit)
        # An observed surface wins over another ray's free evidence in this frame.
        free -= occupied
        for cells, update in ((free, self.miss_log_odds), (occupied, self.hit_log_odds)):
            for cell in cells:
                self._evidence[cell] = float(np.clip(self._evidence.get(cell, 0.) + update,
                                                     -self.clamp_log_odds, self.clamp_log_odds))
        self.frames_integrated += 1
        return {"valid_depth_pixels": valid_count, "rays_integrated": len(points),
                "free_updates": len(free), "occupied_updates": len(occupied), **self.counts()}

    def counts(self):
        free = sum(value < -1e-12 for value in self._evidence.values())
        occupied = sum(value > 1e-12 for value in self._evidence.values())
        result = {"free_cells": free, "occupied_cells": occupied, "observed_cells": free + occupied,
                  "frames_integrated": self.frames_integrated}
        if self.shape is not None:
            result["unknown_cells"] = math.prod(self.shape) - free - occupied
        return result

    def export(self):
        """Return a report-v1-compatible voxel_map (no oracle-filled unknowns)."""
        known = {cell: (2 if value > 0 else 1) for cell, value in self._evidence.items() if abs(value) > 1e-12}
        offset = np.zeros(3, dtype=np.int64)
        if self.shape is not None:
            shape = list(self.shape)
        elif known:
            indices = np.asarray(list(known), dtype=np.int64)
            offset = indices.min(axis=0)
            shape = (indices.max(axis=0) - offset + 1).tolist()
        else:
            shape = [1, 1, 1]
        origin = self.origin + offset * self.resolution
        return {"resolution": self.resolution, "origin": origin.tolist(), "shape": shape,
                "cells": [[int(cell[i] - offset[i]) for i in range(3)] + [label]
                          for cell, label in sorted(known.items())]}
