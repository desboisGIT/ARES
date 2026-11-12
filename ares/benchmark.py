"""Paired benchmark against the frozen Phase 0 axial-ray implementation.

The reference normalizes machine speed in hosted CI. This measures axial scans,
not GPU performance, full mission latency, or a statistically universal bound.
"""

import gc
import math
import random
import statistics
import time


def reference_axial_scan(world, cell):
    """Frozen axial-v1 reference; changing it requires a new approved baseline."""
    if not world.in_grid(cell) or cell in world.occupied:
        return {}
    result = {cell: 1}
    for direction in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        for distance in range(1, math.floor(world.sensor_range / world.resolution) + 1):
            nxt = tuple(cell[i] + direction[i] * distance for i in range(3))
            if not world.in_grid(nxt):
                break
            result[nxt] = 2 if nxt in world.occupied else 1
            if nxt in world.occupied:
                break
    return result


def run_benchmark(world, baseline, *, repeats=11, samples=1200):
    cells = sorted(world.reachable)
    rng = random.Random(1741)  # Explicit, versioned benchmark seed.
    workload = [cells[rng.randrange(len(cells))] for _ in range(samples)]
    for cell in set(workload):
        if world.axial_scan(cell) != reference_axial_scan(world, cell):
            raise ValueError("benchmark reference semantics diverged")

    def timed(scanner):
        start = time.perf_counter()
        for cell in workload:
            scanner(cell)
        return time.perf_counter() - start

    current, reference, ratios = [], [], []
    active = gc.isenabled()
    gc.disable()
    try:
        for iteration in range(repeats):
            if iteration % 2:
                ref, cur = timed(lambda c: reference_axial_scan(world, c)), timed(world.axial_scan)
            else:
                cur, ref = timed(world.axial_scan), timed(lambda c: reference_axial_scan(world, c))
            current.append(cur)
            reference.append(ref)
            ratios.append(cur / ref)
    finally:
        if active:
            gc.enable()
    ratio = statistics.median(ratios)
    limit = baseline["normalized_ratio"] * 1.10
    return {"schema_version": "ares.benchmark.v1", "workload": "axial-v1", "seed": 1741,
            "samples_per_repeat": samples, "repeats": repeats,
            "median_seconds": statistics.median(current),
            "reference_median_seconds": statistics.median(reference),
            "normalized_ratio": ratio, "baseline_ratio": baseline["normalized_ratio"],
            "slowdown_fraction": ratio / baseline["normalized_ratio"] - 1,
            "maximum_slowdown_fraction": 0.10, "passed": ratio <= limit}
