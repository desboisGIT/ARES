"""Host-local synchronous callable timing; never evidence of onboard performance."""
from __future__ import annotations

import math
import time
from collections.abc import Callable


def measure_callable(fn: Callable[[], object], *, warmup: int = 5,
                     iterations: int = 100, target_hz: float | None = None,
                     clock: Callable[[], float] = time.perf_counter) -> dict:
    """Measure completed calls. GPU callers must synchronize inside ``fn``.

    Percentiles use linear interpolation between ordered observations. Throughput
    is completed calls / total measured execution time, excluding harness gaps.
    Target passes only when p99 execution time fits the target period.
    """
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if target_hz is not None and (not math.isfinite(target_hz) or target_hz <= 0):
        raise ValueError("target_hz must be finite and positive")
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iterations):
        start = clock()
        fn()
        elapsed = clock() - start
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("clock must produce finite positive durations")
        samples.append(elapsed)
    ordered = sorted(samples)
    def percentile(q):
        index = (len(ordered) - 1) * q
        lo = int(index)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered)-1)] - ordered[lo]) * (index-lo)
    result = {"iterations": iterations, "warmup": warmup,
              "median_s": percentile(.5), "p90_s": percentile(.9),
              "p99_s": percentile(.99), "hz": iterations / sum(samples)}
    if target_hz is not None:
        result.update(target_hz=target_hz, meets_target=result["p99_s"] <= 1 / target_hz)
    return result
