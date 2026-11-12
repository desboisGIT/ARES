# Phase 0 benchmark policy

`python3 -m ares benchmark --check` measures 1,200 explicit-seed axial scans per
repeat, in 11 paired repeats. The frozen `reference_axial_scan` implements the
initial scanner. Each pair alternates timing order and compares equivalent
outputs before timing. Garbage collection is restored after timing.

The median of paired current/reference ratios is compared to the packaged
`ares/data/benchmarks/axial_v1.json` baseline. A ratio exceeding baseline × 1.10
fails CI. Paired ratios normalize hardware speed on shared runners but can still
experience scheduling noise; rerun a noisy failure before drawing conclusions.
No automatic baseline refresh or silent exemption is permitted. A justified
baseline change must explain the workload/version, measured regression, and
reason in the code-review description. Intentional reference changes require a
new version and baseline; they must not hide a regression.

`reports/benchmark.json` records measured seconds, ratio, slowdown, seed, repeat
count, and check outcome separately from deterministic mission artifacts.
This covers one CPU primitive; later phases must add world-generation, simulation,
mapper, model, and full-stack latency benchmarks on their relevant hardware.
