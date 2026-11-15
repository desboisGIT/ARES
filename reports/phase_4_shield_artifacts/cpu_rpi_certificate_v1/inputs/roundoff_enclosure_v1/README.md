# Conditional Float32 plant arithmetic enclosure

This research artifact bounds one H=0.01 s step of the ORIGINAL eager CPU adapter and 13D vendor RK4 program, including five h=0.002 s substeps, motor lag and the fractional two-slot SI command queue. It changes no ARES package source. It supplies an additive arithmetic disturbance for a separately proved REAL model; it is not a flight certificate or an RPI admission test.

`conditional_projective21_budget.json` gives outward rational ceilings in order `(p3, v3, pi(raw_q)_xyz3, w3, Fmotor, current_queue4, old_queue4)`:

| Components | Absolute error upper bound per H |
|---|---:|
| position x/y/z | 5e-6 m each |
| velocity x/y; z | 6.5e-8; 5.3e-7 m/s |
| projected quaternion x/y; z | 1.2e-7; 1.7e-8 |
| angular velocity x/y; z | 1e-6; 2e-7 rad/s |
| applied motor thrust | 7e-6 N |
| eight queue copy components | zero |

The reference quaternion is the exact projective normalization `pi(raw_q)=raw_q/||raw_q||` in its positive-w hemisphere. Input Float32 quaternions need only have norm in `[1-8u,1+8u]`, u=2^-24, and the specified projective xyz box. Output error is again measured after exact projective normalization. The arithmetic analysis executes the vendor's ORIGINAL optimized quaternion multiplication and homogeneous rotations. It does not substitute the factored REAL research graph.

The current SI action is an already stored binary32 word. Its separately verified public controller point-law error (at most 1e-6 N or rad/s) must be propagated once through the REAL command channel Gamma. It is not added again to queue copies here. The new command deviations are bounded by `[.116,.224,.224,.0243]`; this includes that quantization allowance. Exact entry ranges and full Float32-expanded physical parameter ranges are in `conditional_budgets.json`, under the verified-decay, stored-word, world-position-16 row. Motor, two history words, projective orientation, velocity and rates must also satisfy these ranges. World position is bounded componentwise by 16 m; using only a center-6-m position bound would undercount position addition roundoff elsewhere.

## Arithmetic construction

`ScalarRoundInterval` stores an outward REAL interval and an absolute actual-binary32 error. Error propagation uses exact Fraction arithmetic for bound decisions and outward binary64 endpoints. Basic arithmetic assumes full-precision binary32 round-to-nearest-even and adds `u*max_abs_exact_actual_result + 2^-126`. The latter is a conservative output-flush allowance; the runnable worker actually requires FTZ and DAZ disabled. Denominators, overflow, quaternion normalization clamp and actuator branches are checked over the whole interval domain.

Short dot products and sums of squares use `gamma_(2n)*sum_abs_products + 4n*2^-126`, n=3 or 4; this encloses ordinary product/add association and FMA association. Sqrt assumes at most 0.5 ULP error. The motor exp has two explicit options: an unaudited candidate native-exp allowance (one extra ULP beyond RNE), and the selected exact verified-decay backend (zero extra ULP). The published ceiling uses only the latter. Its exact alternating-series witness establishes the correct rounded binary32 exp of the represented argument; input argument casts/division remain in this analysis.

All h, dt, g, nominal inertia, mass and parameter conversions and the rounded inertia reciprocal are included. The original stable motor expression is bounded as the SAME three operations, but its correlated input errors use the exact convex combination rather than losing old/target cancellation. Delay-floor crossings use piecewise-linear history interpolation: the adjacent-command slope times the delay quotient error bounds either branch. Normalization's four-square sum, sqrt and divisions imply a rational norm-shell bound strictly below 8u on this domain. Unit-quaternion REAL identities tighten range and positive-w error only after normalization; no exact-unit assumption is made about binary32 inputs or RK4 stages.

`run_roundoff.py` computes the full forward enclosure. `publish_budget.py` uses exact rational comparisons to verify every selected component fits the displayed decimal ceiling. No finite simulation data fit or define these constants.

## Runnable guarded worker

Run from the repository with `.venv/bin/python`; put this directory on `sys.path`:

```python
import sys
sys.path.insert(0, 'reports/phase_4_gate_artifacts/shield_research/roundoff_enclosure_v1')
import torch
from ares.phase2 import default_profile
from conditional_cpu_backend import ConditionalCPUBackend, GuardFailure
worker = ConditionalCPUBackend(default_profile(), delay_s=.0125)
result = worker.step(torch.tensor([[4.903325, 0., 0., 0.]], dtype=torch.float32))
assert result['flight_certificate'] is False
```

Use a dedicated process. Construction selects Torch threads=1, highest Float32 matmul and disables oneDNN, then verifies the 51 frozen original Python source hashes. The worker executes the verified-decay subclass and original full-body dynamics, with declared causal delay and represented battery authority in `[.95,1]`. It never initializes a controller estimate from sampled weight; its nominal history seed is 4.903325 N. Each step checks current-thread MXCSR RNE/DAZ/FTZ, C rounding mode, eager CPU Float32/no autocast, unchanged plant/controller words, exact input domains, q shell and world position. It rejects nonfinite or malformed commands. It disables the oneDNN heuristic preceding tiny bmm, selecting the source's ordinary three-product path under the stated build/source-match assumption.

A pre-step failure latches blocked and leaves state unadvanced. A post-step domain failure also latches blocked and reports `GuardFailure.state_advanced=True`; the simulated step has already occurred. Reset must establish valid domains before further calls. These guards are numerical program preconditions, not evidence that an external observed state contains the physical state or that a trajectory is collision free. In-place hostile mutation or concurrent mode changes are outside the cooperative dedicated-worker contract.

## Installed CPU primitive audit scope

The pinned upstream source matches installed `torch.version.git_version` 70d99e998b4955e0049d13a98d77ae1b14db1f45. Downloaded official source hashes and URLs are in `sources/index.json`:

- [PowKernel.cpp](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cpu/PowKernel.cpp): exponent 2 uses multiplication, avoiding libm pow.
- [ReduceOpsKernel.cpp](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp) and [SharedReduceOps.h](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/SharedReduceOps.h): Float32 last-dimension norms use Float32 squares/additions and `std::sqrt`; dimensions 3/4 fit the short-sum budget.
- [LinearAlgebra.cpp](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/LinearAlgebra.cpp): after disabling oneDNN, size 3x3 times 3x1 meets the less-than-400 scalar bmm threshold and accumulates three Float32 products.
- [vec512_float.h](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/cpu/vec/vec512/vec512_float.h): basic Float32 vector operations use AVX512 add/sub/mul/div/sqrt and FMA intrinsics.

These source facts narrow the remaining contracts. They are not a proof that this installed wheel's binary was built from those exact sources with the expected dispatch and compiler options. Still required are that source/build/dispatch correspondence, full-precision IEEE binary32 basic instruction behavior, correct-rounded sqrt lowering (`std::sqrt` or AVX/SSE sqrt), short-reduction bounds on the actual strides/dispatch, and absence of concurrent arithmetic-mode changes. RNE/DAZ/FTZ, threads, dtype, oneDNN and eager settings are checked at runtime. Exp and atan library guarantees are unnecessary for the selected verified-decay and separately verified point-law backends. No GPU, compiled kernel, general BLAS, or reduced-precision route is covered. These are explicit conditional architecture/software contracts, not a demand for a whole-chip physical proof.

## Verification and limits

Twelve focused tests pass: six software-RNE arithmetic/association/shell/domain tests and six fail-closed CPU worker tests. `finite_cpu_audit.json` records 128 actual original full-body CPU steps compared with an original Float64 reference; all 21 errors fit the analytic ceiling and all input/output q shells pass exact checks. These finite checks are regression evidence, not a universal proof of binary implementation behavior. `platform_contract_audited` and `flight_certificate` remain false. External hard state/map suppliers, REAL nonlinear RPI/coupling, between-step swept collision coverage and runtime admission are separate obligations.
