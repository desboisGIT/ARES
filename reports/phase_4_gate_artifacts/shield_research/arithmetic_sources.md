# CPU Float32 arithmetic source evidence

Research notes, 2026-11-13. This is not a numerical certificate. It concerns the
full DiffAeroAdapter plant in eager CPU mode, not NumPy, GPU, compiled dynamics,
the backup controller, or the entire closed loop. No package or environment was
modified and no GPU job was run for this lookup.

## Actual installation and host

The checkout interpreter reports `torch 2.11.0+cu130`, source commit
`70d99e998b4955e0049d13a98d77ae1b14db1f45`, GCC 13.3, Release `-O2`, MKL
2024.2 (build 20240605), and AVX512 CPU dispatch. The host is x86-64 AMD Ryzen
7 7700X, glibc 2.43. Installed headers are under
`.venv/lib/python3.14/site-packages/torch/include/`.
`torch.__config__.show()` does not list fast-math, but that is not a per-object
compiler/assembly audit. CPU capability is a dispatch capability, not evidence
that every operator executes AVX512 instructions.

A fresh read-only CPU probe reported `fegetround() == 0` (the host's nearest
rounding constant) and preserved the smallest binary32 subnormal under a tensor
multiply by one. This is an observation, not an MXCSR or per-worker-thread proof.
No `torch.set_flush_denormal` call was made. FTZ and DAZ are separate processor
controls and require explicit treatment. [Intel FTZ/DAZ documentation](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-0/set-the-ftz-and-daz-flags.html)

## Plant paths and relevant implementations

`ares/diffaero_adapter.py:112` computes motor decay with
`torch.exp(-integration_dt / motor_lag)`. Reset normalizes a four-component
quaternion with `torch.linalg.vector_norm(..., dim=-1, keepdim=True)`;
`ares/vendor/diffaero/quadrotor.py:38` repeats that normalization after RK4.
Other plant operations include divisions, cross products and matrix products;
covering exp/sqrt alone does not bound the entire plant.

| Primitive/path | Source evidence | Consequence |
| --- | --- | --- |
| Float32 eager CPU `exp` | `UnaryOpsKernel.cpp` registers the AVX512-capable exp kernel and calls `vml::vexp`; strided data are packed when needed. The installed `ATen/cpu/vml.h` contains the float MKL specialization `vmsExp(..., VML_HA \| VML_FTZDAZ_OFF \| VML_ERRMODE_IGNORE)` when MKL is enabled. | This installation's source-selected path is MKL HA, not merely SLEEF. Resolve the actual binary dispatch and bundled MKL implementation before adopting a bound. |
| Non-MKL vector exp alternative | Installed `ATen/cpu/vec/vec512/vec512_float.h:307` calls `Sleef_expf16_u10`. | Relevant to a different build/fallback, not a replacement justification for current MKL execution. |
| Last-dimension L2 norm of four Float32 values | `ReduceOpsKernel.cpp:223–253` selects the last-dimension path when input/output dtypes match and the reduced dimension is contiguous. Accumulator is Float32. AVX512 float width is 16, so length four uses the scalar tail `buffer[0] += data_val * data_val`, followed by `std::sqrt(buffer[0])`. | It is a sum of squares followed by sqrt, not a correctly rounded exact four-vector norm. Row stride may be larger than four without invalidating inner contiguity. Verify the actual tensor strides and kernel dispatch. |
| Other norm shapes/strides | `NormTwoOps` reduces using `acc + data * data`, combines with addition and projects with `std::sqrt` on CPU. | A different reduction layout can change association and accumulation path. Do not generalize the length-four tail trace to arbitrary shapes. |

These paths are visible in the matching pinned primary sources:
[UnaryOpsKernel.cpp](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp),
[vml.h](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/cpu/vml.h),
[ReduceOpsKernel.cpp](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp),
[Reduce.h](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cpu/Reduce.h),
[SharedReduceOps.h](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/SharedReduceOps.h), and
[vec512_float.h](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/cpu/vec/vec512/vec512_float.h).
The first two kernel `.cpp` files were fetched from that exact commit into
`/tmp/ares-torch-arithmetic/`; reduction/vector headers were also inspected locally.

## Documented bounds and a conditional budget

Intel documents HA's design requirement as less than 1 ULP, with processor/release
variation, and notes that function-specific deviations must be checked. Treat
**1 output ULP for MKL HA exp** as a candidate supplier bound, contingent on the
actual 2024.2 CPU/implementation, its function-specific exceptions and the
binary dispatch. It is not a proof derived from observed agreement.
[Intel VM accuracy overview](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-vector-math-performance-accuracy-data/2021-1/overview.html),
[matching 2024.2 reference](https://cdrdv2-public.intel.com/823881/onemkl_developer-reference-c_2024.2-766684-823881.pdf)

SLEEF documents 1.0 ULP for its `exp[f]_u10` family. Its separate sqrt `u05`
functions have a different bound and are not the `std::sqrt` in this norm path.
A **1 output ULP SLEEF exp** budget is therefore only conditional on using the
matching SLEEF implementation, supported rounding/environment and domain.
[SLEEF exp documentation](https://sleef.org/purec.xhtml#Sleef_exp_u10),
[SLEEF sqrt documentation](https://sleef.org/purec.xhtml#Sleef_sqrt_u05)

The GNU C Library distinguishes sqrt/fma, whose results follow IEEE operations,
from general libm functions for which correctly rounded results are not its aim.
For a confirmed IEEE nearest-rounded sqrt implementation, finite nonnegative
input and valid floating-point environment, a candidate local rounding budget is
**0.5 ULP for sqrt**. That does not cover error already accumulated in its input.
Do not give scalar `expf` the same correctly rounded guarantee.
[GNU libm accuracy policy](https://sourceware.org/glibc/manual/2.32/html_node/Errors-in-Math-Functions.html)

For normal, finite intermediates under nearest rounding, the usual binary32
model uses `u = 2^-24`. A deliberately conservative source-level four-element
sum-of-squares budget can count four products and four additions and use
`gamma_8 = 8u/(1-8u)` relative to the exact nonnegative sum, followed by the sqrt
rounding budget. This is a proposed derivation under assumptions, not an audited
bound for the installed binary. FMA contraction, reassociation or another
reduction tree needs either an explicitly covering model or exact disassembly.
Subnormal cases require absolute error terms (and explicit FTZ/DAZ behavior),
not a normal-only relative model. Interval evaluation must propagate the
argument division error into exp and propagate sum uncertainty through sqrt.
Use outward ULP spacing bounds over the whole operand interval, including binade
boundaries; do not multiply a sample's ULP by a constant and call it uniform.

## Unclosed obligations

- Confirm kernel/binary symbols and exact MKL 2024.2 exp guarantee on this AMD
  dispatch, including small vector lengths and every reachable decay argument.
- Audit per-object flags, FMA contraction, reassociation and scalar sqrt lowering;
  source inspection alone does not establish generated instructions.
- Bound all reachable intermediate values, exclude or model overflow, underflow,
  NaNs, infinities, zero norm and clamp branch boundaries. Prove relevant tensor
  shape/stride paths and rounding controls for every worker thread.
- Cover scalar arithmetic, RK4, quaternion algebra and small matrix products
  (including their actual BLAS/CPU paths) in the full plant error enclosure.
- Keep backup command numerics separate. No assertion here covers TorchInductor,
  `torch.compile`, CUDA exp/sqrt, GPU reductions, TF32 or learned network execution.

PyTorch explicitly warns that equivalent computations, batching and backends can
produce different floating-point results. Neither repeatable tests nor this
source lookup supplies a universal numerical or safety certificate.
[PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
