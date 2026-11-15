# Third-party source notices

ARES includes modified DiffAero dynamics source under BSD-3-Clause. Its complete
license and provenance are retained in `ares/vendor/diffaero/LICENSE` and
`ares/vendor/diffaero/UPSTREAM.md`. The NVIDIA quaternion subset retains its
copyright and BSD conditions in `ares/vendor/diffaero/math.py`.

The frozen arithmetic audit includes copies of PyTorch source files:
`LinearAlgebra.cpp`, `PowKernel.cpp`, `ReduceOpsKernel.cpp`,
`SharedReduceOps.h`, and `vec512_float.h`. These also appear inside the portable
verified-safety ZIP. They come from PyTorch commit
`70d99e998b4955e0049d13a98d77ae1b14db1f45`. Their complete upstream copyright,
conditions and disclaimer are provided in `third_party/PyTorch-LICENSE.txt`.
A copy of that license is also packaged beside the portable archive as
`ares/verified_safety/PyTorch-LICENSE.txt`.
This notice covers the copies under `reports/` and inside packaged archives.

Optional dependencies are declared in `pyproject.toml` and inventoried in
`LICENSES.yaml`. Their installed wheels and container libraries are not included
in this source repository; distributing those binaries requires their own
applicable notices and terms.

Publication cleanup pseudonymizes local home paths and updates affected SHA-256
bindings and archive checksums. Mathematical values and the frozen flight
runtime are unchanged. Original evidence is retained in a private local backup;
this publication copy does not represent a new flight validation run.
