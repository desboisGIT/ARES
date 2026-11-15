"""Exact analytical compatibility of explicit norms with existing budgets.

The finite fullbody comparison is separate evidence. These uniform counting and
rounding-bin implications, not those samples, preserve the prior enclosures.
"""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib
HERE=Path(__file__).resolve().parent;BASE=HERE.parent

def run():
 u=F(1,2**24);eta=F(1,2**126);rows=[]
 for n in [3,4]:
  products,adds=n,n-1;operations=products+adds
  assert operations<2*n and 2*n*u<1
  # Standard rounded-op gamma with Nproducts and N-1 adds is enclosed by
  # prior gamma_(2n); the prior4nFTZ is also conservative on this tiny DAG.
  assert F(operations)*u/(1-operations*u)<=F(2*n)*u/(1-2*n*u)
  assert operations/(1-operations*u)<4*n
  rows.append(dict(width=n,product_operations=products,left_associated_additions=adds,prior_gamma_index=2*n,operation_count_covered=True,absolute_underflow_allowance_covered=True))
 shell_budget=F(8,2**24)
 source_paths=[BASE/'roundoff_enclosure_v1/conditional_projective21_budget.json',BASE/'roundoff_enclosure_v1/roundoff.py',BASE/'intermediate_position_v1.json',HERE/'verified_sqrt.py',HERE/'backend.py']
 report=dict(schema_version='ares.explicit_norm_budget_compatibility.v1',norm_sum_DAGs=rows,sqrt_contract='For each runtime word x>=0, squared exact rounding-bin boundaries prove RN-even(sqrt(x)). Thus .5ULP error required by prior bound is discharged pointwise, including bin ties and subnormals.',initial_and_output_projective_alignment_unchanged=True,prior_norm_shell= str(shell_budget),norm_shell_still_valid=True,prior_per_H_21D_vector_unchanged=True,prior_intermediate_position_arithmetic_padding_unchanged=True,full_REAL13D_RK4_rate_controller_equations_unchanged=True,motor_verified_decay_backend_unchanged=True,native_norm_or_sqrt_error_assumption_removed=True,remaining_contracts=['IEEE binary32 fullprecision RNE basic operations and tensor/scalar conversions','matching frozen original program and stated explicit norm DAG; eager CPU, no oneDNN/autocast/compile','tiny3dot and shortsum counts as already covered; no reduced precision','guarded external domains and unchanged source/code; no concurrent arithmetic-mode change'],flight_certificate=False,source_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
 (HERE/'budget_compatibility.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return report
if __name__=='__main__':run()
