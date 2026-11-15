# Verified point kernel v2: adaptive exact atan enclosure

`verified_backup_point_v2.py` is a versioned extension of v1. An AST comparison proves the entire module is identical except `I.atan`: the REAL backup law, controller gains, supported local branch, force/attitude construction, every other interval operation, actuator guards, `output32` and the exact final1e-6 command-error acceptance are unchanged. v1 and its archived proof remain intact. Wiring this version into the executing shield requires an explicit SHA-bound compatible extension; these artifacts do not silently replace the old import.

For 0<=x<=1/2, integrate the finite geometric identity

`1/(1+t²) = sum_{j=0}^k (-t²)^j + (-1)^(k+1)*t^(2k+2)/(1+t²)`.

The remainder after integrating from0 to x has sign `(-1)^(k+1)` and magnitude at most `x^(2k+3)/(2k+3)`. Thus after adding term k, with total S and next-term magnitude r, atan(x) lies in `[S-r,S]` for even k and `[S,S+r]` for odd k. The algorithm computes every quantity with exact Fractions and stops only when r<=1/10^12, or after33terms. On the whole domain, the final r is bounded by `(1/2)^67/67<1/10^12`, an exact rational inequality recorded in the compatibility artifact. Negative arguments reflect the bounds using odd symmetry. Monotonicity, from positive derivative1/(1+x²), makes the lower/upper endpoint evaluation a valid interval enclosure.

This changes neither the REAL atan value nor the final command tolerance. Each command is still accepted only after unchanged `output32` proves its binary32 word lies within1e-6 of the entire exact command interval. Wider early atan enclosures may select a different tiny binary32 command word, but the same absolute-error box applies; one word differed across the129 tested public points. The RPI's existing Gamma-propagated1e-6 command allowance therefore remains the same. No physical dynamics, noise, domain or certificate premise is relaxed.

Five tests pass: AST identity outside atan; containment of v1's narrower33-term enclosure in the adaptive bracket for rational/dyadic and ±.5 endpoints; tiny cancellation intervals and odd symmetry; invalid domain rejection; the original20 finite command cases with all exact output checks retained. The proof of enclosure is the integral remainder argument, not these samples.

`verified_backup_point_v2_audit.py` checks compatibility and benchmarks standalone point calls. It reads the real public admission case0 and constructs128 additional represented-Float32 local points whose exact current-C membership is independently checked before timing (maximum channel quadratics range from .2003 to .3905). All final command-error checks pass. Atan uses1–2terms on these points, versus33 in v1. On this run:

| Standalone calls | v1 median | v2 median | v2 p99 | v2 maximum |
|---|---:|---:|---:|---:|
| admission case0,300 each | 4.288ms | .537ms | .951ms | 1.123ms |
|128 local C points,3 passes | 2.834ms | .562ms | .967ms | 1.095ms |

Medians improve by about7.98x and5.04x respectively. Timing is finite regression evidence and gives no hard deadline guarantee. The full fresh coupled deadline gate remains a separate run owned by the parent task. `verified_backup_point_v2_benchmark.json` preserves precise results, source/input hashes and all-point exact membership/error flags. No flight certificate is emitted.
