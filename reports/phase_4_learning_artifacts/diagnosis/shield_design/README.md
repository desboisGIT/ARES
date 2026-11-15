# Shield research index

No flight certificate or overall shield gate is enabled by these artifacts.

Start with `conditional_gate_design.md` for the precise conditional domain, proof argument, missing external services, and operational limits. `manifest.json` hashes the source-model and research inputs. Research scripts use archived companion drivers and some `/tmp` paths from their original executions; reproduce by copying the archived drivers to those original basenames in `/tmp`. The SDP search uses an isolated `/tmp/ares-shield-sdp` installation; ARES environment dependencies were not modified.

Completed exact mathematical building blocks:

- `backup_Bernstein_exact_audit.json`: full parameter-interior local-linear stability with outward Float32 physical bounds and battery authority [.95,1].
- `backup_direct_contraction_audit.json`: exact direct-metric contraction bounds, plus exact inverse-P component projections.
- `backup_linear_noise_exact_audit.json`: useful externally bounded measurement-noise local-linear RPI over joint Bernstein (A,B) coefficient sets. Exact LDL replaces numerical SDP feasibility as the matrix proof.

Unclosed full-plant obligations and retained failures:

- `backup_product_RPI_feasibility.json`: all tested naive nonlinear/noise/drift triangle-ISS sufficient inequalities fail. This is a failed proof method, not a demonstrated physical collision.
- `backup_oriented_Hessian_feasibility.json` and `backup_noise_Q_oriented_Hessian_feasibility.json`: numerical correlation-aware remainder research, still insufficient for a nonlinear certificate.
- Floating execution/rounding, actual nonzero nonlinear invariant region, changing-authority drift, trusted bounded state/map service availability, and a proof-backed coupled moving-flight guard remain open.

`public_trim_contract.md` describes an optional public-history calibration correlation without exposing sampled mass or initial actual mg thrust. The root's separate exact trim-history lemma is under `reports/phase_4_gate_artifacts/shield_research/trim_history_v1`.

Earlier decimal-only and battery=1 local proof artifacts remain in versioned subdirectories. These are not the current operational domain.
