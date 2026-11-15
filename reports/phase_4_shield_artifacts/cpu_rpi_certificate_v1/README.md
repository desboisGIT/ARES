# Conditional CPU backup proof bundle

This is a finite-horizon, full three-dimensional backup invariant-set proof **under explicit bounded-service and arithmetic contracts**. It does not certify the default Gaussian/CUDA policy, a physical vehicle, or its own execution merely by loading a JSON file. `certificate.json` is the machine-readable statement and lists remaining runtime obligations separately from completed mathematical checks.

## Closed argument

1. The 21 coordinates include position, velocity, positive-hemisphere projective quaternion xyz, body rates, actual motor force, and two delayed command words. The product ellipsoid includes nonzero uncertainty in every physical state direction. Physical parameter endpoints enclose represented Float32 endpoints. Battery authority is in [.95,1], with a correlated, verified per-period equilibrium shift.
2. Exact quaternion algebra converts the same real full-quadrotor RK4 graph into an increment-factor graph. For each substep, independently lifting `F_motor/m` into the proved interval `20562575/2097152 +/- 7/8` contains the actual plant. It removes a multiplicative force term from additive Taylor error without replacing the plant by a point mass or ideal servo.
3. Exact polynomial stage-affinity identities and tensor Bernstein coefficients enclose the full parameter interior. The five gains have 32 sign combinations. Controller/measurement and plant input matrices share their convex weights; their parameter correlation is preserved.
4. Interval second derivatives cover the whole outer product domain. Exact rational LDL verifies covariance outer factors and 1183 oriented operator-norm bounds. Controller joint state/noise remainder propagates once through the actual command-input matrix. Physical-flow remainder, conditional Float32 error, and correlated battery equilibrium drift are then added. Command and queue errors are not counted twice.
5. Every exported matrix/sign inequality is checked by independent outward binary64 interval LDL, including both command signs. The fixed certificate checks 2,709,504 planar, 48,384 vertical and 504 yaw matrices, as well as positive-definiteness and outer-domain containment. All passed. Solver output and floating eigenvalues only proposed the matrix; neither establishes its sign.
6. For each channel, `Q >= A Q A^T / alpha + w w^T / (1-alpha)` contains the image of its ellipsoid. Joint matrix convexity extends the checked coefficients/signs to their hull. The outer-domain remainder bounds make this a nonlinear product-set argument. Induction yields the declared 1000-period conditional endpoint invariance.
7. The attached intermediate-position proof encloses every original RK4 stage/substep endpoint and the simulator collision chords within the positional reserve plus 5 mm per axis. It makes no claim about an unspecified continuous interpolant. The runtime must still prove that the entire reserve, including body radius and public equilibrium uncertainty, is known free.

## Runtime acceptance

Require a current independent hard-bound state/map packet, exact initial augmented-state set membership, public command/motor-history correlation, the stated arithmetic backend and primitive contract, a current free-space reserve, and sufficient contract/authority lifetime for the cached backup. The bundle does not supply these services. Stale, RMS, Gaussian-only or probabilistic bounds cannot establish the hard contract. The installed norm/sqrt arithmetic obligation is explicitly unresolved by the MXCSR mode guard alone.

A proposed new anchor can reuse this **fixed-anchor** theorem if the current complete public uncertainty set is first proved to lie inside the new anchor's ellipsoid and the new whole reserve is known free. Retain the old anchor until both checks pass. Arbitrary anchor translation is not authorized by this fixed certificate.

## Replay

Run from any directory, using absolute script paths and an environment with NumPy (unittest for tests):

```sh
python /absolute/path/to/cpu_rpi_certificate_v1/replay_exact_remainder.py
python /absolute/path/to/cpu_rpi_certificate_v1/replay_sector_construction.py
python /absolute/path/to/cpu_rpi_certificate_v1/export_coefficients.py
python /absolute/path/to/cpu_rpi_certificate_v1/interval_lmi_audit_v1.py /absolute/path/to/cpu_rpi_certificate_v1/planar_x_interval_coefficients.npz --alpha 4458563631096791/4503599627370496 --both-command-signs --output /tmp/replayed_planar_audit.json
python /absolute/path/to/cpu_rpi_certificate_v1/interval_lmi_audit_v1.py /absolute/path/to/cpu_rpi_certificate_v1/vertical_interval_coefficients.npz --alpha 4460365070947739/4503599627370496 --both-command-signs --output /tmp/replayed_vertical_audit.json
python /absolute/path/to/cpu_rpi_certificate_v1/interval_lmi_audit_v1.py /absolute/path/to/cpu_rpi_certificate_v1/yaw_interval_coefficients.npz --alpha 8872091265919877/9007199254740992 --both-command-signs --output /tmp/replayed_yaw_audit.json
```

The exact-remainder replay reproduced the original rational remainder values and all 1183 norm certificates. The construction and exporter have no dependency on a lost `/tmp` driver. Source interval graphs, signed derivative arrays, arithmetic analyses, the physical source51 freeze and vendor source hashes are archived and linked by `bundle_sha256.json`. Finite vendor/autograd comparisons remain implementation sanity checks; they do not supply universal derivative, sign or rounding bounds.
