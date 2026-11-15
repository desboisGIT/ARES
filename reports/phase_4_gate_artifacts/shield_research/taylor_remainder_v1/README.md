# Real nonlinear Taylor bounds; no flight certificate

This research supplies interval Hessian and centered remainder bounds for the
actual **real-arithmetic** 21-state backup map, including its nominal rate
controller, motor recurrence, causal fractional command queue and five RK4
substeps. It also separates the physical flow from the new command. It does not
establish a machine-execution bound, nonlinear terminal invariant set, swept-body
flight certificate or Phase 4 shield gate. No existing ARES package source,
physics, measurements, collision checks or gates were edited.

## Coordinates and external assumptions

The centered state order is
`[p3, v3, q_xyz3, body_w3, motor_thrust-mg, queue1_4, queue2_4]`.
Queued collective thrust is centered at `mg/b`; queued rates are centered at zero.
The positive unit-quaternion chart is `q_w=sqrt(1-|q_xyz|²)`. Quaternion input is
exact in this model; public position/velocity measurement errors are additional
six-dimensional bounded inputs. This is an explicit model restriction, not a
claim about the current noisy simulator estimator or a supplier guarantee.

The observer-free backup holds a fixed anchor with net acceleration
`-1.5*(p_measured-anchor)-2.5*v_measured`, fixed world-x heading, nominal mass .5,
nominal gravity 9.80665, attitude gain 6 and heading-contribution limit 1 rad/s.
The deployed controller needs only public measurements and these nominal
constants. Physical parameters enter the proof coordinate transformation and
uncertainty sets; they are not controller observations.

For fixed physical parameters and battery authority `b`, the exact equilibrium
position offset is `(9.80665-mg/(.5*b))/1.5` vertically, applied motor thrust is
`mg`, and queued requested thrust is `mg/b`. This family is analytically centered.
Changing battery authority shifts that center and creates a separate disturbance;
the centered fixed-parameter map does not remove or cover that disturbance.

All physical profile intervals are widened to enclose binary32 endpoints.
The nominal controller-inertia lift includes both decimal constants and their
binary32 representations. Battery authority spans `[19/20,1]`. The real substep
is exactly `1/500 s`; five substeps form a .01 s control period. Unknown delay is
split into `[.005,.01]` and `[.01,.02]` s branches. Queue and motor states remain
explicit; no point-mass braking constant replaces them.

## Interval arithmetic and branch domains

The second-order interval AD tracks value, gradient and Hessian over all state,
noise and physical-parameter boxes. Basic binary64 interval endpoint operations
round outward with `nextafter`, under the ordinary IEEE basic-operation contract.
There is no hidden BLAS reduction: dot products are explicit interval additions.
Square-root endpoints are checked against exact rational squares. The near-hover
atan domain `|x|<=1/2` uses exact rational alternating-series bounds. Motor
`exp(-a)`, `0<=a<=1`, uses rational alternating-series partial sums of orders
31/30. Output remainder sums use exact `Fraction` arithmetic, then round upward.
These bounds come from interval/analytic formulas, not sampled Hessian maxima.

Before differentiating a smooth expression, every queried domain proves:

- Positive initial and intermediate quaternion scalar/chart and nonzero
  normalization denominator, including every RK4 normalization.
- Force norm above the converter's clamp threshold; the structured graph also
  proves positive force z.
- Positive heading x above the heading-validity threshold, without an atan branch
  cut, and heading contribution strictly inside its 1 rad/s limit.
- Current and both queued thrust/rate commands strictly inside policy and plant
  limits (including the minimum possible 22 N thrust authority), and positive
  bounded motor thrust.
- Nominal gyroscopic-compensation norm strictly below its saturation threshold.

A domain crossing any unhandled branch is rejected. Clamp derivatives are not
silently assumed. Accepted bounds quantify the real expression graph only.
Actual float32 conversion, fused/reassociated kernels, normalization, actuator
updates and compiled arithmetic need a separate explicit error budget. None is
supplied or inferred from the float64 audits here.

## Taylor decomposition

For fixed admissible parameters `theta`, let `F_theta(z,e)` be the centered real
map and `A_theta=partial_z F_theta(0,0)`. Analytically `F_theta(0,0)=0`. Product
ellipsoids imply component bounds `|z_j|<=d_j`. On that whole box, interval AD
supplies `M_ijk >= sup |partial_j partial_k F_i|`. Integral Taylor's theorem gives

```
|F_i(z,0) - (A_theta*z)_i| <= .5 * sum_jk M_ijk*d_j*d_k.
```

A separate mean-value noise term bounds
`|F_i(z,e)-F_i(z,0)| <= sum_l sup|partial_e_l F_i|*E_l`.
The original full-map JSON files report these two separate quantities. The noise
term includes the *linear* noise response; it must not be added again on top of a
linear-noise ellipsoid proof that already includes it.

`controller_split_bounds.json` instead gives the unambiguous joint decomposition

```
u(z,e) = C_theta*z + D_theta*e + r_controller(z,e),
```

using the complete 27-dimensional Hessian to bound the final term. Local C/D are
intervals over the whole physical-parameter family, with analytic causal zeros.
This permits linear measurement noise and nonlinear residuals to be counted once.

Ellipsoid-to-component coefficients are independently audited with exact rational
matrix inverses and square comparisons. `projection_audit.json` proves the chosen
P projection bounds and Q diagonal projection bounds, and positive definiteness.
It does not prove dynamics or nonlinear invariance. Source matrices are retained
in `source_common_P_direct_audit.json` and `source_linear_noise_Q_candidate.json`.

## Equivalent real controller graph

The original expression graph is retained in `interval_taylor_original.py`;
its hash matches the original `taylor_bounds.json` source hash. The structured
option uses these exact unit-quaternion identities:

```
R^T*((R*e_z) cross desired_z) = e_z cross (R^T*desired_z)
tilt_rates = 6 * [-desired_body_y, desired_body_x, 0]
```

For positive force z, normalize via
`rx=fx/fz, ry=fy/fz, desired_z=(rx,ry,1)/sqrt(1+rx²+ry²)`.
This removes false dependence caused by normalizing fz and later dividing by it.
Heading, yaw axis and body-z are explicit unit-quaternion polynomials. Centered
thrust is `.5*dot(delta_force,body_z)-force_eq_z*(qx²+qy²)`. These are identical
real maps; no bitwise equivalence of the changed floating evaluation order is
claimed.

For product P radii `[.1,.1,.1,.05]`, the maximum zero-noise full-map component
remainder decreases from .001504 to .000746. On the useful-noise Q box, it
reduces .016199 to .008871. Full vectors and signed derivative intervals are
retained; scalar maxima are not sufficient for an invariance test. Genuine
horizontal-force/vertical-force quadratic coupling remains. An absolute
componentwise Hessian bound also loses correlations inside the P/Q ellipsoids.

## Independent-command physical flow

`interval_flow.py` treats the new four-component SI command deviation as an
independent input. It preserves the full rigid body and actual RK4 update. Its
real local map has

```
F_flow(z,u) = A_flow_theta*z + Gamma_theta*u + r_flow(z,u).
```

The joint Hessian bounds `r_flow` on the whole state/command domain. The motor
recurrence and queue outputs are exactly affine for fixed parameters, so their
flow remainders are **exactly zero**. Local Gamma is bounded over the entire
parameter and delay branch; it is never a sampled point gain. Exact causal zeros
and the new-queue identity are explicitly retained. On the longer-delay branch,
Gamma contains only that queue identity because the new command has not yet
reached the plant.

Substitution gives the correct combined decomposition

```
F_backup = (A_flow+Gamma*C)*z + Gamma*D*e
           + Gamma*r_controller + r_flow.
```

Controller nonlinearity is propagated through Gamma once. Reusing full-backup
remainders and also adding `Gamma*r_controller` would double count it.

The safe outer command box derived directly from the structured controller is
approximately `[.145432 N,.355726,.355726,.008416 rad/s]`. The joint-flow remainder
has maximum components .0001303/.0001646 on the two branches, with zero motor and
queue remainders. An independently exact-audited Q-support command domain narrows
this to `[.078063 N,.149683,.149683,.008415185 rad/s]`; the corresponding flow maxima
are .0001237/.0001646. The delayed branch is unchanged because it uses old queue
entries. The separate `narrow_joint_flow_*` artifacts preserve that comparison.

An additional outer-domain study inflates x/y/z Q component boxes by 1.5 and yaw
by 3 while keeping measurement bounds .005 m/.01 m/s. Its controller Hessian and
joint-noise remainder data are separately retained. A later Q candidate must be
proved to fit this domain; diagonal bounds alone do not justify reusing a
stronger `C*Q*C^T` support bound from a different ellipsoid.

## Quaternion-increment physical flow refinement

`interval_flow_increment.py` keeps the same exact real RK4 map while factoring
its quaternion update. Angular dynamics do not depend on quaternion. For each
substep, hold its incoming unit quaternion `q_in` fixed and integrate the
quaternion increment `r` from identity using the same angular RK4 stages.
Bilinearity and RK4 affine equivariance give `q_stage=q_in*r_stage` and
`q_raw=q_in*r_raw`. Quaternion norm multiplicativity then gives
`q_out=q_in*normalize(r_raw)`. No normalization denominator depends on `q_in`.
Positive raw/output scalar and normalization branches are still checked.

The vendor rotation is the homogeneous quaternion rotation, including at nonunit
RK4 stages. Consequently `R(q_in*r)=R(q_in)R(r)` and isotropic drag is
`drag*||r_stage||^4*v`, independent of the incoming quaternion. Only `R(q_in)` uses
the unit shortcut. A direct Hamilton product avoids artificial interval
cancellation in the vendor's algebraically optimized multiplication. These
identities are independently checked symbolically in
[quaternion_algebra_exact.json](../quaternion_algebra_exact.json), including the
vendor optimized-product equivalence. They justify the real rewrite, without
assuming unit intermediate stages or asserting floating reordering equivalence.

On the inflated Q domain with command box
`[.115898 N,.223855,.223855,.024254336 rad/s]`, this reduces x/y quaternion
remainder bounds from .000149/.000188 to .00001046/.00001260 on the two delay
branches (about 14 times tighter). Overall maxima become .0002725/.0003613,
dominated by velocity components; angular-rate remainders remain unchanged.
Motor and queue remainders remain exactly zero. Full signed arrays, global
parameter-family Gamma intervals and a matching six-case actual-vendor audit are
in `increment_inflated_flow_*`. Genuine three-dimensional force/motor/quaternion
and gyroscopic coupling is retained.

## Verification, failure evidence and limits

Eight focused research tests cover interval derivatives, exact sqrt comparisons,
transcendental brackets, mixed Hessians, rejected branch domains and overlap of
the two equivalent real graphs at a point. Matching six-case vendor float64 audits
check the original full map, structured map and independent flow, including
values, Jacobians, selected Hessians and Taylor remainders. Reported excesses are
about 1e-15, below the explicitly diagnostic 1e-9 tolerance. These finite checks
can catch implementation errors; neither their tolerance nor sampled derivatives
is used to define the interval remainder bounds. They do not certify floating
execution.

The companion product-P ISS triangle test fails the original .1/.1/.1/.05 domain:
for example planar nonlinear P-norm bound about .00434 exceeds its .0003356
contraction budget even without measurement noise. Useful noise is larger still.
This is failure of a sufficient method, not evidence of physical instability.
Structured expressions, noise-aware Q geometry and the flow split improve the
building blocks, but no nonlinear invariant-set conclusion follows here.

Remaining obligations include a complete noise-aware nonlinear inclusion proof,
parameter/battery center drift, actual machine-arithmetic and scheduling bounds,
validated current public-state and map contracts, capture-to-now reachability,
entry/transition conditions and intermediate swept-body coverage. Default
Gaussian sensor models do not satisfy deterministic hard bounds merely because
these artifacts use bounded input variables. No flight or shield gate is marked
passed by this research.

## Reproduction

Run `projection_audit.py`, `run_structured_taylor.py`,
`run_structured_noise_q_domain.py`, `run_inflated_q_domain.py`,
`controller_split.py`, the applicable flow driver, and the matching audits using
the repository virtual environment. `test_interval_taylor.py -v` runs the focused
research tests. Re-running a driver writes adjacent research outputs, so preserve
this frozen artifact directory when comparing a future candidate. The manifest
records hashes. Existing package sources remain unchanged.
