# Interval full-state backup feasibility: no certificate

This prototype does **not** establish an actual compiled-DiffAero reachability
contract or an invariant terminal set. It records a failed natural-box approach,
a useful short-horizon enclosure candidate, and the proof obligations that remain.
No ARES package, dynamics, randomization, evaluator, success tolerance or gate
was edited. The NN training freeze was respected.

## Actual model and sets

The expression graph follows the existing adapter's motor exponential update,
its SI-to-relative-thrust divide/multiply path, the vendor nominal-inertia rate
controller including gyroscopic compensation saturation, the actual sampled
inertia gyroscopic term, full quaternion rotation, all four RK4 stages, and
quaternion normalization after each <=.002 s substep. The control period is .01 s.
State includes all 13 rigid-body coordinates plus motor thrust and the public
command queue. Uncertain but constant physical parameters span the complete
profile mass [.45,.55] kg, gravity [9.78,9.83] m/s², drag [.05,.2], motor lag
[.015,.04] s, inertia endpoints and thrust authority [22,27] N. Natural boxes
lose parameter correlations; they never use a sampled true value in control.

Unknown transport delay [.005,.02] s is covered by the convex hull of every
possible causal fractional queue interpolation. The initial public queue is
nominal hover and zero body rate; initial hidden motor thrust is [4.401,5.4065] N.
This is an explicit near-hover history domain, not arbitrary prior commands.
The motor recurrence retains its repeated-command dependence using exact
multilinear corner extrema plus global arithmetic-error padding.

The small state box uses position components +/- .005 m, velocity +/- .01 m/s,
quaternion-vector components +/- .001, and angular rate +/- .01 rad/s. Repeated
feedback measurements additionally have the declared position/velocity and
quaternion-component error boxes. The larger state box uses .03, .05, .01 and
.05, respectively. Unit-norm shell tightening and a positive quaternion chart
are explicit conditional improvements; their target arithmetic bounds are not
claimed audited. The moving case centers velocity at [.1,.1,.05] m/s and
quaternion vector near [.005,-.005,0].

The room is a 12 m cube centered at [6,6,6], with a declared known-free ball of
radius 2 m and spherical body radius .17 m. The backup uses public observations
only: desired acceleration -1.5*(p-anchor)-2.5*v, fixed world-x heading, existing
geometric conversion, yaw contribution limit 1 rad/s and physical command bounds.
The alternative is a fixed nominal-hover command.

## Wrapping results

Propagation deliberately stops when quaternion halfwidth exceeds .4 or angular
rate halfwidth exceeds 10 rad/s, rather than overflowing and presenting useless
boxes. The table gives the last completed control step before that limit.

|Initial set|Input|Unit shell|FP32 padding|Last completed time s|Position center radius m|Max angular-rate halfwidth rad/s|
|---|---|---|---|---:|---:|---:|
|small_public_set|fixed_hover|False|True|0.02|0.00961|0.01951|
|small_public_set|fixed_hover|True|True|0.16|0.06209|3.10949|
|small_public_set|pd_backup|False|True|0.02|0.00962|0.08361|
|small_public_set|pd_backup|True|True|0.10|0.03060|5.63199|
|larger_public_set|fixed_hover|False|True|0.02|0.05438|0.09820|
|larger_public_set|fixed_hover|True|True|0.11|0.08382|2.83649|
|larger_public_set|pd_backup|False|True|0.02|0.05440|0.61870|
|larger_public_set|pd_backup|True|True|0.06|0.06447|8.77209|
|small_public_set_no_rounding|fixed_hover|True|False|0.16|0.06202|3.10941|
|small_public_set_no_rounding|pd_backup|True|False|0.10|0.03055|5.63099|
|small_public_set_moving_tilted|pd_backup|True|True|0.10|0.04112|7.35885|

Small-set PD covers a short prefix through .10 s with center-position radius
.03060 m (swept-body radius about .201 m), which fits the analytic 2 m free ball.
At that point the angular-rate halfwidth is already 5.63 rad/s. It cannot reach
the proposed .50 s return-map horizon, much less establish indefinite terminal
hold. Plain normalization boxes become unusable after only .02 s. Removing
arithmetic padding changes the small-set PD final position radius .03060 to
.03055 m and leaves the same horizon: dependency wrapping dominates this failure.
The moving/tilted set also stops after .10 s. These failures of the enclosure
method do not prove physical instability or unavoidable collision.

A CPU float32 sanity check covered 256 actual DiffAero plants per scenario,
including endpoint and interior parameters: fixed hover/small and large sets,
PD/small and large sets. All 9,216 physical transitions fell inside the recorded
state and motor boxes. These finite point checks can expose implementation
errors; they do not prove universal set containment or a compiled-CUDA bound.

## Numerical status and swept bodies

Host endpoint arithmetic uses float64 outward nextafter and conservative FP32
primitive padding. Reductions receive an absolute gamma_n*sum(abs(term)) bound.
Separate rounded interval subexpressions also enclose a fixed-DAG FMA contraction.
The exp/atan error padding, actual vector_norm normalization bound, arbitrary
compiler reassociation, and possible reduced-precision matrix products remain
**unverified target assumptions**. The prototype is not a soundness claim for
those actual compiled kernels. A validated MPFR/directed-rounding implementation
and an audited target instruction/error model must replace these assumptions.
PyTorch explicitly does not promise bitwise identity for mathematically identical
floating-point computations; NVIDIA's current mathematical-function tables label
maximum error as observed values. See [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
and [NVIDIA floating-point computation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html).

Every recorded macro-step includes an axis-aligned hull of input/output and RK4
stage position boxes. Inflating this hull by body radius covers the discrete
model's endpoint chord and any piecewise-linear interpolation of the enclosed
nodes; in particular it covers the continuous segment used by the existing
macro-step collision evaluator. An RK4-stage hull does not establish containment
of a continuous ODE solution or an unspecified dense-output interpolant. A
physical-flow contract would require an additional continuous-flow enclosure;
none is claimed here.

## Feasible next algorithm and remaining obligations

1. Preserve stable linear parts with a centered/QR/Lohner or common-P enclosure,
   retaining physical parameter symbols across the horizon. In the unsaturated
   nominal gyroscopic-compensation region, the exact real angular equation is
   wdot=K*(u-w)+J^-1*cross(w,(Jnom-J)*w), K=30*Jnom/J componentwise.
   For the existing RK4 update, let z=h*K and
   R=1-z+z²/2-z³/6+z⁴/24. The exact linear contribution is
   R*w+(1-R)*u, with stage nonlinear remainder weights h/6 times
   [1-z+z²/2-z³/4, 2-z+z²/2, 2-z, 1]. Enclose the actual stage remainders and
   floating evaluation error; do not replace the plant with an ODE exponential.
2. Preserve the isotropic-drag identity in the **vendor** homogeneous quaternion
   rotation: rotate(q,d*rotate(conj(q),v))=d*||q||^4*v, including nonunit RK4
   stage quaternions. The controller uses a different unit-quaternion shortcut;
   that shortcut must not silently replace the plant formula. Any algebraic
   simplification needs an explicit target floating-error remainder.
3. Center terminal sets about the physical-parameter-dependent equilibrium
   family, not only nominal hover, and enclose measurement-error injection.
   Common-P finite vertex matrices from the parallel terminal-design work are
   promising coordinates; finite vertices alone do not enclose nonlinear or
   interior-parameter Jacobians.
4. Prove a nonlinear return-map inclusion and every intermediate swept-body
   inclusion, then prove invariance by induction. Cover all queue/motor states
   admitted at entry and every allowed delay; initial histories here are narrow.
5. Audit the compiled arithmetic and connect independently enforceable public
   state, parameter and map assumptions to the contract. Statistical sensor
   noise tails cannot be relabeled deterministic bounds from these tests.

No terminal invariant set, all-time backup guarantee, actual simulated shield
certificate, or Phase 4 gate pass follows from this artifact.

## Reproduction

Run the archived prototype and containment scripts with the repository virtual
environment. They write adjacent JSON files. Source hashes in containment.json
must match the intended runtime. feasibility.json contains complete prefix boxes,
swept-position hulls, parameter sets and unresolved numerical assumptions.
