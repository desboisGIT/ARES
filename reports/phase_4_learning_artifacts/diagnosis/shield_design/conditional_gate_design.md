# Conditional full-quadrotor shield: design and proof obligations

Status: research/design, not a passed gate. No ARES runtime sources changed. Existing `SafetyShield` v2 stays unchanged and its coupled caller must continue passing `assumptions_valid=False`.

## The precise boundary of the claim

The build plan's section 4 requires a stopping path in known free space; its Phase 4 gate requires that no collision preventable under the shield's assumptions escapes prevention. The appropriate proof-backed software target is:

> Given a sound bounded set containing the actual augmented plant state and parameters, a sound static known-free voxel set, bounded feedback measurement errors, and declared command/timing/arithmetic semantics, every command accepted by the guard has a collision-free reserved full-quadrotor backup and an invariant terminal continuation while the declared contracts remain valid. The same guard can continue accepting mission motion while preserving that property.

The operational gate may use an explicitly declared finite mission/stopping horizon (currently 10 s), with contract expiry covering that horizon and enough time reserved to start the backup. A mathematical invariant-tail result applies only while its authority/state/map assumptions hold; it does not require physically infinite battery hover. This is a conditional simulation theorem plus actual coupled simulation verification. It is not a hardware certification. It is also not an unconditional theorem about the current Gaussian sensing pipeline. The existing unbounded Gaussian benchmark remains unchanged, retains every run, and cannot assert the bounded external service merely because observed errors happened to be small.

Finite extrema, including measured minimum braking .881 m/s², are not universal bounds. The current straight-path braking model also fails to describe attitude/rate/motor transients. A larger finite terminal radius fitted to traces would not solve either proof problem.

## What current interfaces cannot establish

1. Position, velocity, angular-rate and depth errors have unbounded tails. An innovation gate cannot establish actual error containment from a noisy sample. For the actual finite mission horizon, a bounded-error event can have positive probability, but a public innovation gate still cannot deterministically establish its truth. Over an infinite horizon of independent nondegenerate Gaussian samples, a finite bound on every error has probability zero; that observation limits an infinite unconditional interpretation, not the finite Phase 4 mission requirement.
2. Sparse sampled rays do not automatically prove an entire voxel is free. Pose error, range error, camera calibration, capture time, voxel occupancy semantics and first-hit rendering must be accounted for before marking free. The existing observed map does not supply such a certificate.
3. The current controller does not provide a minimum-rate straight stop or a robust invariant hold set. Saturated thrust/rates bound commands but do not establish the requested 5 m/s² acceleration bound. With unrestricted attitude and 24.5 N command, a .45 kg plant can have acceleration far beyond 5 m/s².
4. The current state input omits guaranteed applied-motor-thrust and command-queue sets. Both affect future motion after a new override. The parameter ranges and queue history must be part of the reachability model even though sampled parameters must remain hidden from the policy/guard.
5. Timing validity must cover the reserved backup and its feedback, not just the latest packet. If future measurement service fails, a previously reserved feedback backup is not automatically safe. A bounded sensor-outage prediction phase needs its own proof and deadline; arbitrary indefinite loss of a trusted state service is not safe in this free-flight model.
6. Current floating-point/GPU compiler/transcendental semantics are not a validated interval arithmetic contract. A proof of real-valued equations alone must not be presented as a proof of arbitrary compiled numerical execution.

A minimal external requirement is therefore a bounded safety-state/map service (including its availability horizon), or an equivalent sound set-membership estimator with a justified measurement model. A conditional test harness may provide and independently check those assumptions using evaluator truth. Truth does not enter the guard. That harness is a separately declared bounded-contract simulation domain, not a change to or a filtered subset silently replacing the old Gaussian benchmark.

## Bounded state and map contracts

Use explicit typed packets with capture time, expiration, coordinate frame, bounds, provenance and map version. State packets provide sets for position, velocity, orientation, angular velocity and any required controller state. A model-based supervisor propagates motor-thrust and delayed-command sets from known issued commands and full profile bounds. Reject stale, malformed, frame-mismatched or inconsistent packets and fail closed when set containment or numerical enclosure is not established. Rejection does not imply the physical state is safe or a collision unavoidable.

`state_map_contract_prototype.py` implements report-only provenance/finiteness/expiry checks, outer state boxes and a sufficient robust free-voxel witness. Its returned `runtime_certificate` is always false because it has not been integrated with validated numerical arithmetic and plant reachability.

For a voxel-constant occupancy world, nominal capture position p, rotation R, unnormalised optical ray d, and common optical range s, every physical ray point is within

`e_p + s * (e_d + 2*sin(e_theta/2)*norm(d))`

of `p + R*d*s`. Here position error is at most e_p, orientation angle error e_theta, and ray-vector calibration error e_d. If this entire ball lies strictly inside one voxel and `0 < s < measured_depth - e_depth`, every possible physical ray passes through that voxel before its first possible hit. Therefore that homogeneous voxel is free. This proves a whole voxel from robust traversal, not from an unjustified free-space cone. No witness means unknown. The geometry requires validated outward arithmetic before use as an executable proof.

Map-free statements must remain valid throughout the reserved tube and terminal continuation. This is reasonable in the current static scene model; dynamic obstacle intrusion needs additional assumptions. Unknown voxels and map support boundaries remain blocked.

## Full-state candidate and backup guard

Augment the actual 13-state rigid body with applied motor thrust and command-queue memory. The local minimum representation has 21 coordinates: position3, velocity3, quaternion-vector3 on a positive-scalar local chart, body rates3, motor thrust1 and two previous four-component commands8. A global implementation must use full quaternion sets/chart transitions or reject outside its verified local chart. Include any additional backup observer memory if used; do not omit hidden state.

A simple proof candidate is an independent observer-free backup:

- Capture a fixed position anchor and fixed heading when backup starts.
- Request net acceleration `-Kp*(p-anchor) - Kv*v`, with explicit gains.
- Use the existing nominal-mass geometric attitude/thrust conversion with fixed heading, rather than the mission's gaze target.
- Retain actual full rate dynamics, inertia mismatch, motor lag, command delay, saturation and sampled battery authority if present. The original feasibility probe omitted battery depletion; the later exact local embedding includes a battery authority interval [.95,1]. Its nonlinear effect and equilibrium drift still require explicit coverage.

This backup uses only public bounded state inputs and fixed nominal constants. True mass/gravity are used only symbolically by the offline proof to describe its equilibrium family. With level hover and no integral observer, the equilibrium offset is

`p_eq,z - anchor_z = (g0 - (m/m0)*g) / Kp`, with `T_eq = m*g`.

For Kp=1.5, m0=.5, the full mass/gravity range gives offset [-.6709,.66977] m. A finite invariant region must include that offset family; claiming millimetre hover about the anchor would be wrong. This sacrifices clearance for a simpler proof candidate. A later integral/trim backup may improve the region only after its augmented dynamics are verified.

At each guard step, form the reachable tube for the proposed **physical** thrust/body-rate command for one control interval, followed by the backup law until a verified terminal set is reached. Cover every activation queue state and all physical substeps. Check tube disjointness from occupied/unknown space after body inflation, speed/domain constraints, and terminal inclusion. Accept only with a sound inclusion proof. Cache the reserved backup and its domain/clock obligations. Replanning may improve it, but a missed check cannot erase the existing reservation.

The guard should not call a finite mesh of controller simulations a verified tube. It must use validated interval/set propagation or an analytic tracking-error enclosure. Source hashes, parameter ranges, arithmetic contract and proof artifact hashes accompany every enabled contract version.

## A finite proof of infinite terminal hold

Let theta contain the static uncertain profile parameters. Shift the augmented state by its parameter-dependent equilibrium; call the resulting state z. Let C be a finite-dimensional set with nonzero uncertainty in every position, velocity, attitude and angular-rate axis, motor thrust and queue coordinates. A point/vertical-only terminal set is insufficient.

Choose an N-step return map of the actual sampled controller/plant. Verify universally:

1. For every `(z,theta)` in C and every permitted sequence of feedback errors, every one of the N intermediate reachable sets is inside the verified coordinate/model domain and its position projection stays in a recorded terminal region.
2. The N-step reachable set is contained in C.

Induction gives invariance at each N-step boundary and the recorded intermediate sets cover all later times. Their union, inflated by the physical body radius, is the terminal geometric reserve. This allows curved motion and corrective reversals without a fictitious straight minimum-braking guarantee.

An alternative sufficient proof uses a positive definite metric P and validated return-map bounds

`norm(F_theta,N(z, errors))_P <= rho*norm(z)_P + d`,

with `rho < 1` and `d <= (1-rho)*r` for C contained in the radius-r domain. The nonlinear remainder, parameter interiors, bounded feedback errors, numerical rounding, timing and every intermediate state must all be covered. Floating-point eigenvalues or sampled corners alone do not verify this inequality. The full P-ball may admit large angular rates or motor excursions: additional intersected domain constraints must themselves be checked on return, not merely assumed.

`backup_linear_feasibility.py` is a finite floating-point local-linear probe of this route. It uses full actual rigid-body dynamics, motor exponential updates, two delayed-command slots, nominal inertia rate control and existing geometric conversion. For 64 endpoint combinations (all-low/all-high inertia vectors, not every mixed inertia), every local Jacobian has spectral radius below .993616. In a metric obtained from the nominal discrete Lyapunov equation, worst sampled one-step norm is 1.08334, while the 50-step norm is .793454 and 200-step norm .302438. This makes a .5 s return-map proof plausible. It is NOT a nonlinear or parameter-interior certificate. Stored A/P candidates are research artifacts only.

## Nonvacuous coupled validation protocol after proof artifacts exist

Use a 12 m analytic room initially, with a certified known-free region large enough for the derived terminal reserve (likely at least 1.5–3 m with the simple untrimmed PD backup). This is a declared proof-development domain, not a replacement for cluttered waypoint cohorts.

Require full actual quadrotor execution with nonzero 3-D position/velocity/attitude/rate uncertainty, nonzero moving starts and coupled x/y/z proposals; include profile endpoints and mixed interiors, adverse lag/queue states, occupied walls, unknown slabs and corner approaches. Require positive certificates, accepted mission motion, measurable waypoint/progress distance, interventions when proposals would exhaust the reserved backup, and zero physical collisions/unknown entries among runs whose external assumptions remain valid. Independently check actual states lie inside every claimed enclosure; a miss is a proof/implementation failure, not a relabelled invalid case.

Retain all deliberately invalid cases: stale/broken bounded-service contracts, timing misses, inconsistent sets, inversion/large rates outside the verified chart, parameter/authority loss, initial overlap and insufficient free backup region. No certificate outside the domain. Report their physical outcomes and distinguish certificate rejection from unavoidable collision.

Re-run the unchanged full Gaussian benchmark separately; report how often its current estimator/map cannot supply the required external contract. A proof-backed bounded-domain module cannot by itself close operational integration if the actual mapper/estimator cannot provide its contract. No overall Phase 4 or hardware completion claim follows from the conditional module alone.


## Exact local parameter-interior certificate completed

The subsequent `backup_Bernstein_exact_audit.json` strengthens the finite local feasibility result. An exact rational polynomial representation of the linearized held-input RK4 channels is converted to Bernstein coefficients. Matrix convexity places every local Jacobian in the coefficient-matrix convex hull over conservative independent lifted parameter intervals. Exact Fraction-based LDL verifies positive definiteness of `P - .0001*I` and `P - A.T*P*A - epsilon*I` for every coefficient matrix: 5,292 planar matrices with epsilon=.003, 1,512 vertical matrices with epsilon=.001, and 63 yaw matrices with epsilon=.008. The second planar channel follows by a sign similarity, including independently varying inertia axes. No floating-point eigenvalue sign is used for this matrix audit.

Physical mass/gravity/drag/inertia/motor-lag endpoints are widened outward by one Float32 ULP, and nominal-inertia constants include their actual Float32 representations. Delays are represented by an explicit normalized queue-fraction contract [.5,2], with breakpoints .5,1,2 and two stored commands. The proof still uses exact real substep `1/500`; float32/compiled arithmetic is a separate unclosed obligation. The derived channel expression matches actual full-model autograd Jacobians at 104 endpoint/interior/mixed-inertia cases with maximum entry discrepancy 4.44e-16; that numerical comparison is a regression crosscheck rather than the universal proof.

Battery authority is included in the local embedding: planar force-direction feedback depends on `b*m0/(m*g)`; vertical motor input multiplies the queued SI thrust request by b while the queue stores unscaled requests. A real-model 10 s upper energy budget follows without sampled truth: thrust at most 24.5 N, `sqrt(24.5)<4.95`, hence power at most 970.2 W and energy at most 9702 J. With minimum battery energy 43200 J, the authority factor remains at least .9550833, so [.95,1] is conservative. Numerical power/energy errors still need an explicit arithmetic allowance.

This proves a local-linear parameter-interior stability lemma, not a nonlinear invariant set. In particular, the battery-dependent equilibrium is `pz=(g0-m*g/(b*m0))/Kp`, and changing authority adds an equilibrium-coordinate disturbance. The real 100 Hz battery decrement is at most `.2*970.2*.01/43200 = .0000449167` per step under this budget. Position and queued-command equilibrium shifts can be bounded by `m*g/(m0*Kp*b_min**2)*abs(delta_b)` and `m*g/b_min**2*abs(delta_b)` respectively. They cannot be omitted from the nonlinear/ISS proof.

Next proof step: validated Jacobian/nonlinear remainder bounds in the minimal positive-scalar quaternion chart, with nonzero 3-D state and measurement uncertainty, then an explicit invariant-region/measurement-gain calculation. The guarded backup must execute the exact observer-free fixed-heading law used by the proof. Calling the existing observer-biased mission conversion or smooth stop/hold function would change that law and invalidate this lemma's use as its terminal proof. No production guard certificate is enabled by these artifacts.

## Useful bounded-noise local-linear RPI certificate completed

`backup_linear_noise_exact_audit.json` adds an exact local-linear invariant-ellipsoid result with nonzero public measurement errors: position components at most .005 m and velocity components at most .01 m/s; orientation measurement is exact in this particular simulation service, while the plant's position/velocity/attitude/rate initial states have finite 3-D uncertainty. The latter distinction matters: this is not a point-state or vertical-only proof.

A cutting-plane SDP first proposes channel ellipsoids E(Q). The proof then uses exact rational Bernstein coefficient matrices, not the SDP's numerical status. It checks every joint (A,B) coefficient against

`Q - A*Q*A.T/tau - (B*e)*(B*e).T/(1-tau) >= 1e-10*I`, with tau=.99.

The two measurement columns are exactly collinear: position and velocity errors enter the same commanded acceleration component. Their worst aligned box corner dominates the opposite and mixed corners in the positive-semidefinite order. Exact Fraction LDL passes all 5,292 planar and 1,512 vertical coefficient matrices. The same state/noise Bernstein weights represent the actual joint (A,B) local map throughout the independently lifted continuous parameter box, so matrix convexity extends the inequality to parameter interiors.

To see invariance, for an arbitrary support direction l the image support is `sqrt(l.T*A*Q*A.T*l) + abs(l.T*B*e_worst)`. Weighted Cauchy-Schwarz bounds this by the square root of `l.T*(A*Q*A.T/tau + (B*e_worst)*(B*e_worst).T/(1-tau))*l`, which is at most `sqrt(l.T*Q*l)`. Thus the local linear image is inside E(Q) for every bounded input error. This is a genuine noisy local-linear RPI lemma, not merely an observed stable rollout.

Exact outward component radii are:

- Planar x and sign-similar planar y, ordered p,v,q,w,queue1,queue2: [.077783 m, .070442 m/s, .010529 quaternion-vector units, .197912 rad/s, .288661 rad/s, .326455 rad/s].
- Vertical, ordered p,v,motor-thrust,queue1-thrust,queue2-thrust: [.067302 m, .055462 m/s, .173839 N, .202997 N, .254497 N].
- The yaw channel retains the finite common-P ellipsoid in the local model; it has no first-order position/velocity measurement input. Nonlinear cross-coupling is still unclosed.

This does not establish nonlinear full-quadrotor invariance. The independent full real-map interval Hessian research currently fails the simpler product-P triangle ISS test even with zero noise at the tested radii. Signed-Hessian bounds retaining product-ellipsoid covariances reduce some remainder estimates but do not yet close the useful-noise nonlinear inequality. Those failed sufficient tests remain archived; they do not establish physical instability. Numerical execution/rounding and changing-authority equilibrium drift remain separate missing budgets.

The next precise proof target is a structured nonlinear extension: retain the new-command nonlinear remainder in the same held-command/queue channels as measurement noise, separately bound genuine physical-flow remainder, and use covariance-aware quadratic/LMI inclusion rather than treating all 21 output errors as independent. A useful real identity for a tighter controller graph is `R.T*((R*e3) cross (F/norm(F))) = e3 cross (R.T*F)/norm(F)` for unit quaternions: tilt feedback has exactly zero body-z component. This may remove interval dependency without changing the real controller; it does not remove the separate numerical reordering budget. A full coupled moving 3-D certificate test and runtime reservation mechanism must follow a completed nonlinear proof before the shield gate can pass.
