# Conditional bounded inputs and free-voxel witnesses

`ares.safety_contract` provides an exact numerical building block for externally
bounded public measurements. It does **not** provide a flight certificate,
controller bound, backup policy, reachable tube or terminal invariant set. Every
result contains `flight_certificate=False`; this field cannot be enabled through
the result constructor. The module is unused by the current flight environment
and shield. Existing sensor noise, maps, dynamics, checkpoints and gates are
unchanged.

## External supplier contract

`MeasurementContract` requires a supplier identifier, nonempty provenance,
`source_kind='hard_external'` and
`bound_basis='deterministic_guarantee'`. Measurements must use world coordinates
and SI units. Unsupported source/basis tags, including Gaussian, probabilistic,
RMS, confidence intervals and finite observed maxima, fail validation. Numerical
bounds and times must be finite; error bounds must be nonnegative.

These strings are a supplier's **conditional declaration**. The module cannot
verify supplier honesty, hardware calibration or the validity of an external
proof. Relabeling Gaussian noise as a hard guarantee does not make it one.
Innovation tests, residual RMS and finite experiments never manufacture the
required deterministic bounds. An external argument and enforcement mechanism
must justify them over the entire requested domain.

`capture_time_s <= time_s <= expires_s` is checked using exact rational
representations of the binary64 timestamps. Expiry is inclusive; any later
representable time is stale. Time before capture is rejected. A fresh contract
does not propagate a state estimate: `state_boxes` returns position and velocity
boxes **at capture epoch**, retaining that epoch in its result. A consumer needs
separate validated reachability from capture time to the present. Position and
velocity Euclidean error balls are conservatively enclosed by componentwise
boxes. Floating output endpoints round outward.

```python
from ares.safety_contract import MeasurementContract, state_boxes

contract = MeasurementContract(
    position_error_m=.005,
    velocity_error_mps=.01,
    world_ray_error=.003,
    depth_error_m=.02,
    capture_time_s=1.,
    expires_s=1.1,
    supplier='external bounded measurement supplier',
    provenance='identifier for an independently justified deterministic guarantee',
    source_kind='hard_external',
    bound_basis='deterministic_guarantee',
    voxel_model='homogeneous_static_until_expiry',
    depth_model='first_hit_range',
)
boxes = state_boxes([1., 2., 3.], [.1, 0., 0.], contract, time_s=1.05)
assert boxes.conditional_input_valid
assert boxes.capture_time_s == 1.
assert not boxes.flight_certificate
```

## Robust homogeneous-voxel crossing

`robust_free_witness` accepts public world position `p`, a world ray direction
`d`, measured first-hit depth, a voxel's world lower/upper coordinates, and a
common positive range parameter `s`. It accepts the complete **world direction
error bound directly**. Calibration, pose, rotation and timestamp effects must
already be included in the supplier's bound. The implementation performs no
trigonometry, rotation or direction normalization.

The ray need not be unit length. Measured depth and `s` must use the same range
parameter along `p+s*d`; they are Euclidean ray lengths only if the external
supplier defines a unit direction. `depth_error_m` bounds error in that range
parameter. `world_ray_error` is a Euclidean bound on direction error and is
dimensionless when `s` is in metres.

Let `ep` be the position bound, `ed` the complete world direction bound, and `ez`
the first-hit range error. For a fixed `s`, every admissible actual ray point is
inside the componentwise box

```
[p_i + s*d_i - (ep+s*ed), p_i + s*d_i + (ep+s*ed)].
```

The witness requires both:

1. `0 < s < measured_depth-ez`, strictly before every possible first hit.
2. This entire box lies strictly inside the queried voxel on all three axes.

Contact with any voxel face or the conservative first-hit endpoint is rejected.
A sufficient crossing need not exist for every free voxel; rejection leaves it
unknown.

Inferring the **whole voxel** is free additionally requires the explicit external
model `voxel_model='homogeneous_static_until_expiry'`: each queried voxel is
uniformly free or occupied, and that classification remains fixed through
expiry. It also requires `depth_model='first_hit_range'`: depth bounds the first
obstacle on the admissible actual ray, without unmodelled holes, false negatives
or occluders. A ray crossing a partly occupied real-world voxel does not establish
that voxel is free. Missing homogeneity or first-hit semantics fails closed.

Under those assumptions, every admissible physical interpretation has an
obstacle-free point inside the voxel before its first ray hit. Homogeneity rules
out an occupied voxel in each interpretation, so the whole voxel is conditionally
free. This argument handles three dimensions and uncertain ray/position sets;
it uses no scene, collision mesh, occupancy oracle or simulator state.

```python
from ares.safety_contract import robust_free_witness

witness = robust_free_witness(
    position=[.5, .5, .5], world_direction=[1., .2, -.1],
    measured_depth=2., voxel_lower=[1.4, .6, .3], voxel_upper=[1.6, .8, .5],
    optical_range=1., contract=contract, time_s=1.05,
)
assert witness.conditional_input_valid
assert witness.free_witness
assert not witness.flight_certificate
```

## Exact arithmetic and rejection behavior

Each numeric input is first interpreted as a finite binary64 value. All
freshness, depth, box and strict-interior decisions then use
`Fraction.from_float`, preserving that value exactly. Intermediate products,
sums, uncertainty growth and slack are not rounded to binary64. Output bounds
are converted back only after the decision and advanced with `math.nextafter`
as necessary to enclose the exact rational endpoints. No unproved epsilon,
rotation-norm bound or floating GPU kernel is involved in these decisions.

This is exact for the represented binary64 inputs, not for an intended decimal
number before conversion. The supplier's error contract must cover any prior
representation, transport or preprocessing uncertainty. A tiny positive exact
slack can round to zero in the reported outward lower slack; consumers must use
the exact decision flag rather than reclassifying rounded diagnostics.

Malformed bounds, missing provenance, nonfinite vectors/times and nonpositive
voxel dimensions raise `ValueError`. Unsupported declarations, stale/future
contracts and missing map semantics return `conditional_input_valid=False`.
Unavailable/nonfinite depth or range returns invalid flags. A valid observation
whose range or uncertainty box has no strict crossing returns valid input with
`free_witness=False`. If finite outward endpoints cannot be represented, the
operation fails closed with `nonfinite_outward_bounds`. No path sets a flight
certificate or substitutes a missing backup invariant proof.

## Verification

Run `.venv/bin/python -m unittest tests.test_safety_contract -v`.
The fixtures cover a positive 3D crossing and all component-error corners,
first-hit and voxel-face contact, exact timestamp boundaries, capture-epoch state
boxes, unsupported Gaussian/RMS/statistical sources, missing source semantics,
nonfinite and overflow inputs, outward endpoints, and 150 comparisons with an
independent rational reference. A binary64 cancellation adversary that a naive
floating calculation accepts is correctly rejected as exact voxel-face contact.
These tests verify this input building block; they do not close the shield's
full-motion or terminal-invariance gate.
