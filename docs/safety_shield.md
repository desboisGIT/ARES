# Phase 4 observed-map safety shield

`ares.safety.SafetyShield` is a NumPy-only onboard component. It accepts the
mapper's **xyz** grid (`0` unknown, `1` observed free, `2` occupied), position,
velocity and a proposed **net world-frame acceleration in m/s²**. It never reads
a scene, collision mesh or simulator truth. Unknown cells and the area outside
the supplied grid are blocked. Resolution and origin are explicit, in metres.

The flight policy interface remains collective thrust in newtons and body rates
in radians/second. The integration layer must obtain the proposed net acceleration
from its dynamics/controller model, run the shield, and convert an override's
desired acceleration back through the low-level attitude/thrust controller. Add
gravity when converting net acceleration to thrust. `safe_acceleration` is not a
body-rate command. The returned `emergency` flag requests active velocity braking;
the controller must honor the configured active-braking and terminal-hold
contract described below. The default contract continuously opposes actual
velocity until stopped, without reversal. The opt-in terminal contract permits
bounded corrective motion near rest. A zero-thrust, zero-rate, or hover command
does not replace braking a moving vehicle.

```python
from ares.safety import SafetyShield, ShieldConfig

shield = SafetyShield(ShieldConfig(
    body_radius_m=.15,
    position_uncertainty_m=.03,
    velocity_uncertainty_mps=.05,
    control_interval_s=.02,
    # Example only: measured compute/transport + motor + attitude reaction bounds.
    latency_s=.02 + .04 + .20,
    max_acceleration_mps2=5.,
    min_braking_acceleration_mps2=1.,
    max_speed_mps=4.,
))
decision = shield.filter(
    observed_grid, position, velocity, proposed_net_acceleration,
    origin=map_origin, voxel_size=map_resolution,
    assumptions_valid=validated_operating_envelope,
)
```

The example numbers are configurable assumptions, not measured airframe limits.
Include every compute, scheduling, sensor-age, transport, motor and attitude
reaction delay in the latency upper bound, or account for map/state age separately
with conservative estimation bounds. A caller that has not validated these bounds
must pass `assumptions_valid=False` and report uncertified operation. An arbitrary
quadrotor attitude, tumble, motor failure or tracking transient cannot inherit this
certificate from a point-mass calculation.

## What is checked

The shield first checks whether an immediate emergency stop has a known-free
swept envelope. It then tries the proposal and three blends toward braking
(fractions `1, .75, .5, .25`). Each candidate must fit the acceleration bound,
remain below the speed limit, and retain a complete stopping envelope after one
control period. When none passes, the shield requests the immediate braking
controller. It never substitutes hover for braking a vehicle moving toward a wall.
The finite candidate set is conservative; a rejected state may have a safe curved
escape that this shield does not search.

Every swept segment includes physical body radius, position error, velocity error,
and the appropriate latency/tracking reachability inflation. Segment-versus-voxel
slab intersection checks the full segment, including both endpoints and contact.
Occupied and unknown voxels are inflated by a cube enclosing the required sphere.
This deliberately overestimates corner clearance and prevents thin-wall tunnelling
and diagonal or vertical corner grazing. No sampled-path collision check is used.
Only voxels in the segment's bounding box are considered after map validation.

`ShieldDecision` contains:

- `safe_acceleration`: selected desired net acceleration; braking if necessary.
- `accepted`: the original proposal passed unchanged.
- `certified`: this selected candidate or immediate braking envelope passed **under
  all assumptions below**. This does not certify the applied rotor command.
- `state_viable`: the current immediate braking geometry and speed bound fit.
  This is a geometric/model property, independent of `assumptions_valid`.
- `already_unsafe`: the current state has no certificate for this braking
  envelope. This does **not** prove that all evasive trajectories collide.
- `emergency`: invoke the active braking controller, including when uncertified.
- `reason`: `accepted`, `limited`, `emergency_brake`,
  `no_braking_certificate`, or `assumptions_invalid`.
- `stopping_distance_m`: conservative radial travel bound for the immediate stop,
  including position/velocity uncertainty, excluding the physical radius.
- `checked_segments`: center-segment endpoints and inflated radii for the selected
  certificate, or for the failed immediate stop. These support inspection/logging.
- `candidate_fraction`: fraction of the proposal in the selected acceleration;
  zero for immediate emergency braking.

A currently overspeed vehicle, a body already touching unknown space, and a wall
closer than the stopping distance return `certified=False`. A physically unavoidable
collision must be counted separately from a preventable failure; neither successful
braking nor collision avoidance is promised from these initial states. Failure of
this conservative envelope alone is insufficient evidence that a collision was
physically unavoidable. Evaluation must retain its initial-state and assumption
status instead of relabeling every failure as unavoidable.

## Conditional model and derivation

The assumptions are all required:

1. Free voxels are truly free and remain so throughout the full stopping horizon.
   Moving obstacles, wrong map labels and stale geometry need an external bounded
   model or invalidate this assumption.
2. The body fits within `body_radius_m`. Initial Euclidean position and velocity
   errors are at most their configured radii.
3. During each activation delay `L`, arbitrary net acceleration has Euclidean norm
   at most `A = max_acceleration_mps2`. The controller can begin the next requested
   behavior within this delay; latency is an upper bound, not an average.
4. During an accepted control interval `H`, the requested constant net acceleration
   is tracked with acceleration-error norm at most `E`. Its requested norm plus
   `E` must be at most `A`. There is no additional unmodelled velocity jump.
5. After the braking activation delay, the emergency controller follows a straight
   path along the **actual** velocity and decreases speed by at least
   `b = min_braking_acceleration_mps2`. With the default
   `terminal_speed_mps=0`, this must continue until zero speed without reversal.
   With positive `terminal_speed_mps=v_terminal` and
   `terminal_stop_radius_m=R`, this active guarantee is required only until speed
   first reaches `v_terminal` or lower. Thereafter the entire trajectory must
   stay within Euclidean distance `R` of that terminal-entry position, and speed
   must remain below `max_speed_mps`. Corrective reversal and lateral position
   hold are permitted within that terminal ball. If a hold anchor differs from
   terminal entry, `R` must include both the entry-to-anchor offset and all later
   tracking excursions; it cannot simply be the hold error about a moving anchor.
   All net forces, attitude/thrust limits and tracking errors must already permit
   the applicable guarantee. `b` is a certified lower bound, never peak thrust.
6. The shield is called within `H` or the caller starts the reserved braking
   controller. If a new check or assumption fails, braking begins within the
   reserved activation delay. The caller must preserve the emergency mode.

For nominal position `p`, velocity `v`, position uncertainty `ep`, and velocity
uncertainty `ev`, an activation delay has nominal segment `p -> p+vL`. Its extra
radius is `ep + ev L + A L²/2`; at its end the error bounds become
`ep' = ep + ev L + A L²/2`, `ev' = ev + A L`. This includes worst-case speed growth
and lateral acceleration during motor/attitude lag, rather than assuming that the
vehicle coasts or is already braking during latency.

A constant commanded acceleration `a` has endpoint `p+vH+aH²/2`. Its parabolic
path differs from this chord by at most `|a| H²/8`. Inflate by this amount and
`ep + ev H + E H²/2`, then propagate the state error bounds. A candidate reserves
another full braking activation delay after this interval. The two delay tubes are
intentionally conservative: the first covers action activation; the second covers
an immediate fallback after that action.

For a braking start velocity `u` with `|u-v| <= ev`, a straight minimum-rate stop
has displacement `u alpha`, with
`0 <= alpha <= (|v|+ev)/(2b)`. Thus the whole braking path lies around the nominal
segment `p -> p+v alpha_max` with extra radius `ep + ev alpha_max`. Larger actual
braking acceleration only shortens this path. Maximum speed is bounded throughout
by the nominal endpoint speed plus velocity-error radius.

In the one-dimensional, zero-estimation-error case, the immediate stopping bound
is

```
d_stop = v L + (A L²)/2 + (v + A L)² / (2 b).
```

With `A=0` this reduces to `v L + v²/(2b)`. Omitting the latency speed growth is
unsafe when the old command or actuator lag can continue accelerating the vehicle.
The implemented 3D tubes are more conservative than this scalar distance: a scalar
forward ray cannot exclude a sideways obstacle.

### Optional terminal stopping reserve

`SAFETY_VERSION` is `ares.safety.v2`. Both terminal parameters default to zero,
which preserves the existing envelope, stopping distances and emergency requests.
Positive terminal speed requires a positive terminal radius. A positive radius
with zero terminal speed only adds conservative geometric reserve to the legacy
straight-to-rest contract.

The full straight-stop tube derived above contains every permitted active prefix:
ending active braking early at a nonzero terminal speed reaches a point on that
tube. Every subsequent terminal position is within `R` of this entry point.
Therefore Minkowski-expanding the original tube by a radius-`R` ball contains both
the active prefix and its entire terminal trajectory. The implementation adds
`R` to **every** checked segment radius, including activation and candidate
segments, and adds `R` to the reported stopping-distance bound. Inflating all
segments deliberately overcovers the necessary set. It retains the full
straight-stop center segment instead of shortening it at the terminal threshold;
this can reject some safe states but cannot omit the reserved terminal motion.
Physical body radius and estimation uncertainty are still additional terms.

The nominal one-dimensional distance expression consequently becomes

```
d_stop = v L + (A L²)/2 + (v + A L)² / (2 b) + R.
```

For opt-in terminal operation, the shield's fallback acceleration request is
`-v * min(b / v_terminal, 1 / H)`, capped to norm `b`. This approaches zero
continuously at rest and cannot reverse an ideal velocity in a single Euler
control period. With both terminal parameters zero, the original constant-norm
braking request remains unchanged. Request shaping does not itself prove actual
minimum-rate braking, latency, terminal radius or speed bounds. In particular,
a smooth physical stop controller must have its active and terminal phases
validated together before its caller sets `assumptions_valid=True`.

## Validation and limits

Run `.venv/bin/python -m unittest tests.test_safety -v`. The analytic fixtures cover
unknown one-voxel slabs, occupied thin walls, side obstacles during acceleration
latency, diagonal physical-radius grazing, vertical motion, out-of-map stops,
uncertainty inflation, command and speed bounds, a wall already too close for even
ideal braking, invalid controller assumptions, and exact voxel-face contact.
Independent constant-lag-acceleration trajectories with subsequent minimum-rate
stops are also checked against the continuous envelope. Terminal-contract tests
reject an unknown slab reached only by reserved terminal motion, verify unchanged
zero-default behavior and continuous near-zero commands, and independently check
a straight braking prefix plus a reversing three-dimensional terminal curve
inside the expanded capsule. These are mathematical fixtures, not evidence that
a physical or randomized simulated controller satisfies the configured bounds. Tests use no ground-truth
scene API.

These tests establish the implemented model's behavior, not Phase 4's randomized
flight gate. Real or simulated thrust/body-rate execution needs separate validation
of the assumptions and collision outcomes. An episode cannot claim the shield's
conditional guarantee while running outside its latency, attitude, acceleration,
estimation, mapping or braking envelope. No actual flight device is controlled by
this module.
