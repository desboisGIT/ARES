# Bounded-service SIL fixture v1

This isolated reusable research supplier produces nonzero deterministic bounded
measurements and a known-free voxel map from first-hit rays. Every output says
`conditional_fixture_only=true` / `flight_certificate=false`. It implements no
full-state safety shield, backup proof or flight certificate.

From the repository root:

```bash
.venv/bin/python reports/phase_4_gate_artifacts/shield_research/bounded_service_v1/run_fixture.py --seed 71
```

The command regenerates artifacts only in this directory, runs the tests and
hashes every `ares/**/*.py` file before/after. The complete v2 profile is retained
in `full_profile_config.json`; Gaussian benchmark sensors, runtime sources,
checkpoints and evaluation cohorts are unchanged. Tests use low/high/random full
profiles with the actual DiffAero rigid-body model, on CPU only.

## Public boundary and bounded errors

`BoundedSupplier.capture` returns only measured p/v/q/w, a measurement contract
and the explicit body-rate bound. The packet contains no sampled mass, gravity,
inertia, motor state, thrust, energy or true-state fields. Its p/v errors have
Euclidean bounds .005 m / .01 m/s. The separate body-rate fixture extra has a
.001 rad/s Euclidean bound. `boxes(time_s=...)` and
`body_rate_box(time_s=...)` return capture-epoch boxes with freshness checks.
Exact q measurement matches the existing Phase4 state-feedback abstraction; it
does not prohibit a future initial-state set from containing a nonzero range of
quaternions. Neither angular uncertainty nor motor-state uncertainty is proved
by this fixture.

Each deterministic error component is a signed rational `E*m/3`, with
`0 < m <= 1`. Therefore the ideal squared vector error is at most `E²/3 < E²`.
After rounding the measured values to binary64, the supplier checks the **actual
represented** squared error against `E²` using exact fractions and fails closed
if the bound cannot be enforced. Depth error is signed `E/3`, likewise checked
exactly after binary64 conversion. These are synthetic enforced guarantees,
not estimates inferred from RMS, Gaussian samples or observed maxima.

`capture_time_s` and inclusive `expires_s` are explicit (generated example:
0 and 30 s). All scan rays share an instantaneous snapshot epoch; a moving SIL
loop must create a supplier from each new renderer snapshot. Freshness never
propagates capture-time state boxes to the present. Occupancy remains static
through expiry. The supplier identity/provenance is a conditional declaration,
not cryptographic authentication.

`AnalyticRenderer` and `TestOracle` are the only truth boundary. The former
receives the physical state and analytic environment. Map and reservation
consumers receive returned packet values only. The renderer/guard API boundary
is enforced by code separation and tests, not a hostile-process sandbox.

## Map construction

The analytic room is 12×12×4 m, with homogeneous static .25 m voxels. Its outer
wall voxels and x=[9,9.25] m slab are physically occupied. Exact rational slab
intersection computes first-hit range along a possibly nonunit actual world
ray. Actual and reported directions differ nontrivially with Euclidean error
at most .001. The first-hit range-parameter error is at most .01; it is not
implicitly Euclidean arc-length error for nonunit rays.

`preflight_scan` enumerates public candidate voxel centers. For each target it
requests one bounded ray and calls the existing exact
`ares.safety_contract.robust_free_witness`. The map never reads a truth volume,
renderer occupancy or sampled plant values. Only a successful strict-interior,
strictly-before-first-hit witness marks a voxel free. Every free label retains
its public ray packet, range parameter and witness. No failed ray is relabeled
occupied: failure leaves that voxel unknown.

The x=[7,7.25] m target slab is deliberately unscanned and remains unknown.
Targets in boundary walls, the interior occupied slab and behind that slab are
queried and rejected. `TestOracle` separately verifies that no physical occupied
voxel received a free label; that oracle does not build or correct the map.

The default seed yields 2,028 witnessed free cells and a free 27-cell
neighborhood around the measured anchor. `reservation` tests a static sphere
with radius .17 m (full profile maximum body radius) + .15 m margin + .005 m
position uncertainty. It rejects reservations touching unknown/occupied targets
or using expired evidence. This is only a static capture-epoch ball reservation.
It certifies neither a reachable motion tube nor terminal invariance.

`WitnessMap.from_dict(json_data, time_s=...)` reloads `known_free_map.json` by
recomputing every witness from the retained public ray packet; it never trusts
a serialized free flag. Expired packets yield no free cells.

## Verification artifacts

`fixture_report.json`, `state_packet.json`, `known_free_map.json` and `tests.log`
record the seed, bounds, epochs, full-profile hash, fixture/source hashes and
outcomes. Twelve tests cover:

- 1,024 signed p/v/ray/depth corner combinations, including all body-rate sign corners;
- nonzero deterministic replayable Euclidean-bounded state and ray measurements;
- expiry/future epochs, Gaussian/statistical/unsupported contract rejection;
- ray-derived map validity and static reservation with unknown/occupied rejection;
- map reload with witness rechecking and expiry rejection;
- closed left-face one-point contact with unknown, occupied and out-of-support cells;
- malformed packet/body-rate/quaternion/grid metadata and outward-box overflow rejection;
- eight physical p/v/q/w variations advanced with full-profile DiffAero rigid-body dynamics.

Finite corner/plant tests complement the exact construction checks; they do not
establish a universal 13D control guarantee or close any flight safety gate.

## Closed geometry and input validation

Reservation enumeration includes every closed voxel intersecting the ball's
axis-aligned bounding box: the lower index is
`ceil((center-radius-origin)/resolution)-1`, and the upper index is
`floor((center+radius-origin)/resolution)`. Exact rational squared distance to
each closed voxel then tests `distance² <= radius²`. Thus a ball whose leftmost
point lies exactly on a voxel face must also reserve the cell on the left.
Tests require rejection when that one-point contact is unknown, occupied or
outside public support.

Packets validate finite numeric p/v/w coordinates, w width three, a finite
nonnegative non-boolean angular-rate error, and a finite nonzero four-coordinate
quaternion. Quaternion norm must lie within eight binary32 unit roundoffs of
one; no normalization changes the supplied exact feedback value. Both q and -q
are accepted, so a positive scalar component is not required. Unrepresentable
body-rate box endpoints fail closed. Deserialization applies the same checks.

Grid support must have three positive integer dimensions, finite positive numeric
resolution, and a finite three-coordinate origin with distinguishable finite
voxel boundaries. For non-dyadic arithmetic, witness queries use inward-rounded
sub-boxes of exact represented grid cells; reservation geometry uses the exact
represented cells. This prevents boundary rounding from enlarging a witness
voxel into its neighbor.

## Exact rational proof-bound representation

The state proof constants are exactly 1/200 m, 1/100 m/s and 1/1000 rad/s
per axis. The nearest binary64 values of decimal .005/.01/.001 lie slightly
above those rationals. Accordingly, the supplier advertises the next
representable value toward zero for each of these three state radii and
generates its errors using exactly that advertised radius. Exact Fraction
comparisons verify that every advertised Euclidean radius lies below its
rational proof constant; therefore each component error also lies below that
constant. Neither runtime checks nor tests add an epsilon. Ray/depth bounds
retain their existing represented geometric-contract semantics. This changes
representation binding, not the deterministic sensor-noise assumption.

## Orientation service tag

Every packet explicitly carries
`orientation_contract='exact_current_raw_quaternion'`. The raw quaternion is
exact at the same capture epoch as p/v/w under this SIL supplier; the service
does not normalize it, choose its sign or require positive qw. Both q and -q
remain supported. A local guard may independently reject states outside its
positive quaternion chart. Unsupported tags reject on packet construction; missing or unsupported tags
reject during map deserialization. This tag describes the current simulated state-feedback
abstraction and makes no guarantee about a future physical provider.
