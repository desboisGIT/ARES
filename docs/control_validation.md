# Independent controller and coupled stopping regression

`ares.control_validation.run_control_validation(profile, device='cpu', count=8,
seed=0, output_dir=None, ...)` exercises the actual DiffAero full rigid body v2
without constructing a learned policy. The controller sees noisy public position,
velocity and quaternion, issued commands and its own observer memory. Sampled mass,
gravity, inertia, drag and motor lag enter the plant only.

```python
from ares.control_validation import run_control_validation
report = run_control_validation(
    'ares/data/configs/drone_sim_v2.json', count=64, seed=17,
    device='cuda', batch_size=64, duration_s=4., settle_s=2.,
    compile_dynamics=True, command_delay_s=.01,
    noise_bounds={'position_m': .005, 'velocity_mps': .01},
    output_dir='/tmp/ares-control-validation')
```

CPU defaults are intentionally small. Increase count to 1,000 for a broader seeded
stress run. Dynamics and controller tensor operations are batched in groups of 64;
CUDA dynamics compilation is optional. The actual Python/NumPy shield and exact
collision checks remain in the loop, so this is not a fully compiled controller
rollout and does not establish real-time scheduling. CUDA collision requires the
same existing CuPy dependency as `FlightCollisionChecker`.

The first eight cases cover hover, low speed hold, upward/downward vertical
braking, horizontal braking, 3-D braking, an occupied wall and an observed unknown
slab. The first eight use every profile parameter's low endpoint; the next eight
use every high endpoint. Four further fixed low-mass, low-altitude placements
exercise the old nominal-weight failure mechanism on independent analytic
geometry. They are explicitly **counterexample families**, not reproductions of
held-out scene 60000's four starts. Later cases use parameter-seeded random
profiles and fixed-seed position/speed perturbations. This covers endpoints and
interiors, not every combination of endpoints. No held-out scene is loaded.

The analytic room is 12 m per side with occupied voxel walls. The unknown slab is
an observed-map restriction at x=[9,9.2] m, with free physical space behind it.
Physical sweeps use the actual exact `FlightCollisionChecker`; slab crossing uses
the maximum x of each complete straight sweep plus the sampled physical radius,
which is exact for this full-span slab. Unknown entry is a separate failure from
occupied collision. The shield receives only its observed ternary map.

Each run includes initial hover acquisition, followed by a declared external
velocity perturbation to establish the braking initial condition. Observer trim
survives that perturbation; its velocity state is synchronized to the newly
measured public velocity. Acquisition is part of the collision/failure gate and
has its own maximum error and saved trace. It is not silently excluded from
results. Hover acquisition is useful to separate disturbance-estimation startup
from braking response; it is not evidence about unobserved in-flight recovery.

Transport queues apply sampled compute latency plus `command_delay_s`, rounded
up to the next .01 s interval. The actuator integrates motor lag with at least
500 Hz rate-loop updates. Position and velocity noise are independent uniform
componentwise draws within the explicit bounds. Their seed, full sampled profile,
transport age, motor time constant, software source hashes and configuration are
recorded. Command age here is simulated transport time; wall-clock throughput is
reported separately and is not onboard sensor-to-actuator latency.

The wall and unknown-slab families use the actual shield every .1 s. Its emergency
command goes through the controller's smooth bounded stop/hold function each
physical step. Other families test controller stop/hold directly. All decisions
pass `assumptions_valid=False`; certificate claims remain false. This intentionally
separates empirical behavior from the still-unproved acceleration-error,
reaction-time and terminal hold bounds.

The report contains requested and achieved acceleration, acceleration error,
minimum projected braking above .25 m/s after transport plus three motor time
constants, stopping distance, reverse overshoot, final displacement, hold-anchor
error, peak tilt/speed, clearance lower bounds and terminal speed. Reaction and
near-stop exclusions apply only to the named braking metric; their trajectory
samples and failures remain in the report. Reverse overshoot means displacement
opposite the prescribed initial velocity. Final displacement includes the normal
stopping distance; hold-anchor error measures terminal holding separately.

Every case remains in the report. Declared initial overlap, overspeed and
insufficient stopping space have separate initial classifications. The report
also counts failures among `valid_brakable` cases. An unexpected acquisition error
above .3 m, occupied collision, out-of-bounds excursion, unknown entry or final
speed above .25 m/s marks a failure. These are regression gates, not a proof of
safe motion or validation of the shield's universal assumptions.

`control_validation.json` contains aggregate counts and all case records. Every
failure and one representative of each initial family retains its full sampled
CPU-list trajectory and acquisition trace with column names. A frozen old
nominal .5 kg mapper runs the same cases by default, providing a matched baseline
for the vertical mass/gravity counterexample. Disable with `include_nominal=False`
for throughput runs. The baseline is a fixed geometric mapper, with no neural
network parameters or learned residual.

The terminal envelope is fixed before running cases: terminal entry speed is
.4 m/s, with every later position required to remain within .2 m of the first
entry position. The measured tail includes corrective hold motion through the
end of the run. Exceeding that radius marks `terminal_contract_failed` and fails
the case; the radius is not fitted to the observed results. This is a finite
horizon regression gate and says nothing about drift after the recorded horizon.

`fixed_latency_min_active_braking_mps2` measures projected braking after a global
.4 s cutoff while speed exceeds .4 m/s. `fixed_latency_max_tracking_error_mps2`
uses the same .4 s cutoff. Each metric carries its eligible sample count and
returns null if there are no eligible samples. The older per-airframe reaction
metric uses privileged sampled motor lag and is explicitly a diagnostic; it is
not a runtime latency bound. All earlier trajectory samples remain recorded and
all failures remain eligible for the aggregate regression gates.

`empirical_bounds` reduces **all valid repaired cases, including failures** into
maximum acquisition drift, maximum terminal radius, maximum tracking error after
the fixed latency and minimum measured active braking. Case/sample counts accompany
the reductions. The report marks these finite empirical observations and never
universal certification. `valid_gate_passed` requires a nonempty valid population
and zero valid failures; separate counts retain every invalid initial condition.

The shield body radius uses the profile range maximum. The configured uniform
componentwise observation noise must have a worst-case vector norm within the
shield position and velocity uncertainty assumptions, or the call is rejected.
Software hashes are captured before dynamics and checked again before returning
or saving results. A source change during the run raises an error instead of
writing a report with mismatched provenance.

An optional `initial_tilt_deg=0.` argument adds a separate attitude-recovery
extension. Values must lie in [0,20] degrees. A case-local deterministic random
stream selects signed roll and pitch components, each of magnitude the requested
tilt divided by √2; their combined body-axis tilt stays within the requested
limit. The report records both components, actual combined tilt and the imposed
xyzw quaternion. Yaw is zero and body rates are set to zero at the imposed
post-acquisition velocity/quaternion perturbation. The controller receives the
public quaternion observation, and its previous-quaternion memory is synchronized
to that observation. No sampled airframe parameter enters this initialization.

The default zero mode preserves the original protocol: it adds no quaternion or
angular-rate perturbation and retains the acquired attitude. The independent
tilt stream leaves profile, position, velocity and invalid-stratum sampling
unchanged, so the same seed supports matched level/tilted extension comparisons.
These are separate recorded experiments; a level-only passing run does not
establish the controller's attitude-recovery envelope.
