# Observed-state acceleration and stop control

`ares.flight_control` converts requested **net world-frame acceleration in m/s²**
to collective thrust in **N** and body rates in **rad/s**. It is a Torch batched
CPU/CUDA controller. It accepts public position, velocity, xyzw orientation and
command history. Sampled mass, inertia, drag, gravity, motor state, battery state
and transport latency are not controller inputs. Nominal constants are declared
in `ControllerConfig` and remain identical across randomized episodes.

The observer estimates acceleration disturbance, including hover trim error,
from a filtered velocity innovation. If a light vehicle produces upward
acceleration under nominal hover thrust, the estimated vertical disturbance
becomes positive and action conversion subtracts it. This fixes the case where
a nominal `-1 m/s²` request produces nearly the light vehicle's actual hover
thrust. It does not identify or read the sampled mass.

The prediction integrates public commanded thrust through two nominal first-order
states, approximating transport delay and motor lag. It averages endpoint
observed thrust directions, predicts velocity, and corrects predicted velocity
and acceleration disturbance with a Luenberger innovation. It never computes a
finite difference of noisy measured velocities. The disturbance correction is
bounded in Euclidean norm. The observer predicts from bounded physical commands,
not the requested acceleration; it therefore has no requested-acceleration
integrator accumulating error behind a saturated thrust command. This bounded
trim adaptation cannot compensate arbitrary authority loss or extreme attitudes.

## Development controller configuration

The current explicit development candidate uses
`ControllerConfig(observer_bandwidth=10., max_yaw_rate=1.)`. Other observer and
hold gains remain unchanged. Completed earlier experiments are archived with
their runtime wheels; their results describe their original configurations.
Explicit bandwidth and yaw limits permit controlled component comparisons; the
archived wheel is the source for complete earlier-experiment reproduction. The
legacy nominal-v1 policy conversion remains unchanged.

In a finite development ablation, bandwidth 10 reduced maximum cold-hover sag
from about .204 m to .090 m across mass/gravity/drag corners, motor lag .04 s,
command transport .02 s, and control periods .01 s and 1/60 s. Public velocity
and position noise norms were bounded by .025 m/s and .01 m, respectively.
Stationary actual-acceleration RMS increased from at most .025 to .039 m/s²;
settled acceleration tracking error and terminal radius decreased. No growing
oscillation appeared in those 18-second trajectories. These observations motivate
the candidate and focused regression fixture; they do not establish worst-case
bounds, a shield certificate, physical flight safety or generalization to the
full noisy environment. Controller configuration must be retained with each
experiment, and new evaluation must use a frozen configuration.

## Heading contribution and tilt feedback

The heading error from public orientation, position and gaze produces a scalar
world-vertical heading rate. `max_yaw_rate`, in rad/s, caps this scalar before
transformation into body coordinates. Geometric tilt feedback retains its gain,
and the combined body rates still pass through the physical actuator bounds.
The cap applies to the heading contribution; tilt feedback may itself have a
world-vertical component. A positive finite explicit limit is required.

The 1 rad/s development candidate resolved all three collisions in an isolated
zero-residual validation replay: 128/128 tasks succeeded, compared with 125/128
for the preceding uncapped-heading mapping. Scene/task/profile/noise inputs and
full CUDA sensor processing were identical. This finite developmental comparison
motivates a recorded candidate; it does not establish unseen-scene performance,
a tracking guarantee, learned-policy performance or a flight certificate.

A focused actual rigid-body fixture starts pitched .15 rad, turns toward an
opposite gaze and tracks public target velocity [0, 1, 0] m/s across mass/inertia
corners, motor lag .04 s and transport .02 s. The cap reduces peak unwanted
horizontal and vertical speeds and peak thrust-direction error in that fixture.
A separate constant-acceleration prototype showed increased integrated lateral
speed while yawing more slowly under sustained tilt. Individual axis errors
therefore need separate evaluation; no general improvement is claimed.

For a comparison with the former adaptive heading law at nominal attitude gain
6, a large explicit finite cap such as `max_yaw_rate=100.` leaves all possible
heading requests below that limit. Nominal-v1 policies continue using their
existing action conversion branch.

## Explicit state and repeatable conversion

```python
from ares.flight_control import (
    initialize_memory, update_memory, acceleration_to_action, stop_acceleration,
)

# obs contains only public measurements p, v, q, and optional gaze.
memory = initialize_memory(obs)
obs.update(memory.observation_state())

# Conversion is pure: repeated evaluation does not change the observer.
action = acceleration_to_action(obs, requested_acceleration)
next_obs = plant_step_and_measure(action)
memory = update_memory(memory, next_obs, action, elapsed_s)
next_obs.update(memory.observation_state())
```

`previous_action` in `update_memory` means the SI command sent over the interval
ending at the supplied observation. It must not be replaced by a plant's sampled
initial hover thrust or actual applied motor thrust. Initialize after the first
public observation, and advance exactly once per environment transition. A
retained observation contains cloned, detached observer snapshots, so minibatch
reordering and PPO repeated evaluation see the same conversion state. Conversion
is differentiable with respect to requested acceleration and public state; the
online observer update is detached from the rollout graph.

`update_memory(..., reset_mask=mask)` initializes every observer and stop tensor
in selected lanes. A zero interval is accepted only for a masked reset and leaves
all other lanes exactly unchanged. `initialize_memory` always starts bias at zero
and actuator estimates at fixed nominal weight. `ControllerMemory` is a frozen
dataclass; tests may use `dataclasses.replace` to synchronize its measured velocity
after an explicit observable test-state injection while preserving learned trim.

The observation snapshot keys are `controller_bias`, `controller_velocity`,
`controller_hold_anchor`, `controller_holding`, and `controller_stopping`.
The initial snapshot is sufficient for conversion before the first update.
Without a bias snapshot, conversion uses zero disturbance compensation.

## Braking and position hold

```python
obs.update(memory.observation_state())
requested = stop_acceleration(obs, max_acceleration=2.)
action = acceleration_to_action(obs, requested)
next_obs = plant_step_and_measure(action)
memory = update_memory(memory, next_obs, action, elapsed_s, stop_requested=True)
```

Above the stopping band, acceleration opposes filtered measured velocity with a
bounded proportional command. Its magnitude smoothly decreases toward zero,
avoiding noisy unit-vector sign changes near rest. When filtered speed falls
below `hold_enter_speed`, the observer captures the current measured position.
Position and velocity feedback hold that anchor. The hold remains latched until
speed exceeds the larger `hold_exit_speed`; a subsequent low-speed acquisition
captures a fresh anchor. `stop_requested=False` releases the mode;
`stop_requested=None` preserves its current request. The mode mask can be a scalar
or one Boolean per environment.

This controller's smooth approach and position hold can make small corrective
velocity reversals. It does **not** establish the shield's default straight,
minimum-rate, no-reversal braking assumption. Its low-speed behavior may instead
be evaluated against the opt-in bounded terminal-stop contract, whose active
braking and terminal-radius bounds still require separate validation. A configured acceleration bound
limits requested net acceleration, not actual plant acceleration or tracking
error. Shield certification remains conditional on separately validated tracking,
latency, state error and stopping bounds. The simulation results do not certify a
physical airframe, arbitrary initial attitude, motor failure or unseen profile.

## Verification

Run `.venv/bin/python -m unittest tests.test_flight_control -v`.
Tests cover finite gradients including vertical gaze, deterministic conversion,
physical action limits, bounded observer trim, detached snapshots, isolated masked
resets, smooth low-speed commands, capture mode release and a capped world
heading contribution with preserved tilt feedback. Actual
`ares.diffaero.v2` trajectories exercise low/high profile endpoints and mixed
mass/drag/lag corners: observed hover acquisition, bounded-noise cold-hover
acquisition with .02 s transport latency, a subsequent `-1 m/s²` vertical
command, and coupled horizontal/vertical braking followed by anchor hold. A CUDA
64-lane observer/conversion test runs when CUDA is available. A separate public
velocity-feedback fixture checks simultaneous large heading change and tilt
tracking through actual rigid-body dynamics. The separate control
validation harness measures larger coupled scenarios with noise and transport
latency; its actual metrics and limits belong to its generated report.
