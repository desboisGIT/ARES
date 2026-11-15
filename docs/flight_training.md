# Phase 4 learned flight policies

`ares.flight_policy.FlightPolicy` combines observed-state adaptive control (`adaptive_v2`) with a learned depth/state acceleration residual. BPTT and PPO use this **same architecture and initialization**. The actor starts at zero residual, so an untrained checkpoint acts as the declared controller reference. A short training run establishes that optimization works; it does not establish flight reliability or a held-out success rate.

The network has a two-layer depth CNN, a 54-input state encoder (43 base features plus 11 observed controller-memory features), a shared 64-unit MLP, a three-component actor and a scalar critic. Its residual is `2*tanh(latent)` m/s² per body axis, rotated into world coordinates. The shared `braking_v2` guidance tracks the incoming waypoint direction, caps speed using a declared braking/reaction budget, slows for corners and reversals, and requests zero endpoint speed. Policy and controller validation use the same pure guidance function. These braking assumptions are not certified aircraft bounds. Velocity feedback uses gain 2.5/s. Attitude tracking uses gain 6/s, nominal mass 0.5 kg and gravity 9.80665 m/s². Physical actions are collective thrust in `[0,24.5]` N and body angular rates in `[-6,6]` rad/s. These public constants are not the randomized simulator parameters.

## Observations and shield integration

The network consumes `depth: [B,2,24,32]` (normalized depth and validity) and `features: [B,43]`. Features are body velocity (3), body gravity (3), measured angular velocity (3), the two body waypoint offsets (6), body gaze offset (3), xyzw quaternion (4), speed limit (1), depth ring ranges (16), and normalized previous physical action (4). The adaptive encoder appends body-frame observer bias (3), observer velocity (3), hold-anchor offset (3), holding flag (1), and stopping flag (1). These snapshots come from public observations and command history, are advanced once per physical step, and are stored explicitly for PPO re-evaluation. The base feature tensor remains `[B,43]`; the adaptive state encoder receives 54 values. Training and inference use sensor observations. Hidden occupancy, exact collision distances, randomized mass, wind and other simulator internals are not network inputs.

The geometric controller also uses named observation tensors: world `p`, `v`, world waypoint positions `wp1`, `wp2`, world gaze target position `gaze`, body-to-world xyzw `q`, and `speedlimit` (`[B]` or `[B,1]`). These must represent the same measured state and mission data as the feature vector.

```python
from ares.flight_policy import FlightPolicy

policy = FlightPolicy().to(env.device)
observation = env.reset()
acceleration = policy.desired_acceleration(observation)
# A separate safety shield may replace acceleration using its permitted inputs.
action = policy.action_from_acceleration(observation, acceleration)
observation, reward, done, info = env.step(action)
```

`policy(observation)` performs both operations directly. `policy.sample(observation)` returns physical action, Gaussian latent, squashed-residual log probability, and value for PPO. A shielded deployment should be identified separately in evaluation; the policy does not silently invoke a shield during training.

## Training

```python
from ares.flight_training import TrainingConfig, train_policy, save_checkpoint

policy, result = train_policy(
    env, algorithm='bptt',
    config=TrainingConfig(steps=50, rollout_steps=128, seed=42),
)
save_checkpoint(
    'outputs/fresh-run/policy.pt', policy,
    algorithm=result['algorithm'], seed=result['seed'],
    training_steps=result['training_steps'],
    training_curve=result['training_curve'],
    metadata={'training_config': result['config'], 'heldout_evaluated': False},
)
```

`steps` means optimization windows for BPTT and rollout/update rounds for PPO. `training_steps` reports the actual number of environment transitions, including batch size. PPO additionally reports the actual number of minibatch optimizer updates. `train_bptt(env, policy, config)` and `train_ppo(env, policy, config)` train an existing policy in place and return the result dictionary. `train_policy` seeds initialization and returns `(policy, result)`. The environment should be created with the same declared seed; training seeds do not alter an already constructed environment's independent randomization stream.

BPTT unrolls the real differentiable rigid body. After each physical action and transition it obtains `env.differentiable_loss(action, previous_action)`, averages `total` over time and batch, backpropagates, clips parameter gradients and applies Adam. State and actuator memory are detached **after** each optimizer window. Resetting an episode creates another graph boundary. No target-action imitation or reward-gradient shortcut replaces the dynamics. Every component supplied by the environment is recorded alongside reward, gradient norm and depth-CNN gradient norm. Zero initialization means the first BPTT actor update can have zero CNN gradient; later updates must demonstrate sensor-encoder gradients.

`TrainingConfig.residual_penalty` is an optional finite, nonnegative coefficient,
defaulting to zero. It penalizes a sustained learned correction even when physical
commands change smoothly. For a policy latent `z`, the cost rate for one transition
is the squared Euclidean norm

```
r_world = rotate(q, residual_acceleration * tanh(z))
residual_l2 = sum_xyz(r_world ** 2)
```

This uses public orientation and policy outputs only. It does not use a true route
axis, sampled dynamics, occupancy or evaluator clearance. For BPTT, `z` is the
actor mean, and the update objective is the existing physical dynamics loss plus
`residual_penalty * residual_l2`, averaged over live lanes at each step and then
over the window. Squared vector components are summed, not divided by three;
the unweighted rate has units m²/s⁴. Action and residual share one actor/depth
encoding, retaining the same deterministic action conversion. The curves record
`residual_l2` and `residual_l2_weighted`, the latter being the coefficient times
the unweighted rate. With coefficient zero, the original objective and parameter
updates remain unchanged; reporting this metric adds no stochastic draws.

For PPO, `z` is the actual sampled latent already retained for the rollout.
Before GAE, each environment reward becomes
`env_reward - residual_penalty * residual_l2 * env.dt`. The coefficient therefore
weights a cost rate integrated over the actual public control period. A positive
coefficient requires finite positive `env.dt`; zero preserves old fixtures that
do not expose a timestep. The exact latent log probability, tanh Jacobian and
clipped ratio are unchanged. The sampled cost can discourage both mean bias and
stochastic residual variance. It is a reward cost, not an additional differentiable
PPO minibatch penalty or a gradient through the plant. PPO curves retain the
unweighted and weighted rates, `residual_cost` (the weighted per-transition cost),
`env_reward` (before this cost) and `reward` (after it). All means exclude invalid
transitions. BPTT and PPO retain their existing different objective scaling;
their raw reported total losses are not directly comparable.

For example, `TrainingConfig(residual_penalty=1.)` explicitly enables the cost for
both learners. The complete configuration is returned in the training result and
should be retained in checkpoint metadata. A coefficient must be selected on
development data and frozen with the candidate; this regularizer supplies no
collision, speed or reliability guarantee. Prior experiments keep coefficient
zero and their archived checkpoints remain unchanged.

The obstacle training loss uses the physical sphere-to-occupied-box clearance and scene boundaries. Nearest-box selection runs outside autograd in bounded chunks; the selected box's analytic distance retains the position gradient. Clearance is exact outside occupancy. Inside occupancy, signed box distance supplies a penetration surrogate rather than claiming exact penetration depth for the voxel union. This optimization signal remains separate from the unchanged continuous swept-sphere collision evaluator.

Each sampled two-leg axis-aligned route receives a fixed, evaluator-only soft clearance margin: `min(.10 m, .5 * minimum physical route clearance)`. The route clearance includes the sampled body radius and world boundaries. Nonpositive route clearance fails explicitly; tasks are never filtered or silently repaired. Selected resets replace only their own budget. This keeps the soft objective feasible for the assigned task without weakening collision evaluation or exposing geometry/budgets as policy inputs.

The reason for replacing the earlier conservative EDT loss is a concrete objective conflict. At .2 m voxel resolution, a nearest axial occupied-cell center .4 m away gives true sphere clearance .13 m for radius .17 m, but the old `EDT - sqrt(3)*resolution - radius` surrogate gives -.1164 m. Its .25 m soft margin then contributes about 2.014 weighted obstacle loss, versus only .04 squared velocity error for stopping instead of following a .2 m/s corner command. Exact distance alone would still impose an infeasible .25 m margin. In a diagnostic sample of 640 initial training tasks (64 per scene 100–109; stage/lane seeds `1741 + stage*10007 + lane`), 442 first waypoints had negative surrogate clearance although all had positive physical clearance. Five complete routes had physical clearance below .10 m, with minimum .01586 m. These measurements describe that development sample, not all future resets or held-out performance.

PPO collects detached on-policy rollouts, computes GAE and fits a clipped policy objective plus value regression and squashed-residual entropy. Old and new log probabilities evaluate the stored Gaussian latent with the exact `tanh` Jacobian. The density belongs to the residual policy; the physical controller can clamp its output, so no invertible physical-action density is claimed. True terminal transitions do not bootstrap. Time limits bootstrap the final observation's value and stop the advantage trace. `FlightEnv.reset_idx` resets only completed environments, including their observer and hold memory; surviving episodes retain their state and randomization. Advantage traces stop at each selected reset. Legacy fixtures without selective reset retain the whole-batch fallback. PPO never backpropagates through environment dynamics, hidden occupancy, collision checks, or reward construction.

The Phase 4 runner uses staged training scenes with one policy and optimizer across stages. Stage boundaries intentionally reset the training scene; ordinary episode completion resets only its own environment. Adam state continues across declared stages. The repair target uses 128-step BPTT windows to expose longer braking and hold behavior. BPTT is the primary reliability candidate; PPO is a comparison.

Both trainers accept an optional `progress=callback` keyword (also accepted by `train_policy`); the callback receives a copy of each completed update row for live logging. Both trainers reject nonfinite losses/gradients. They record actual training curves and gradient diagnostics. Their configuration includes rollout length, Adam learning rate, clipping norm, seed, and PPO discount, GAE, clipping, value/entropy coefficients, epochs, minibatch size and residual penalty. Optimizer state is not saved; these are inference checkpoints, not exact training-resume snapshots.

## Checkpoints and evaluation

`save_checkpoint` stores a schema version, algorithm, policy configuration, architecture label, tensor state dictionary, seed, transition count, training curve and caller metadata. Publication uses a temporary file plus an exclusive atomic link: it refuses an existing destination, including concurrent writers. Use a fresh run directory. Metadata must be finite JSON data. `load_checkpoint(path, device='cpu')` explicitly uses PyTorch `weights_only=True`, validates the schema/algorithm, strictly loads tensors and returns `(policy, payload)` in evaluation mode.

New policies default to `adaptive_v2` and `braking_v2`. Checkpoints missing these version fields load as `nominal_v1` and `legacy_v1`, preserving the historical 43-input encoder and waypoint blending behavior rather than silently changing old weights. Existing checkpoints remain immutable.

Run `make control-check` and `make control-validation` before `make phase4`. The repaired run freezes checkpoints before evaluating 1,000 test episodes starting at seed 70000, separate from training scene seeds. Its 99% waypoint reliability gate remains unresolved until measured; shield certification and onboard inference gates also remain unresolved. Development controller checks cannot satisfy those gates. Historical results in `reports/phase_4.md` describe the earlier controller.

Record scene/profile identifiers and hashes, randomization mode, environment seed, training and held-out seed sets, dynamics version, training configuration and shield configuration with an experiment. Evaluate BPTT and PPO with the same controller architecture, sensor inputs, initialization convention, physical dynamics and evaluation seeds. Include the zero-residual geometric reference to distinguish learning benefit from the PD initialization. Report completion, collision, intervention, tracking error and runtime only from the corresponding actual runs, with training and held-out results separated. No built-in training result implies a 99% success claim.

Validation includes physical-action bounds and signs, exact squashed log probabilities, terminal versus time-limit GAE, exclusive weights-only checkpoint round trips, and real DiffAero BPTT/PPO updates with nonzero actor and CNN training evidence:

```sh
.venv/bin/python -m unittest tests.test_flight_policy tests.test_flight_regularization -v
```

The regularization tests compare zero-cost BPTT against its original forward/loss
ordering through actual DiffAero transitions, verify the added sustained-bias
gradient alongside the physical-state gradient, check one encoding per BPTT
transition, and verify exact zero-cost PPO RNG/weights, unchanged sampled latent
densities, timestep-scaled rollout costs and invalid-coefficient rejection.
