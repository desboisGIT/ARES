# Phase 4: learned residual authority investigation

The reduced-authority candidate improves validation reliability, but **Phase 4
remains incomplete**. The next held-out cohort, starting at 80000, has not been
used. Previous test cohorts and checkpoints remain sealed.

## Current requirements and plan

The learned primary policy still needs at least 99% waypoint-pair success on a
fresh frozen test across the declared randomization population. The shield needs
enforceable controller, estimator and latency bounds plus adversarial coupled
coverage. Inference under 5 ms must be measured on the actual onboard computer.
That hardware is unavailable. The PPO comparison now has newly trained policies
on the same current architecture, with paired validation results below; final
held-out comparison waits for candidate selection.

The work order is to remove repeatable learned lateral drift and speed excursions
on validation, freeze the selected BPTT/PPO/reference policies, and evaluate once
on another 1,000-task test cohort. Conditional shield domains and actual-device
timing remain separate gates. See `docs/phase4_completion_plan.md` for the complete
requirements map, and `phase_4_learning_protocol.json` for the declared protocol.

## Confirmed timeout mechanism

The previous 118/128 development result had nine timeouts and one contact. Every
failure was a U-turn; all 97 right-angle tasks succeeded. Full CUDA/sensor replays
reproduced the original outcomes. In stalled cases the learned residual opposed
about .20–.22 m/s² of reference acceleration near the .10 m/s reversal guidance
floor. Vehicles could remain .110–.197 m from a waypoint with a .10 m acceptance
radius. Obstacle loss was already zero there.

With the old weights unchanged, projecting only the opposing residual recovered
127/128 tasks; reducing global residual authority by four recovered 128/128 and
removed the contact. These are validation ablations, not retraining or new
held-out evidence. Drivers, traces, the immutable checkpoint and all 51 original
source files extracted from its archived wheel are retained under
`phase_4_learning_artifacts/diagnosis/` and linked by
`phase_4_learning_diagnosis.json`.

## New training and paired validation

The new explicit candidate limits learned body-axis corrections to .5 m/s²,
instead of 2 m/s². The nominal controller and explicit braking retain their
authority. This is a componentwise bound; the rotated residual norm can reach
sqrt(3) times that value. The old default and existing checkpoint configurations
are preserved. The CLI accepts `--residual-acceleration .5`, records the setting,
and seeds initialization consistently for both learners.

Each learner saw 50 updates × 128 steps × 64 environments, or 409,600 transitions.
BPTT has 50 optimizer updates; PPO has 6,400 minibatch optimizer updates. Both use
the same architecture, declared initial seed, sensor interface and physical
dynamics. Compute budgets differ. A second independent BPTT training seed tests
reproducibility. All losses, gradients and checkpoint tensors are finite, with
nonzero depth-encoder training evidence.

| Policy | Development 45000–45001, 128 tasks | Prospective 46000–46003, 256 tasks | Prospective contacts | Prospective timeouts | Prospective speed-exceeding runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| BPTT, seed 1741 | 127/128 | 254/256 (99.22%) | 2 | 0 | 1 |
| BPTT, seed 1742 | — | 254/256 (99.22%) | 2 | 0 | 1 |
| PPO, seed 1741 | 126/128 | 255/256 (99.61%) | 1 | 0 | 2 |
| Untrained controller reference | 128/128 | 256/256 | 0 | 0 | 1 |

Every model uses matching task/profile cohorts within each column. Both BPTT
seeds contact the same two prospective obstacles: scene 46001/task 41 and scene
46002/task 12. No failure is discarded. The primary's prospective minimum swept
clearance is −.002504 m and maximum speed 1.51790 m/s. PPO's corresponding values
are −.001065 m and 1.59669 m/s. The reference's maximum is 1.52033 m/s, so slight
speed overshoot also exists without learning. Its minimum clearance is .01455 m.

The declared candidate-selection rule also requires no validation contacts or
speed excursions. It is not met. These validation percentages do not establish
the 1,000-run held-out gate, and the reserved 80000 cohort remains untouched.

## Remaining contacts and next experiment

Paired full-sensor CUDA replays traced the three development contacts to learned
cross-track acceleration bias. Removing only the learned component perpendicular
to the active route leg cleared all three contacts, while retaining tangential
learning and the complete nominal controller. The route axes in that ablation
come from evaluator geometry: this is a privileged diagnostic, not a deployable
policy rule or a new policy observation.

The current loss penalizes changes in physical commands, but a sustained learned
sideways bias can incur little smoothness cost. The next isolated experiment is
an explicitly weighted penalty on learned acceleration corrections, calculated
from public policy outputs. Compare it with this .5-authority baseline, retaining
the same tasks, physics and evaluation criteria. Any new coefficient must be
selected on validation. A dense ordered-route progress term remains a separate
alternative if stalling returns; it is not the first change for these contacts.

One reference trajectory uses almost all its .13 m route clearance, so lateral
learning is not the only uncertainty budget. The speed excursions also occur in
the reference. Tracking-aware speed reserve and enforceable shield bounds need
separate validation rather than assuming a small residual is safe.

Full traces and causal controls are linked from `phase_4_learning_diagnosis.json`
at `phase_4_learning_artifacts/diagnosis/contact_diagnosis.json`.

## Verification and artifacts

All 331 local tests pass, including configurable residual bounds under rotated
attitudes, unchanged nominal braking, exact PPO squashed probabilities,
historical checkpoint behavior and seeded configuration forwarding to both
learners. The wheel matches the frozen 51-file package source manifest and passes
outside-checkout CPU training/reload/evaluation with CuPy blocked. Packaging
evidence is in `phase_4_learning_packaging.json`; Docker was not rebuilt for this
CLI extension, and the previous repair image is identified separately.

The machine-readable index is `phase_4_learning.json`. Full training curves,
immutable checkpoints, paired per-task outcomes and randomization audits remain
in the referenced training/evaluation artifacts. Verification logs are retained
under `phase_4_learning_artifacts/verification/`. No changes to success tolerances,
task geometry, collision evaluation or simulator physics were used to obtain the
new percentages.
