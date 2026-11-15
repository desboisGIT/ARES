# Conditional alternative: dense remaining-route objective

Status: proposal only; no runtime implementation. First evaluate new policies trained from scratch with .5 m/s² residual authority. The unchanged-weight scaling experiment is separate evidence and does not predict that training outcome.

## Why consider this only if needed

On the 45000/45001 development cohort, nine timeouts were recovered by constraining the residual component opposing guided velocity. This establishes an authority failure mechanism. It does not establish which training term caused that behavior. One plausible incentive is that staying just outside first-waypoint acceptance incurs only about .007 velocity loss and negligible gaze loss, whereas the successful reference's turn/reversal transient can incur about .15 weighted gaze loss. There is no dense mission-distance term in the current BPTT objective. Its guided velocity floor is effectively flat with respect to distance near handoff, so low tracking loss alone does not express a deadline or adequate mission progress.

## Minimal addition

Add an evaluator-only, active-transition remaining-route loss to the existing tracking/collision-clearance/smoothness/speed objective:

- Let `d = norm(current_target - physical_position)`.
- Let `L2 = norm(second_waypoint - first_waypoint)` and `L = first_leg_length + L2`, both fixed from the assigned route at reset.
- Define `remaining = d + (L2 if first waypoint has not been accepted else 0)`.
- Add `lambda_progress * remaining / L` to each active BPTT step's loss. Record the unweighted component separately. Keep the route lengths and this term out of policy observations.

This uses an ordered route, so an early shortcut toward the final waypoint cannot earn progress before visiting the first. It supplies a position gradient toward the current target even when the desired-velocity floor is nearly constant. It also accounts for the unfinished second leg: waiting before handoff leaves that distance in the objective.

For a position within the first-waypoint acceptance ball, changing the remaining-route definition to the second leg cannot increase this component: `norm(p-wp2) <= norm(p-wp1) + norm(wp1-wp2)` by the triangle inequality. The discrete handoff itself has no gradient; the continuous distance terms on both sides provide the useful signal.

Use a small, explicitly declared development weight and compare against the .5-authority baseline with the same training seed and exposure. Select its scale using logged component/gradient magnitudes on training and new development data, not sealed test failures. Do not claim a weight from the present diagnostic traces. Preserve the existing physical collision checker and all task acceptance/deadline requirements. This is a change to the training objective, not the inference controller or evaluation success definition.

Do not combine a new progress term, gaze removal, authority projection, and unrelated tuning in the first experiment. If the progress experiment fails, separately test moving the gaze term to a diagnostic metric: yaw tracking already belongs to the nominal controller, while the learned acceleration residual can only affect heading indirectly. Removing its training weight would not remove the heading requirement or justify omitting heading metrics.

## Meaningful regression checks

1. In a collision-free reversed route near the first waypoint, compare stationary states .18 m and .12 m before the .10 m handoff radius. With identical zero obstacle cost, the progress component must prefer the closer state and its physical position gradient must point toward the first waypoint, even though the final waypoint is behind the vehicle.
2. Check that the progress component does not increase when the same physical state crosses the discrete first-waypoint acceptance boundary; exercise straight, right-angle, reversed, and vertical routes. Check finite values and gradients at exact targets.
3. In a real differentiable dynamics rollout initialized just outside handoff, set an acceleration residual opposing the guidance floor. Verify the added component contributes a corrective gradient that increases forward travel rather than rewarding the stationary equilibrium. This checks the actual policy-to-dynamics-to-loss path, including delayed commands.
4. Selected resets must replace only that lane's route lengths; survivors' normalization and graphs must remain unchanged. Inactive transitions contribute no progress loss.
5. Run paired training pilots and evaluate collision rate, handoff time, remaining distance at timeout, speed/clearance, and completion on an independently declared validation cohort. A lower training loss alone is not success.

The PPO reward already includes detached physical distance progress plus event rewards; this proposal concerns the BPTT differentiable objective. It does not silently redefine PPO returns. Record an explicit objective version and coefficient in future checkpoint metadata and source manifests.
