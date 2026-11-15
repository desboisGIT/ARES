# Phase 4 braking guidance

`ares.flight_guidance.guidance_velocity(p, wp1, wp2, speed_limit, config=None)` is a pure Torch function. It returns world velocity from public estimated position, the two mission waypoints, and the declared speed limit. It reads no scene, map, clearance, sampled physical parameters, controller state, or future trajectory. It does not change an observer or maintain history. The environment may use the same function on physical state to construct a differentiable training label.

The implementation keeps the incoming waypoint direction until mission handoff. It therefore does not blend toward the next leg and cut across a right-angle corner. The next leg controls the speed to reach before handoff. With incoming/outgoing unit directions and their dot product `c`, the handoff speed is the speed limit clipped against:

`0.10 + (0.20 - 0.10) * (1 + c) / (1 - c + 1e-6)` m/s.

This gives approximately 0.20 m/s for a right angle, 0.10 m/s for a reversal, and full cruise for sufficiently straight continuation. The small reversal speed permits crossing the waypoint acceptance radius; commanding zero at that radius would strand an ideal velocity tracker before mission handoff. Equal waypoints mean the final leg and use zero terminal speed. Exact position/target coincidence produces zero velocity with finite gradients, including vertical legs and opposite directions.

For turns, the braking distance available before the 0.10 m corner acceptance radius is `d = max(distance - 0.10 - 0.20, 0)`. Developer assumptions are minimum deceleration `a = 0.8 m/s²` and reaction interval `L = 0.4 s`. When the incoming speed `v` exceeds handoff speed `u`, the cap solves:

`v*L + (v² - u²)/(2*a) <= d`

or `v <= -a*L + sqrt((a*L)² + u² + 2*a*d)`.

The resulting cap is clipped to the mission speed limit and floored at `u`. The floor applies after the requested slowdown is complete; the inequality is a deceleration budget and is asserted only where `v > u`. Straight continuation already permits full speed, so it retains cruise without a discontinuity caused by charging a reaction delay for a slowdown that is not needed. Final legs use distance to the final target directly, without subtracting an acceptance radius, and their commanded speed tends to zero at the target.

At a right-angle handoff, 0.20 m/s of residual incoming velocity requires at most `0.20*0.4 + 0.20²/(2*0.8) = 0.105 m` of forward travel under these assumptions. A 0.10 m/s reversal requires 0.04625 m. Both fit a 0.13 m forward margin; the 0.20 m lookahead preparation margin is additional preparation before handoff. These numbers are a design budget. Position error, inadequate actuator authority, tracking lag beyond the assumed interval, and unavailable physical clearance can invalidate that budget. Guidance cannot certify clearance because it receives no geometry. The final-goal acceptance tolerance remains a separate mission setting. The measured known-space shield remains responsible for its separate runtime decision.

The defaults and tests are analytic developer choices. They are not fitted to the frozen held-out cohort. Runtime integration must version this guidance separately so old checkpoints retain their earlier reference and controller semantics.

The v2 environment uses this shared velocity reference in its per-environment target-velocity loss. Its first-waypoint handoff uses `GuidanceConfig.waypoint_radius` (0.10 m). Final mission completion retains its separate 0.35 m position tolerance and speed below 0.8 m/s. The tighter first handoff allows the vehicle to approach the corner center before turning; it does not reduce the declared final-goal acceptance tolerance. Historical checkpoint configurations without a guidance version load as `legacy_v1`; new policy integration selects `braking_v2` explicitly.
