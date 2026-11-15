# Optional public trim correlation for the terminal backup

Proposal only; the current runtime does not supply this history certificate.

The untrimmed PD backup's parameter-dependent equilibrium can differ from its anchor by about a metre. A public, history-derived thrust feedforward can reduce that reserve while preserving the completed local Jacobian proof.

Let `F_hold` be a commanded SI thrust selected using only bounded state measurements, issued command history, known actuator/queue parameter ranges, and a validated dynamics enclosure. Define a correlated parameter posterior Gamma which is guaranteed to contain the true profile. No sampled actual mass, gravity, applied motor thrust, or simulator-initial mg command is exposed to the guard.

For a chosen acceleration tolerance delta_a, the condition

`abs(b*F_hold/m - g) <= delta_a`

holds for every profile in Gamma if and only if

`sup_Gamma m*(g-delta_a)/b <= F_hold <= inf_Gamma m*(g+delta_a)/b`.

For interval-box outer approximations of Gamma, monotone endpoint bounds give a conservative exact interval intersection. If empty, the certificate is unavailable: keep observing/calibrating under an independently safe initial protocol, or reject the contract. Do not filter out the task or declare the physical plant unavoidable. A whole uncalibrated full-mass-range box generally fails this intersection for a useful small delta_a, as it should.

Use net force per nominal mass `-Kp*(p-anchor)-Kv*v+(F_hold/m0)*up` in the observer-free backup. The actual equilibrium is

`p_eq,z-anchor_z = (F_hold-m*g/b)/(m0*Kp)`.

The correlation implies an offset bound `m_max*delta_a/(b_min*m0*Kp)`, approximately `.772*delta_a` for the current outward-rounded full profile domain and b>=.95. With delta_a=.05 m/s² this is about .039 m. At that equilibrium the requested vertical force per nominal mass remains `m*g/(b*m0)`, so the previously derived local linear map, Bernstein embedding, and common-P proof are unchanged. This does not prove the nonlinear invariant region or the calibration service.

A validated history service can derive Gamma from the exact real RK4 velocity balance over a window. For each substep j with RK4 weights beta_k, define conservative interval aggregates

`I_F = sum_j h_j * sum_k beta_k * F_j * body_z(q_jk).z`

and

`I_D = sum_j h_j * sum_k beta_k * norm(q_jk)^4 * v_jk.z`.

The full plant's isotropic-drag identity is used here; stage quaternions need not be unit. Then exactly for the real discrete velocity update,

`m * Delta_v_z = I_F - m*g*T - d*I_D`.

Bound Delta_v using the two bounded velocity packets. Bound the integrals using the validated full-state/command-history reachability service, not sampled stage truth. Remove a candidate parameter box only when this equality is impossible for all values in its sound outer intervals. Repeated windows or safe excitation can reduce uncertainty; passive data need not identify a unique trim, and the method must be allowed to decline certification.

For time-varying authority, update the history certificate or carry the bounded drift as disturbance. A fixed, once-calibrated F_hold does not remove battery depletion. Service provenance, packet times, numerical roundoff, initial motor interval, queue state, and every window's consistency tests belong in the certificate. Existing Gaussian observations cannot supply deterministic bounds simply by being labelled as this service.
