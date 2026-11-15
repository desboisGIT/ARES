# Zero-rate history stage and numeric-domain lemma

`zero_rate_history_v1.py` checks every one of the eight recorded public capture histories, without accepting sampled mass, motor, battery energy, gravity or inertia values. It establishes the additional stage and forward-arithmetic DOMAIN hypotheses of `trim_history_v2` conditional on hard public measurement, command-history, full parameter-set and CPU primitive contracts. It does not establish that the experimental sensor's declared hard bounds are truthful, or that the preceding PID acquisition was safe.

The public p/v/w packet errors are componentwise ep=.005m, ev=.01m/s and ew=.001rad/s, including measurement arithmetic; q is an exact current raw Float32 quaternion. Histories must be complete on the exact .01s macro grid. Every issued/stored command over settle+history is `(Fhold,0,0,0)` after the bounded transport interval. These are explicit supplier/history assumptions, not quantities inferred from innovations or RMS noise. All eight cases and all packets are retained.

For each physical inertia in the Float32-expanded parameter set and the represented nominal inertia, the unsaturated rate equation is exactly

`w_i' = -lambda_i w_i + c_i w_j w_k`.

Fraction endpoint checks give lambda in [27,34] and |c_i|<.278<1. Under |w_i|<=W=.01, the compensation gyro norm is below 1.81e-6, well below its clip threshold100. Thus rmax=W² is valid on every RK4 stage. With a=lambda*h, h=.002, a lies in [.054,.068]. Substituting the scalar linear term into the actual four RK4 stages yields coefficients

- S2 base `(1-a/2)w`, forcing `h*r1/2`;
- S3 base `(1-a/2+a²/4)w`, forcing `-h*a*r1/4+h*r2/2`;
- S4 base `(1-a+a²/2-a³/4)w`, forcing `h*a²*r1/4-h*a*r2/2+h*r3`.

Their exact triangle upper bounds are respectively .0097301, .0097373934 and .0094743933712, each below W. The RK4 endpoint coefficient is `R(a)=1-a+a²/2-a³/6+a⁴/24`; it is positive and decreasing on this interval. Each forcing coefficient is nonnegative and their sum is `1-a/2+a²/6-a³/24<=1`. Five substeps therefore give `R(amin)^5 W + h W² sum_{k=0}^4 R(amin)^k + 1e-6 < .007635696 < W`. The last term is the conditional per-H plant angular-velocity arithmetic allowance. This proves an inductive rate bound for recorded macro states in the entry domain, using the discrete map rather than an ODE solution.

The elementary norm bound `||w||2<=|wx|+|wy|+|wz|<=3W=.03` gives Nq<=1/(1-h*.03) and stage quaternion displacement D<=h*.03*Nq/2. For unit q at each REAL substep start, the normalized endpoint displacement is at most `2D/(1-D)`. Thus each stage xyz component is bounded by its measured initial projective component plus five such displacements plus D. For every packet the raw norm shell is checked with exact squared Fractions, and its projective coordinate bound is `abs(raw_qi)/(1-8u)`. Combining the REAL stage norm and qx/qy bounds gives

`|R_homogeneous(qstage).zz-1| <= max(Nq²-1,1-(1-D)²)+2*qxmax²+2*qymax²`.

The maximum over every retained packet/case is below .000120661, hence below the requested .0002. No exact-unit assumption is made for Float32 raw q or unnormalized RK4 stages.

The published full-plant arithmetic budget requires a narrow motor entry interval. To avoid assuming it during acquisition, a separate scalar original-recurrence analysis starts from motor[0,6.5] and uses the larger domain[0,6.501]. It includes stored-word fractional queue mixing, battery multiplication, rounded motor-lag division and verified-decay exp. The calculated five-substep forward arithmetic error is below1.01815e-5N; the chosen rational ceiling is2e-5N/H. Its stable geometric accumulation fits below6.501, and a positive target>=3.8N leaves a strictly positive lower motor bound even after this allowance. This establishes the settling-domain invariant independently of the later full-body error budget.

After .7s of constant command, removing maximum transport and one macro period leaves at least66 complete periods. A positive-series rational upper bound on decay gives residual `(6.5+Fhold)*exp_upper(-.66/tau_max) + 2e-5/(1-exp_upper(-H/tau_max))`. The resulting motor interval `[.95*Fhold-residual,Fhold+residual]` fits the published forward-budget motor domain for all eight cases. Each packet's p/v outer box, projected q box, W rate box, current command and both history words also fits the published input domain. These facts license use of that full-plant arithmetic allowance during the recorded history, subject to its CPU primitive contracts.

Four focused regressions cover all eight positives, strict rate/motor margins, a noisy velocity box crossing despite its center, invalid q/rate/thrust and a missing timestamp. The JSON preserves exact rational decision values and display-only decimals. All flight-certificate and bootstrap-safety flags remain false.
