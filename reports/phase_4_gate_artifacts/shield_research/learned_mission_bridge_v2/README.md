# Renewable learned-command mission development

This version keeps the frozen BPTT policy, original full-profile Gaussian/depth
observations, global60Hz sensor/policy clock and100Hz controller observer from
v1. Only the certified100Hz backup clock renews. It is prospective development;
no acceptance or hardware claim is made.

The physical scene adds shelving that is represented identically on the .2m NN
sensor lattice and .25m bounded mapper lattice. Fresh forward first-hit rays at
20Hz must fit the sampled camera FOV and maximum range; exact robust witnesses
are the only source of additional free cells. The intentionally unobserved x7m
slab is not silently removed. New public rays may observe it. Existing reserved
free cells remain cached; newly added cells must have current capture times and
leases covering the complete current backup deadline.

`adapter_v2.py` proposes finite component alternatives from learned physical
command intent when a diagonal proposal meets unknown space. It never receives
the mission waypoints or a truth map. Every candidate is admitted only by the
unchanged exact current-family and whole-reserve guard. Actual binary32 word
changes are interventions.

`trim_target.py` proposes a constant trim word from PUBLIC issued words and
bounded vz history using nominal mass/capacity. This numerical estimator has
no safety authority. The immutable renewable producer still requires a complete
constant-word history, full discrete trim bound, fresh-family containment and
future known-free reserve. Rejected/fallback/word-mismatch receipts clear the
window. Target retries start a new window; they cannot inherit prior receipts.

`epochs.prepare_epoch_switch` constructs a new guard only after recomputing the
admission from public receipts. Its first command is already pending and is
issued and acknowledged exactly once before committing the new active guard.
The physical state, motor, energy, delayed queue and global NN clock never reset.
Reports retain the complete successful receipt bundles, source hashes, packet
clocks, public energy bounds and conservative absolute/local time conversion.

The hard SIL state and static-map service lease is55s. A public command-only
energy ledger independently limits admissible renewals. Driver experiments are
capped at45s and keep all unsuccessful targets, windows and missions. The tasks
have .634m and1.086m legs and unchanged .1/.35m reach thresholds.

Current failed attempts are retained: a simple recent-command mean lost vertical
admission, early refresh alone did not fix that, and a balance estimator achieved
one renewal but not the second before its deadline. The public-feedback target
retry also stopped at the next backup deadline. All7 attempts and6432 physical
steps remain indexed in `development_attempts.json`; none completed both waypoints.
Further long runs await a separately proved flexible-history streaming producer.

The synchronous renewable proof assembly costs more than a10ms physical period.
Current results use the discrete SIL clock and must not be described as a real
100Hz wall-time scheduling guarantee. Incremental/asynchronous proof preparation
needs fresh endpoint revalidation before live timing acceptance.
