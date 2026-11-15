# Bounded isolated-worker cyclic-GC diagnostic v1

`../control_worker_gc_v1.py` is a new, independent process-local discipline. It
changes no frozen runner, ARES source, mathematical law, model, noise or gate.
Importing it changes no settings. Use it only in a dedicated child worker:

```python
from control_worker_gc_v1 import ControlWorkerGC

with ControlWorkerGC(max_steps=4500) as worker:
    for tick in range(4500):
        worker.control_step()          # attempted tick, before guard execution
        # existing producer/guard/word/physical pipeline remains unchanged
        worker.physical_step_completed()  # only after an actual word completes
report = worker.report                 # available after finally/restoration
```

The caller must provide the actual isolated-process/event-loop boundary. The
context is not a security sandbox or a preemptive watchdog for a stalled call.
It rejects nonpositive/noninteger/boolean budgets, maxima over4500, negative
counts, over-budget attempts, excess physical completions, nesting and reuse.
The previous GC-enabled state is restored in `finally`, including exceptions and
an initially disabled GC. Thresholds are not changed. Reference counting remains
active throughout. Re-enabling cyclic GC inside a per-tick control context is a
reported discipline error; explicit collections are visible in the counters.

Before arming, the context performs and measures an explicit full `gc.collect()`.
It then disables automatic cyclic collection only within its bounded scope.
Reports retain attempted/completed/reserved counts, preflight/elapsed wall time,
GC generation stats and counts, restoration status and Linux current/lifetime
peak RSS. Neither GC pauses nor OS preemption are subtracted from timed calls.

## One predeclared Docker experiment

The diagnostic ran once on the existing immutable image
`sha256:404a8ed9de35680a82f627899d018e1cc4a8acb90845c88080bc795eaae0a281`,
UID10001, pinned CPU7, network disabled, with repository mounted read-only.
It called the **unchanged frozen** component case0 for1,000 steps inside the GC
context. Its `reserve_declared_steps` mode declares the sealed loop bound before
calling it and reports successful attempted/completed counts afterward; it does
not pretend to instrument each tick. Future mission users should use the
per-tick API above. Failure to obtain complete counts leaves them unknown.

This was a policy change addressing the observed generation2 pause, not an
unchanged retry until a favorable timing sample. Earlier Docker/build failures
remain untouched. The first10 trajectory/command/oracle fields exactly match
`docker_runtime02.json`; only timing differs.

Observed results:

- 1,000 attempted/completed words and positive conditional decisions; zero actual-C,
queue, collision or unknown-entry failures.
- Explicit preflight collection24.099ms,17 objects; zero collections during the
bounded context, prior enabled=True restored=True.
- Guard decide+acknowledge mean2.596ms, maximum2.849ms, first call2.461ms;
zero calls over10ms. Wall timers include scheduling/preemption.
- Current RSS229372→267308KiB; process peak272304KiB. Context elapsed12.409s
includes sealed-runner setup, oracle, physics and serialization; it is not an
end-to-end100Hz throughput claim.
- Parent launcher GC-enabled state and thresholds unchanged; all executing
source/input/proof hashes stable.

Eight helper tests verify success/error restoration, initially-disabled success
and failure restoration, budget/count rejection, completion accounting,
re-enabling/explicit-collection visibility, and nesting/reuse rejection.

`docker_gc_diagnostic.json` retains the context and full component report;
`docker_component_raw.json` is the unchanged runner's output. `docker_launch.json`
records image/UID/affinity/parent settings and prefix comparison. Logs and
`tests.log` retain execution evidence. `summary.json` provides concise numbers.

These are finite component observations, not universal WCET, real hardware,
Gaussian-runtime certification or complete live sensor-to-actuator timing.
No phase-pass flag is added and `flight_certificate` remains false. The4500-step
mission memory/timing envelope has not been measured by this1000-step diagnostic.
