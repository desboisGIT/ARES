# Portable component runner v2

This wrapper preserves the frozen `coupled_reanchoring_v1.run` protocol and calls
it once per case in a separate worker process. Each worker reserves at most 1,000
steps under the unchanged `ControlWorkerGC` helper. The helper performs preflight
collection, disables cyclic GC only in the child context, records RSS/GC state,
and restores the prior setting. Reference counting stays active.

Proposed versioned launcher targets:

```sh
python run_component.py smoke --cases 1 --steps 10 --output /tmp/fresh-component-smoke.json
python run_component.py validate --output /tmp/fresh-component-validation.json
```

Strict validation fixes eight sequential cases of 1,000 steps each. Smoke checks
complete functional outcomes; strict validation also requires every original
10 ms timing check. Both modes report functional, timing and combined conditional
SIL status separately. Nothing establishes universal execution time or physical
flight certification.

The parent report retains case IDs 0..N-1 in order and is compatible with the
existing launcher's `report_status`. Sibling `<output>.workers/` contains each
unchanged raw frozen-runner report, its log and wrapper metadata. Existing files
or worker directories are refused. No old evidence is overwritten.

Source snapshots include the frozen runner's full source/proof/input closure,
this wrapper's source folder and the GC helper. These snapshots bind executing source, proof and input files; they do not bind
historical output reports. Each child snapshot must equal
the parent's initial snapshot; the final snapshot must also match. Failed workers
remain in the aggregate with their exit and log paths.

Counts are explicitly reported for the sealed fixed-count loop, not inserted
per tick: attempts use the retained trace/error tick; completed words use the
original acknowledgement count. An exception during an unacknowledged physical
step is not claimed as completion. Raw errors remain unchanged, and incomplete
functional runs fail. The wrapper does not change guard thresholds, uncertainty,
plant state, queues, observations or requested commands.

Preparation validation only: syntax and four small nonflight tests passed.
No flight or timing run was executed during preparation.
