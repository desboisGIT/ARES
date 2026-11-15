# ARES

Phase 0 foundations for the autonomous indoor mapping project described in the
[supplied build plan](docs/build_plan_v3.md). This repository currently validates scenes and drone
profiles and evaluates deterministic kinematic test episodes. It does not fly
a drone or implement exploration, odometry, learning, or realistic depth sensing.

## Run

With Python 3.11+ and pip available, build and check everything:

```sh
make build
```

This runs contract and adversarial tests, the Phase 0 gate, the paired benchmark,
and creates `dist/ares-0.1.0-py3-none-any.whl`. Build isolation installs setuptools
when needed; the installed application has no third-party Python dependencies.
Internet access may be needed for initial build tooling. No GPU is needed.

```sh
python3 -m ares validate
python3 -m ares phase0 --seed 42 --output reports/phase_0.json
python3 -m ares benchmark --check
python3 -m unittest discover -s tests -v
```

The gate writes its input scene/profile, scoring configuration, results, seed,
and artifact hash to `reports/phase_0.json`. Its sibling
`reports/phase_0_artifacts/` contains the mission report, a runtime log, and an
explicitly named evaluator log. Identical input and seed produce identical
mission, log, result, and gate JSON; wall-clock benchmark timings are separate.

For a container build, use `make container`, then
`docker run --rm ares:phase0`. Container execution needs a running Docker daemon.
The container runs as a nonroot user. CI checks Python 3.11 and 3.12 and uploads
reports and the wheel; CI results require the workflow to run on a remote repo.

## What passes the gate

The packaged scene has three enclosed rooms and two open doors, with a triangle
collision mesh generated from its 20 cm voxel grid. Both representations are
checked for exact agreement in the fixture tests. Scene/profile readers reject
malformed data with a field-specific reason. All random streams take an explicit
seed and name, independent of Python's global random state and salted hashes.

The scorer requires at least 95% supported voxel coverage, mean free/occupied
floor-plan IoU of at least 0.80, all reachable rooms, zero collisions, an episode
within the timeout, and a safe virtual landing. A successful evaluator witness
proves that the scorer can accept an episode. The four attack policies instant
report, endless hover, leaving the map, and an omniscient forged map must fail.
Dense training reward lives separately and cannot certify success.

The witness traverses evaluator-known free space. Its time is a harness fixture,
not an exploration or flight performance result. The evaluator checks swept
spheres continuously between samples, preventing voxel-wall tunnelling.
Only evaluator-executed observations support a policy's report; self-reported
coverage, room IDs, time, and landing claims cannot establish success. The
in-process library assumes the evaluator owns its objects; it does not sandbox
malicious Python code. Runtime log validation rejects the optional ground-truth
block unless explicitly authorized by evaluator tooling.

See [the contract definitions](docs/contracts.md),
[scoring assumptions](docs/scoring.md), [Phase 0 evidence](reports/phase_0.md),
and [research references](docs/references.md). Later phases need realistic camera
and dynamics models and hardware measurements before their gates can be met.

## Benchmarks and licenses

CI measures a versioned axial-scan workload against its frozen, equivalent
reference on the same runner. A normalized slowdown over 10% fails the check.
Baseline/reference changes require an explanation in the PR; this initial
baseline does not measure GPU throughput or full-stack latency. See
[benchmark policy](benchmarks/README.md).

`LICENSES.yaml` records dependencies, research references, and unresolved license
checks. No project source license has been selected. Unverified components are
marked `commercial_ok: false`; this flag also records the project's permissive
deployment policy, rather than asserting GPL prohibits commercial activity.
No external research code is incorporated into this foundation.
