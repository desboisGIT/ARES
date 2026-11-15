"""Replay-verified human baseline aggregation, without trusting stored scores.

``actor='human'`` and real-time metadata are operator attestations. They are not
cryptographic proof of who generated commands. CPU replay verifies the recorded
simulated trajectory and metrics, not operator identity or a tamperproof clock.
"""
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .exploration import extract_rooms
from .exploration_scoring import build_map_report, score_exploration, EXPLORATION_SCORER_VERSION
from .fast_world import GroundTruthFastWorld, FastWorldConfig, FAST_WORLD_VERSION


METRIC_TOLERANCES = {"time_s": 1e-6, "coverage": 2e-4,
                     "floor_plan_iou": .0025, "minimum_clearance_m": .002}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _finite_positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _invalid_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def _compare_result(declared, trusted):
    if not isinstance(declared, dict):
        raise ValueError("result must be an object")
    for name in ("passed", "landed_safely"):
        if type(declared.get(name)) is not bool or declared[name] != trusted[name]:
            raise ValueError(f"stored result differs from replay: {name}")
    for name in ("collisions", "unsupported_map_cells", "seed"):
        if type(declared.get(name)) is not int or declared[name] != trusted[name]:
            raise ValueError(f"stored result differs from replay: {name}")
    for name in ("rooms_found", "rooms_required", "fail_reasons"):
        if not isinstance(declared.get(name), list) or sorted(declared[name]) != sorted(trusted[name]):
            raise ValueError(f"stored result differs from replay: {name}")
    for name in ("scorer_version", "world_version", "scene_version", "scene_id", "profile_version",
                 "profile_id", "scene_profile_sha256"):
        if declared.get(name) != trusted[name]:
            raise ValueError(f"stored result differs from replay: {name}")
    for name, tolerance in METRIC_TOLERANCES.items():
        value = declared.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not math.isclose(
                value, trusted[name], rel_tol=0., abs_tol=tolerance):
            raise ValueError(f"stored result differs from replay: {name}")
    if declared.get("rules") != trusted["rules"]:
        raise ValueError("stored result uses different scoring rules")


def _replay(log, *, expected_scene_sha256, expected_profile, expected_world_config):
    if not isinstance(log, dict) or log.get("schema_version") != "ares.exploration_log.v1":
        raise ValueError("not a supported exploration log")
    if log.get("actor") != "human" or log.get("automated") is True or log.get("ui_test") is True:
        raise ValueError("actor is not an attested human run; automated/UI tests are excluded")
    if log.get("started") is not True or log.get("completed") is not True or log.get("error") is not None:
        raise ValueError("human run was not started and completed without a session error")
    if log.get("realtime_eligible") is not True:
        raise ValueError("run is not marked eligible for a real-time human baseline")
    if log.get("world_version") != FAST_WORLD_VERSION or log.get("scorer_version") != EXPLORATION_SCORER_VERSION:
        raise ValueError("world/scorer version differs from the current benchmark")
    scene, profile = log["scene"], log["profile"]
    scene_hash = _hash(scene)
    if log.get("scene_sha256", scene_hash) != scene_hash:
        raise ValueError("declared scene hash does not match the embedded scene")
    if expected_scene_sha256 is not None and scene_hash not in expected_scene_sha256:
        raise ValueError("scene is outside the expected benchmark cohort")
    if expected_profile is not None and profile != expected_profile:
        raise ValueError("profile differs from the expected benchmark profile")
    config = FastWorldConfig(**log["world_config"])
    if expected_world_config is not None and asdict(config) != expected_world_config:
        raise ValueError("world configuration differs from the expected benchmark configuration")
    seed = log["seed"]
    if type(seed) is not int or seed != scene["seed"]:
        raise ValueError("log seed does not match the embedded scene seed")
    actions = np.asarray(log["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 5 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("actions must be a nonempty finite N-by-5 array")
    if len(actions) > math.ceil(config.max_time_s / config.dt):
        raise ValueError("action trace exceeds the configured episode length")
    wall = _finite_positive(log["wall_duration_s"], "wall_duration_s")
    # Reject speedup/slowdown before the potentially expensive replay as well as
    # afterwards. Neither the eligible flag nor a copied result supplies time.
    action_time = len(actions) * config.dt
    ratio = action_time / wall
    if not .9 <= ratio <= 1.1:
        raise ValueError("simulation/wall duration ratio is outside [0.9, 1.1]")
    claimed_ratio = log.get("sim_wall_ratio", ratio)
    if type(claimed_ratio) not in (int, float) or not math.isfinite(claimed_ratio) or not math.isclose(
            claimed_ratio, ratio, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("declared simulation/wall ratio does not match durations")
    world = GroundTruthFastWorld(scene, profile, backend="cpu", seed=seed, config=config)
    for index, action in enumerate(actions):
        if world.ground_truth_state()["done"]:
            raise ValueError("action trace continues after episode completion")
        world.step(action[None])
    terminal = world.ground_truth_state()
    if not terminal["done"]:
        raise ValueError("replayed action trace is incomplete")
    if not math.isclose(terminal["time_s"], action_time, rel_tol=0., abs_tol=1e-6):
        raise ValueError("replayed duration differs from recorded action count")
    observation = world.observation()
    rooms = extract_rooms(observation["known_grid"], observation["resolution"], observation["origin"])
    report = build_map_report(observation, rooms)
    trusted = score_exploration(world, world.snapshot_for_scoring(), report)
    _compare_result(log["result"], trusted)
    trace_hash = _hash({"world_digest": world.digest, "actions": actions.astype(float).tolist()})
    return trusted | {"actor": "human", "scene_sha256": scene_hash, "action_trace_sha256": trace_hash,
                      "wall_duration_s": wall, "sim_wall_ratio": ratio,
                      "verification": "CPU action replay; stored report is not used as evidence"}


def summarize_human_logs(paths, *, expected_scene_sha256=None, expected_profile=None,
                         expected_world_config=None):
    """Aggregate completed human-attested logs after independent CPU replay.

    Completed collisions/timeouts count as valid failures in the pass-rate
    denominator. Partial, automated, non-real-time, incompatible and duplicate
    traces are rejected with reasons. Distinct human attempts on one scene are
    allowed; copies of the same normalized action trace are counted once.

    Scene hashes are always computed from the embedded scene; an optional
    declared hash must agree. Cohort/profile/config filters allow the caller to
    require the exact classical benchmark conditions. Small numerical tolerances
    accommodate CPU/CUDA arithmetic, while pass/landing/failure flags must agree
    exactly. This does not authenticate people or wall-clock attestations.
    """
    if expected_scene_sha256 is not None:
        expected_scene_sha256 = ({expected_scene_sha256} if isinstance(expected_scene_sha256, str)
                                 else set(expected_scene_sha256))
    if isinstance(expected_world_config, FastWorldConfig):
        expected_world_config = asdict(expected_world_config)
    records, rejected, seen = [], [], set()
    for raw_path in paths:
        path = Path(raw_path)
        try:
            payload = path.read_bytes()
            if payload[:2] == b"\x1f\x8b":
                payload = gzip.decompress(payload)
            log = json.loads(payload, parse_constant=_invalid_constant)
            record = _replay(log, expected_scene_sha256=expected_scene_sha256,
                             expected_profile=expected_profile, expected_world_config=expected_world_config)
            if record["action_trace_sha256"] in seen:
                raise ValueError("duplicate action trace for the same scene/profile/configuration")
            seen.add(record["action_trace_sha256"])
            records.append(record | {"path": str(path)})
        except (OSError, ValueError, TypeError, KeyError, OverflowError, EOFError) as error:
            rejected.append({"path": str(path), "reason": str(error)})
    # Deferred import permits phase3 to consume this helper without a module
    # initialization cycle, while sharing the benchmark's summary arithmetic.
    from .phase3 import summarize_results
    return {"schema_version": "ares.human_benchmark.v1", "summary": summarize_results(records),
            "records": records, "rejected": rejected,
            "world_version": FAST_WORLD_VERSION, "scorer_version": EXPLORATION_SCORER_VERSION,
            "metric_comparison_tolerances": METRIC_TOLERANCES.copy(),
            "trust_note": "Human actor identity and wall durations are unsigned operator attestations, "
                          "not cryptographic proof. Metrics are recomputed by CPU action replay."}
