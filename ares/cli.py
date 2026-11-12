"""Small, offline Phase 0 build and evidence commands."""

import argparse
import copy
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import sys

from .attacks import run_attacks
from .benchmark import run_benchmark
from .contracts import ContractError, load_json, validate_scene, validate_profile
from .evaluator import GroundTruthWorld
from .flight_log import make_flight_log
from .reporting import make_report
from .scoring import score_episode


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def default_inputs():
    data = files("ares") / "data"
    return load_json(data / "scenes/three_room.json"), load_json(data / "configs/drone_nominal.json")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_invalid_scenes(scene):
    """Require each corruption to fail for its intended diagnostic."""
    checks = []
    mutations = (("out_of_bounds", "grid.occupied: voxel index out of bounds",
                  lambda s: s["grid"]["occupied"].append([9999, 0, 0])),
                 ("occupied_spawn", "spawn: must lie in free space",
                  lambda s: s.update(spawn=[0.1, 0.1, 0.1])),
                 ("missing_room", "doors[1].room_ids: requires two distinct existing rooms",
                  lambda s: s["rooms"].pop()))
    for name, expected, mutate in mutations:
        broken = copy.deepcopy(scene)
        mutate(broken)
        try:
            validate_scene(broken)
        except ContractError as exc:
            reason = str(exc)
            checks.append({"case": name, "rejected": True, "reason": reason,
                           "expected_reason": expected, "reason_matches": reason == expected})
        else:
            checks.append({"case": name, "rejected": False, "reason": "accepted invalid scene",
                           "expected_reason": expected, "reason_matches": False})
    return checks


def phase0(*, seed=42):
    scene, profile = default_inputs()
    world = GroundTruthWorld(scene, profile)
    episode = world.witness(seed)
    report = make_report(world, dict(episode.observations), episode.samples[-1].timestamp_s, scene["rooms"])
    witness = score_episode(world, episode, report)
    attacks = run_attacks(world, seed)
    runtime_log = make_flight_log(episode, scene_id=scene["scene_id"], profile_id=profile["profile_id"])
    evaluator_log = make_flight_log(episode, scene_id=scene["scene_id"], profile_id=profile["profile_id"],
                                    include_ground_truth=True)
    second_episode = world.witness(seed)
    second_report = make_report(world, dict(second_episode.observations),
                                second_episode.samples[-1].timestamp_s, scene["rooms"])
    second_result = score_episode(world, second_episode, second_report)
    identity = canonical_json({"report": report, "result": witness, "log": evaluator_log})
    second_identity = canonical_json({"report": second_report, "result": second_result,
                                     "log": make_flight_log(second_episode, scene_id=scene["scene_id"],
                                                             profile_id=profile["profile_id"], include_ground_truth=True)})
    invalid_scenes = check_invalid_scenes(scene)
    checks = {"scene_and_profile_valid": True,
              "three_reachable_rooms": len(world.reachable_rooms) == 3,
              "broken_scenes_rejected": all(item["rejected"] and item["reason_matches"] for item in invalid_scenes),
              "successful_witness": witness["passed"],
              "all_attack_policies_fail": all(not value["passed"] for value in attacks.values()),
              "same_seed_identical": identity == second_identity,
              "runtime_log_excludes_ground_truth": "ground_truth" not in runtime_log}
    result = {"schema_version": "ares.phase0.v1", "seed": seed, "checks": checks,
              "passed": all(checks.values()), "witness": witness, "attacks": attacks,
              "invalid_scenes": invalid_scenes,
              "deterministic_artifacts_sha256": hashlib.sha256(identity.encode()).hexdigest(),
              "reachable_camera_cells": len(world.reachable), "observable_voxels": len(world.observable),
              "limitations": ["Ideal six-axis voxel rays; no D435i noise model or depth image renderer.",
                              "Kinematic evaluator witness uses ground truth; not an exploration baseline.",
                              "Virtual stationary landing at the spawn pad; no flight dynamics or motors.",
                              "No hardware, GPU throughput, or physical flight safety claims."],
              "inputs": {"scene": scene, "profile": profile}}
    return result, {"mission_report": report, "witness.runtime": runtime_log, "witness.evaluator": evaluator_log}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ares", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate a scene and drone profile")
    validate.add_argument("--scene", type=Path)
    validate.add_argument("--profile", type=Path)
    run = sub.add_parser("phase0", help="Run the deterministic foundation gate")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output", type=Path, default=Path("reports/phase_0.json"))
    bench = sub.add_parser("benchmark", help="Measure the versioned paired axial-ray benchmark")
    bench.add_argument("--check", action="store_true")
    bench.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            scene, profile = default_inputs()
            if args.scene:
                scene = load_json(args.scene)
            if args.profile:
                profile = load_json(args.profile)
            validate_scene(scene)
            validate_profile(profile)
            print(json.dumps({"valid": True, "scene": scene["scene_id"], "profile": profile["profile_id"]}))
        elif args.command == "phase0":
            result, artifacts = phase0(seed=args.seed)
            write_json(args.output, result)
            directory = args.output.parent / (args.output.stem + "_artifacts")
            for name, value in artifacts.items():
                write_json(directory / (name + ".json"), value)
            print(json.dumps({"passed": result["passed"], "checks": result["checks"], "output": str(args.output)}))
            return 0 if result["passed"] else 1
        else:
            world = GroundTruthWorld(*default_inputs())
            baseline = load_json(files("ares") / "data/benchmarks/axial_v1.json")
            result = run_benchmark(world, baseline)
            write_json(args.output, result)
            print(json.dumps(result, sort_keys=True))
            return 1 if args.check and not result["passed"] else 0
    except (ContractError, ValueError, OSError) as exc:
        print(f"ares: {exc}", file=sys.stderr)
        return 2
    return 0
