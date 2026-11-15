"""Offline world generation, data import, and phased evidence commands."""

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
    generate = sub.add_parser("generate", help="Generate one deterministic procedural building")
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--kind", choices=("single_room", "apartment", "floor"), default="apartment")
    generate.add_argument("--split", choices=("train", "validation", "test"), default="train")
    generate.add_argument("--output", type=Path, required=True)
    imported = sub.add_parser("import-map", help="Convert an annotated recorded occupancy map")
    imported.add_argument("input", type=Path)
    imported.add_argument("--output", type=Path, required=True)
    worlds = sub.add_parser("phase1", help="Validate a procedural corpus and prepare human review")
    worlds.add_argument("--count", type=int, default=10000)
    worlds.add_argument("--seed", type=int, default=1741)
    worlds.add_argument("--workers", type=int, default=1)
    worlds.add_argument("--output", type=Path, default=Path("reports/phase_1.json"))
    worlds.add_argument("--review", type=Path)
    worlds.add_argument("--smoke", action="store_true", help="Allow a small CI sample; does not satisfy the 10,000-scene gate")
    worlds.add_argument("--require-complete", action="store_true", help="Fail unless software and human gates both pass")
    sensors = sub.add_parser("phase2", help="Check synthetic sensor software and measure depth rendering")
    sensors.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    sensors.add_argument("--batch-size", type=int, default=4)
    sensors.add_argument("--frames", type=int, default=12)
    sensors.add_argument("--seed", type=int, default=1741)
    sensors.add_argument("--scene", type=Path)
    sensors.add_argument("--profile", type=Path)
    sensors.add_argument("--output", type=Path, default=Path("reports/phase_2.json"))
    sensors.add_argument("--require-complete", action="store_true")
    explore = sub.add_parser("phase3", help="Benchmark the fast world and frozen classical explorer")
    explore.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    explore.add_argument("--num-envs", type=int, default=4)
    explore.add_argument("--benchmark-steps", type=int, default=5)
    explore.add_argument("--episodes", type=int, default=1)
    explore.add_argument("--start-seed", type=int, default=100)
    explore.add_argument("--split", choices=("train", "validation", "test"), default="train")
    explore.add_argument("--workers", type=int, default=1)
    explore.add_argument("--record-logs", action="store_true")
    explore.add_argument("--output", type=Path, default=Path("reports/phase_3.json"))
    explore.add_argument("--smoke", action="store_true", help="Check integration without certifying the full phase")
    explore.add_argument("--require-complete", action="store_true")
    flight = sub.add_parser("phase4", help="Train and compare differentiable and PPO flight policies")
    flight.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    flight.add_argument("--algorithms", choices=("bptt","ppo"), nargs='+', default=['bptt','ppo'])
    flight.add_argument("--residual-acceleration", type=float, default=None)
    flight.add_argument("--train-updates", type=int, default=1)
    flight.add_argument("--batch-size", type=int, default=2)
    flight.add_argument("--rollout-steps", type=int, default=4)
    flight.add_argument("--episodes", type=int, default=2)
    flight.add_argument("--max-steps", type=int, default=10)
    flight.add_argument("--shield-episodes", type=int, default=4)
    flight.add_argument("--eval-split", choices=("train","validation","test"), default="test")
    flight.add_argument("--eval-start-seed", type=int, default=70000)
    flight.add_argument("--seed", type=int, default=1741)
    flight.add_argument("--output", type=Path, default=Path("reports/phase_4_repair.json"))
    flight.add_argument("--smoke", action="store_true")
    flight.add_argument("--require-complete", action="store_true")
    flight_eval = sub.add_parser("evaluate-flight", help="Evaluate frozen flight weights without retraining")
    flight_eval.add_argument("checkpoint", type=Path)
    flight_eval.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    flight_eval.add_argument("--episodes", type=int, default=100)
    flight_eval.add_argument("--batch-size", type=int, default=16)
    flight_eval.add_argument("--max-steps", type=int, default=600)
    flight_eval.add_argument("--start-seed", type=int, default=70000)
    flight_eval.add_argument("--split", choices=("train","validation","test"), default="validation")
    flight_eval.add_argument("--shield", action="store_true")
    flight_eval.add_argument("--output", type=Path, required=True)
    control = sub.add_parser("control-check", help="Measure observed-state hover, braking and hold on development dynamics")
    control.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    control.add_argument("--count", type=int, default=8)
    control.add_argument("--seed", type=int, default=31)
    control.add_argument("--duration", type=float, default=4.)
    control.add_argument("--settle", type=float, default=2.)
    control.add_argument("--batch-size", type=int, default=64)
    control.add_argument("--compile-dynamics", action="store_true")
    control.add_argument("--no-nominal", action="store_true")
    control.add_argument("--noise-position", type=float, default=.005)
    control.add_argument("--noise-velocity", type=float, default=.01)
    control.add_argument("--added-command-delay", type=float, default=0.)
    control.add_argument("--shield-period", type=float, default=.01)
    control.add_argument("--initial-tilt", type=float, default=0.)
    control.add_argument("--output", type=Path, default=Path("reports/control_validation.json"))
    fly = sub.add_parser("fly", help="Fly the point-mass simulator in the local browser")
    fly.add_argument("--seed", type=int, default=100)
    fly.add_argument("--split", choices=("train", "validation", "test"), default="train")
    fly.add_argument("--scene", type=Path)
    fly.add_argument("--profile", type=Path)
    fly.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    fly.add_argument("--actor", choices=("human", "ui_test"), default="human")
    fly.add_argument("--log-dir", type=Path, default=Path("logs/manual"))
    fly.add_argument("--port", type=int, default=8766)
    view = sub.add_parser("view-log", help="Inspect a flight, exploration or sensor log locally")
    view.add_argument("log", type=Path)
    view.add_argument("--port", type=int, default=8766)
    human = sub.add_parser("human-benchmark", help="Replay and add actual human logs to a Phase 3 result")
    human.add_argument("logs", type=Path, nargs="+")
    human.add_argument("--baseline", type=Path, default=Path("reports/phase_3.json"))
    human.add_argument("--output", type=Path, default=Path("reports/phase_3_human.json"))
    replay = sub.add_parser("replay", help="Run CPU depth odometry and mapping on a sensor recording")
    replay.add_argument("recording", type=Path)
    replay.add_argument("--max-frames", type=int)
    replay.add_argument("--profile", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    synthetic = sub.add_parser("simulate-recording", help="Write a clearly labelled synthetic depth/IMU fixture")
    synthetic.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    synthetic.add_argument("--frames", type=int, default=24)
    synthetic.add_argument("--seed", type=int, default=1741)
    synthetic.add_argument("--scene", type=Path)
    synthetic.add_argument("--profile", type=Path)
    synthetic.add_argument("--output", type=Path, required=True)
    fit = sub.add_parser("calibrate-depth", help="Fit noise from independently registered depth/reference pairs")
    fit.add_argument("pairs", type=Path, help="NPZ: measured_m, reference_m, material_ids")
    fit.add_argument("--source", choices=("synthetic", "real"), required=True)
    fit.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare-depth", help="Compare a recording to an imported scene at registered reference poses")
    compare.add_argument("recording", type=Path)
    compare.add_argument("--scene", type=Path, required=True)
    compare.add_argument("--profile", type=Path)
    compare.add_argument("--poses", type=Path, required=True, help="NPZ: timestamps_s, camera_to_world (F,4,4)")
    compare.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    compare.add_argument("--pairs-output", type=Path)
    compare.add_argument("--output", type=Path, required=True)
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
        elif args.command == "generate":
            from .worldgen import generate_scene
            from .phase1 import validate_geometry
            scene = generate_scene(args.seed, args.kind, args.split)
            validate_geometry(scene)
            write_json(args.output, scene)
            print(json.dumps({"valid": True, "scene_id": scene["scene_id"], "output": str(args.output)}))
        elif args.command == "import-map":
            from .importer import import_recorded_map_with_provenance
            scene, provenance = import_recorded_map_with_provenance(load_json(args.input))
            write_json(args.output, scene)
            write_json(args.output.with_name(args.output.stem + "_provenance.json"), provenance)
            print(json.dumps({"valid": True, "scene_id": scene["scene_id"], "output": str(args.output)}))
        elif args.command == "phase1":
            from .phase1 import run_phase1
            directory = args.output.parent / (args.output.stem + "_artifacts")
            result = run_phase1(count=args.count, seed=args.seed, workers=args.workers,
                                output_dir=directory, review_path=args.review,
                                progress=lambda done, total, elapsed: print(
                                    f"Validated {done}/{total} scenes in {elapsed:.1f}s", file=sys.stderr))
            write_json(args.output, result)
            print(json.dumps({key: result[key] for key in ("software_passed", "phase_passed", "valid_count", "scenes_per_second", "human_review")}))
            if args.require_complete:
                return 0 if result["phase_passed"] else 1
            if args.smoke:
                return 0 if all(value for name, value in result["checks"].items()
                                if name != "at_least_10000_valid_scenes") else 1
            return 0 if result["software_passed"] else 1
        elif args.command == "phase2":
            from .phase2 import default_profile, run_phase2
            scene = load_json(args.scene) if args.scene else default_inputs()[0]
            profile = load_json(args.profile) if args.profile else default_profile()
            directory = args.output.parent / (args.output.stem + "_artifacts")
            result = run_phase2(scene, profile, directory, backend=args.backend,
                                batch_size=args.batch_size, frames=args.frames, seed=args.seed)
            write_json(args.output, result)
            print(json.dumps({key: result[key] for key in ("software_preparation_passed", "phase_passed",
                "software_checks", "gpu_thousands_gate", "hardware_gates")}))
            return 0 if (result["phase_passed"] if args.require_complete else result["software_preparation_passed"]) else 1
        elif args.command == "phase3":
            from .phase3 import run_phase3
            result = run_phase3(backend=args.backend, num_envs=args.num_envs,
                benchmark_steps=args.benchmark_steps, episodes=args.episodes,
                start_seed=args.start_seed, split=args.split, workers=args.workers,
                record_logs=args.record_logs,
                output_dir=args.output.parent / (args.output.stem + "_artifacts"),
                progress=lambda done,total,item: print(
                    f"Explored {done}/{total}: seed={item['seed']} passed={item['passed']} time={item['time_s']:.2f}s collisions={item['collisions']}",
                    file=sys.stderr, flush=True))
            write_json(args.output,result)
            print(json.dumps({key:result[key] for key in ("benchmark","classical","classical_apartment_gate","human","phase_passed")}))
            if args.require_complete:
                return 0 if result["phase_passed"] else 1
            return 0 if (result["software_smoke_passed"] if args.smoke else
                         result["benchmark"]["passed"] and result["classical_apartment_gate"]["passed"]) else 1
        elif args.command == "control-check":
            from .control_validation import run_control_validation
            from .phase2 import default_profile
            result=run_control_validation(default_profile(),device=args.device,count=args.count,seed=args.seed,
                duration_s=args.duration,settle_s=args.settle,batch_size=args.batch_size,
                compile_dynamics=args.compile_dynamics,include_nominal=not args.no_nominal,
                noise_bounds={'position_m':args.noise_position,'velocity_mps':args.noise_velocity},
                command_delay_s=args.added_command_delay,shield_interval_s=args.shield_period,
                initial_tilt_deg=args.initial_tilt,
                output_dir=args.output.parent/(args.output.stem+'_artifacts'))
            repaired=result['summary']['repaired']
            passed=repaired['initial_classes'].get('valid_brakable',0)>0 and repaired['valid_brakable_failures']==0
            result['development_regression_passed']=passed
            write_json(args.output,result)
            print(json.dumps({'development_regression_passed':passed,'summary':result['summary'],
                'shield_certification_passed':False,'output':str(args.output)}))
            return 0 if passed else 1
        elif args.command == "evaluate-flight":
            import torch
            from .flight_training import load_checkpoint
            from .phase4 import evaluate_flight_policy
            from . import phase4 as phase4_module
            torch.set_num_threads(1)
            policy,payload=load_checkpoint(args.checkpoint,device=args.device)
            source_hashes=phase4_module.runtime_source_hashes()
            metadata=payload.get('metadata',{})
            if not isinstance(metadata.get('profile'),dict):
                raise ValueError('flight checkpoint has no training profile snapshot')
            for name,expected in metadata.get('runtime_source_sha256',{}).items():
                relative=Path(name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('invalid checkpoint source identifier')
                actual=hashlib.sha256((Path(phase4_module.__file__).parent/relative).read_bytes()).hexdigest()
                if actual!=expected:
                    raise ValueError('flight runtime differs from the frozen checkpoint: '+name)
            frozen_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            result=evaluate_flight_policy(policy,metadata['profile'],device=args.device,
                episodes=args.episodes,batch_size=args.batch_size,max_steps=args.max_steps,
                start_seed=args.start_seed,split=args.split,shield=args.shield,
                progress=lambda item:print(json.dumps(item),file=sys.stderr,flush=True))
            phase4_module.require_runtime_unchanged(source_hashes)
            if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()!=frozen_checkpoint_sha256:
                raise ValueError('checkpoint changed during held-out evaluation')
            result.update(schema_version='ares.flight_evaluation.v1',algorithm=payload['algorithm'],
                          checkpoint_sha256=frozen_checkpoint_sha256,
                          runtime_source_sha256=source_hashes)
            write_json(args.output,result)
            print(json.dumps(result['summary']))
        elif args.command == "phase4":
            from .phase4 import run_phase4
            def flight_progress(value):
                print(json.dumps(value,allow_nan=False),file=sys.stderr,flush=True)
            result=run_phase4(device=args.device,train_updates=args.train_updates,
                batch_size=args.batch_size,rollout_steps=args.rollout_steps,episodes=args.episodes,
                max_steps=args.max_steps,seed=args.seed,shield_episodes=args.shield_episodes,
                eval_split=args.eval_split,
                eval_start_seed=args.eval_start_seed,
                algorithms=args.algorithms,residual_acceleration=args.residual_acceleration,
                output_dir=args.output.parent/(args.output.stem+"_artifacts"),progress=flight_progress)
            write_json(args.output,result)
            print(json.dumps({'software_preparation_passed':result['software_preparation_passed'],
                'waypoint_reliability_gate':result['waypoint_reliability_gate'],
                'shield_gate':result['shield_gate'],'onboard_inference_gate':result['onboard_inference_gate'],
                'summary':{key:value['summary'] for key,value in result['evaluation'].items()},
                'phase_passed':result['phase_passed'],'output':str(args.output)}))
            return 0 if (result['phase_passed'] if args.require_complete else
                         result['software_preparation_passed'] if args.smoke else
                         result['waypoint_reliability_gate']['passed']) else 1
        elif args.command == "human-benchmark":
            from .human_benchmark import summarize_human_logs
            from .phase3 import _hash, policy_digest
            from .fast_world import FAST_WORLD_VERSION
            from .exploration_scoring import EXPLORATION_SCORER_VERSION
            result = load_json(args.baseline)
            if not isinstance(result,dict):
                raise ValueError("baseline must be a Phase 3 result object")
            if (result.get("schema_version") != "ares.phase3.v1" or
                    result.get("world_version") != FAST_WORLD_VERSION or
                    result.get("scorer_version") != EXPLORATION_SCORER_VERSION or
                    result.get("policy_sha256") != policy_digest() or
                    result.get("dataset_sha256") != _hash(result.get("dataset"))):
                raise ValueError("baseline identity differs from the frozen Phase 3 benchmark")
            if (not isinstance(result.get("dataset"),dict) or
                    not isinstance(result["dataset"].get("scene_sha256"),list) or
                    not result["dataset"]["scene_sha256"] or
                    not all(isinstance(value,str) and len(value)==64 for value in result["dataset"]["scene_sha256"]) or
                    not isinstance(result.get("profile"),dict) or
                    not isinstance(result.get("world_config"),dict) or
                    not isinstance(result.get("benchmark"),dict) or
                    type(result["benchmark"].get("passed")) is not bool or
                    not isinstance(result.get("classical_apartment_gate"),dict) or
                    type(result["classical_apartment_gate"].get("passed")) is not bool):
                raise ValueError("baseline is missing its cohort, configuration or gate results")
            verified = summarize_human_logs(args.logs,
                expected_scene_sha256=result["dataset"]["scene_sha256"],
                expected_profile=result["profile"], expected_world_config=result["world_config"])
            result["human"] = verified["summary"] | {
                "status": "measured" if verified["records"] else "pending_actual_human_runs",
                "records": verified["records"], "rejected": verified["rejected"],
                "trust_note": verified["trust_note"],
                "metric_comparison_tolerances": verified["metric_comparison_tolerances"]}
            result["phase_passed"] = bool(result["benchmark"]["passed"] and
                result["classical_apartment_gate"]["passed"] and verified["records"])
            write_json(args.output,result)
            print(json.dumps({"human":result["human"],"phase_passed":result["phase_passed"],"output":str(args.output)}))
            return 0 if verified["records"] and not verified["rejected"] else 1
        elif args.command == "fly":
            from .sessions import ManualSession
            from .phase2 import default_profile
            from .viewer import serve_viewer
            from .worldgen import generate_scene
            scene = load_json(args.scene) if args.scene else generate_scene(args.seed,"apartment",args.split)
            profile = load_json(args.profile) if args.profile else default_profile()
            session = ManualSession(scene,profile,backend=args.backend,actor=args.actor,log_dir=args.log_dir)
            print(f"Simulated flight ({args.actor}): http://127.0.0.1:{args.port} — Ctrl-C closes and saves the run",flush=True)
            serve_viewer(session,port=args.port)
        elif args.command == "view-log":
            from .sessions import ReplaySession
            from .viewer import serve_viewer
            session = ReplaySession(args.log)
            print(f"Replay: http://127.0.0.1:{args.port}",flush=True)
            serve_viewer(session,port=args.port)
        elif args.command == "replay":
            from .replay import replay_recording
            result = replay_recording(args.recording, max_frames=args.max_frames,
                                      profile=load_json(args.profile) if args.profile else None)
            write_json(args.output, result)
            print(json.dumps({key: result[key] for key in ("recording_id", "source", "frames", "accepted_frames", "tracking_fraction")}))
            return 0 if result["accepted_frames"] else 1
        elif args.command == "simulate-recording":
            from .phase2 import default_profile, write_synthetic_recording
            scene = load_json(args.scene) if args.scene else default_inputs()[0]
            profile = load_json(args.profile) if args.profile else default_profile()
            result = write_synthetic_recording(args.output, scene, profile, backend=args.backend,
                                               frames=args.frames, seed=args.seed)
            print(json.dumps({"output": str(args.output), "source": result["source"], "frames": result["frames"]}))
        elif args.command == "calibrate-depth":
            import numpy as np
            from .calibration import fit_depth_noise
            with np.load(args.pairs, allow_pickle=False) as pairs:
                if len(pairs.files) != 3 or set(pairs.files) != {"measured_m", "reference_m", "material_ids"}:
                    raise ValueError("pairs archive requires exactly measured_m, reference_m, material_ids")
                result = fit_depth_noise(pairs["measured_m"], pairs["reference_m"], pairs["material_ids"], source=args.source)
            write_json(args.output, result)
            print(json.dumps({key: result[key] for key in ("source", "base_std_m", "quadratic_std_per_m2", "residual_rmse_m")}))
        elif args.command == "compare-depth":
            from .depth_comparison import compare_registered_depth
            from .phase2 import default_profile
            result = compare_registered_depth(args.recording, load_json(args.scene),
                load_json(args.profile) if args.profile else default_profile(), args.poses,
                backend=args.backend, pairs_output=args.pairs_output)
            write_json(args.output, result)
            print(json.dumps({key: result[key] for key in ("source", "compared_pixels", "rmse_m", "registration_status")}))
        elif args.command == "benchmark":
            world = GroundTruthWorld(*default_inputs())
            baseline = load_json(files("ares") / "data/benchmarks/axial_v1.json")
            result = run_benchmark(world, baseline)
            write_json(args.output, result)
            print(json.dumps(result, sort_keys=True))
            return 1 if args.check and not result["passed"] else 0
    except (ContractError, ValueError, OSError, RuntimeError, ImportError, FloatingPointError) as exc:
        print(f"ares: {exc}", file=sys.stderr)
        return 2
    return 0
