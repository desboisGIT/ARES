"""Frozen-seed exploration benchmarks. Human measurements are never fabricated."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import hashlib
from importlib.resources import files
import json
import math
import multiprocessing
from pathlib import Path
import platform
import time

import numpy as np

from .exploration import ClassicalExplorer, ExplorationConfig, extract_rooms
from .exploration_scoring import build_map_report, score_exploration, EXPLORATION_SCORER_VERSION
from .fast_world import GroundTruthFastWorld, FastWorldConfig, FAST_WORLD_VERSION
from .phase2 import default_profile
from .worldgen import generate_scene, GENERATOR_VERSION


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":"),allow_nan=False).encode()).hexdigest()


def policy_digest():
    return hashlib.sha256(files("ares").joinpath("exploration.py").read_bytes()).hexdigest()


def make_frame(observation):
    """Viewer snapshot, with observed-only map and optional depth holes as null."""
    grid = observation["known_grid"]
    index = int(np.clip(np.floor((observation["home"][2]-observation["origin"][2])/observation["resolution"]),0,grid.shape[2]-1))
    depth = np.asarray(observation["depth_m"])
    rounded = np.round(depth.astype(float),3).astype(object)
    rounded[~np.isfinite(depth)] = None
    return {"position": observation["position"].astype(float).tolist(),"velocity": observation["velocity"].astype(float).tolist(),
        "home": observation["home"].tolist(),"yaw": observation["yaw"],"time_s": observation["time_s"],
        "depth_m": rounded.tolist(),"observed_voxels": int(np.count_nonzero(grid)),
        "known_floor": {"resolution": observation["resolution"],"origin": observation["origin"][:2].tolist(),
                        "shape": list(grid.shape[:2]),"cells": grid[:,:,index].T.ravel().astype(int).tolist()}}


def run_classical_episode(scene, profile, *, backend="cuda", config=None, policy_config=None, record=False):
    config = config or FastWorldConfig()
    world = GroundTruthFastWorld(scene,profile,backend=backend,seed=scene["seed"],config=config)
    actor = ClassicalExplorer(policy_config)
    actions, frames = [], []
    started = time.perf_counter()
    if record:
        frames.append(make_frame(world.observation()))
    for step in range(math.ceil(config.max_time_s/config.dt)):
        obs = world.observation()
        action = actor.act(obs)
        world.step(action[None])
        if record:
            actions.append(np.asarray(action,dtype=np.float32).astype(float).tolist())
        state = world.ground_truth_state()
        if record and (step%10==0 or state["done"]):
            frames.append(make_frame(world.observation()))
        if state["done"]:
            break
    final = world.observation()
    rooms = extract_rooms(final["known_grid"],world.resolution,world.origin)
    report = build_map_report(final,rooms)
    result = score_exploration(world,world.snapshot_for_scoring(),report)
    result.update(actor="classical",policy_name=actor.policy_name,policy_version=actor.policy_version,
                  policy_sha256=policy_digest(),wall_time_s=time.perf_counter()-started,
                  policy_config=asdict(actor.config),diagnostics=actor.diagnostics,backend=backend,world_config=asdict(config))
    log = None
    if record:
        log = {"schema_version":"ares.exploration_log.v1","actor":"classical","seed":scene["seed"],
            "scene":scene,"profile":profile,"world_config":asdict(config),"backend":backend,
            "world_version":FAST_WORLD_VERSION,"scorer_version":EXPLORATION_SCORER_VERSION,
            "scene_sha256":_hash(scene),"completed":bool(world.ground_truth_state()["done"]),
            "policy_sha256":policy_digest(),"policy_config":asdict(actor.config),
            "actions":actions,"frames":frames,"report":report,"result":result,
            "wall_duration_s":result["wall_time_s"],"realtime_eligible":False,
            "note":"Automated simulation; never a human baseline."}
    return result,log


def _episode_worker(arguments):
    seed,split,profile,backend,configuration,policy_configuration,log_dir = arguments
    scene = generate_scene(seed,"apartment",split)
    result,log = run_classical_episode(scene,profile,backend=backend,config=FastWorldConfig(**configuration),
        policy_config=ExplorationConfig(**policy_configuration),record=log_dir is not None)
    result["scene_sha256"] = _hash(scene)
    result["split"] = split
    if log is not None:
        destination = Path(log_dir)/f"classical-{split}-{seed}.json.gz"
        destination.parent.mkdir(parents=True,exist_ok=True)
        # One benchmark invocation owns its seed-named outputs; refuse an
        # existing log rather than replacing a prior human/operator artifact.
        with destination.open("xb") as stream:
            with gzip.GzipFile(fileobj=stream,mode="wb",mtime=0) as compressed:
                compressed.write(json.dumps(log,allow_nan=False,separators=(",", ":")).encode())
        result["log_path"] = str(destination)
    return result


def benchmark_fast_world(scene,profile,*,backend="cuda",num_envs=4096,steps=25):
    if type(steps) is not int or steps<2:
        raise ValueError("benchmark steps must be an integer >=2")
    world = GroundTruthFastWorld(scene,profile,num_envs=num_envs,backend=backend,seed=1741)
    action = np.zeros((num_envs,5),dtype=np.float32)
    action[:,0],action[:,1],action[:,3] = .03*np.cos(np.arange(num_envs)),.03*np.sin(np.arange(num_envs)),.7
    for _ in range(5):
        world.step(action)
    world.synchronize()
    before = int(world._host(world._steps).sum())
    times = []
    for _ in range(steps):
        started = time.perf_counter()
        world.step(action)
        world.synchronize()
        times.append(time.perf_counter()-started)
    actual = int(world._host(world._steps).sum())-before
    device = None
    if backend=="cuda":
        properties = world.xp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        device = {"name":name.decode() if isinstance(name,bytes) else name,"cupy":world.xp.__version__}
    rate = actual/sum(times)
    return {"schema_version":"ares.fast_world_benchmark.v1","backend":backend,"num_envs":num_envs,
        "steps_per_environment":steps,"individual_environment_steps":actual,"wall_seconds":sum(times),
        "environment_steps_per_second":rate,"median_batch_s":float(np.median(times)),
        "p90_batch_s":float(np.percentile(times,90)),"all_environments_active":actual==num_envs*steps,
        "done_environments":int(world._host(world._done).sum()),"camera":[world.width,world.height],
        "device":device,"world_config":asdict(world.config),"python":platform.python_version(),"numpy":np.__version__,
        "passed":bool(backend=="cuda" and num_envs>=4096 and actual==num_envs*steps and rate>=100000),
        "scope":"Dynamics, exact swept-sphere collision, yaw, forward-cone raycasting and atomic observed-map updates; synchronized host timing, no policy/scorer/file logging or observation downloads; warmup excluded."}


def summarize_results(results):
    count = len(results)
    passing = [result["time_s"] for result in results if result["passed"]]
    passed = len(passing)
    fraction = passed/count if count else None
    interval = None
    if count:
        z=1.95996398454
        center=(fraction+z*z/(2*count))/(1+z*z/count)
        half=z*math.sqrt(fraction*(1-fraction)/count+z*z/(4*count*count))/(1+z*z/count)
        interval=[max(0.,center-half),min(1.,center+half)]
    return {"runs":count,"passed_runs":passed,"pass_rate":fraction,
        "pass_rate_wilson_95_interval":interval,"median_time_s":float(np.median(passing)) if passing else None,
        "p90_time_s":float(np.percentile(passing,90)) if passing else None,
        "time_statistics_population":"passing runs only; failures remain in pass-rate denominator",
        "collision_segments":sum(result["collisions"] for result in results)}


def run_phase3(*,backend="cpu",num_envs=4,benchmark_steps=5,episodes=1,start_seed=100,split="train",
               workers=1,output_dir=Path("reports/phase_3_artifacts"),progress=None,record_logs=False):
    if (type(episodes) is not int or episodes<1 or type(workers) is not int or workers<1
            or type(start_seed) is not int or split not in ("train","validation","test")):
        raise ValueError("positive episodes/workers and integer start_seed/valid split required")
    profile = default_profile()
    directory = Path(output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    policy_hash = policy_digest()
    # Throughput uses a training scene even when evaluation scenes are held out.
    benchmark = benchmark_fast_world(generate_scene(100,"apartment","train"),profile,
                                     backend=backend,num_envs=num_envs,steps=benchmark_steps)
    world_configuration,policy_configuration = asdict(FastWorldConfig()),asdict(ExplorationConfig())
    args = [(seed,split,profile,backend,world_configuration,policy_configuration,
             str(directory/"logs") if record_logs else None) for seed in range(start_seed,start_seed+episodes)]
    results=[]
    started=time.perf_counter()
    if workers==1:
        for arguments in args:
            result=_episode_worker(arguments)
            results.append(result)
            if progress:progress(len(results),episodes,result)
    else:
        # Spawn avoids inheriting a live CUDA context or the parent's worker threads.
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as executor:
            futures=[executor.submit(_episode_worker,arguments) for arguments in args]
            for future in as_completed(futures):
                result=future.result()
                results.append(result)
                if progress:progress(len(results),episodes,result)
    results.sort(key=lambda item:item["seed"])
    if any(result["policy_sha256"]!=policy_hash for result in results) or policy_digest()!=policy_hash:
        raise ValueError("policy changed during evaluation; results cannot be certified as held out")
    summary=summarize_results(results)
    dataset={"generator_version":GENERATOR_VERSION,"kind":"apartment","split":split,
             "seeds":[result["seed"] for result in results],"scene_sha256":[result["scene_sha256"] for result in results]}
    heldout=split=="test" and episodes>=100
    baseline_passed=heldout and summary["pass_rate"]>=.9
    output={"schema_version":"ares.phase3.v1","world_version":FAST_WORLD_VERSION,
        "scorer_version":EXPLORATION_SCORER_VERSION,"policy_sha256":policy_hash,"policy_config":policy_configuration,
        "world_config":world_configuration,"profile":profile,"dataset":dataset,"dataset_sha256":_hash(dataset),
        "workers":workers,"benchmark":benchmark,"classical":summary,"episodes":results,
        "classical_apartment_gate":{"passed":baseline_passed,"held_out":heldout,"required_runs":100,"required_pass_rate":.9},
        "human":{"status":"pending_actual_human_runs","runs":0,"median_time_s":None,"p90_time_s":None,
                 "pass_rate":None,"collision_segments":None},
        "software_smoke_passed":all(result["passed"] for result in results),
        "phase_passed":False,"evaluation_wall_s":time.perf_counter()-started,
        "phase2_hardware_status":"still_pending; this phase uses ideal state/voxel sensing",
        "limitations":["Point-mass velocity control, perfect pose and ideal voxel cone; no flight/odometry/material dynamics.",
                       "Independent simulated environments share one scene per vectorized batch; drones do not collide with each other.",
                       "Room extraction uses the explicit rectangular-world above-door layer prior.",
                       "Human times cannot be filled by scripted or agent UI runs."]}
    (directory/"baseline_results.json").write_text(json.dumps(output,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return output
