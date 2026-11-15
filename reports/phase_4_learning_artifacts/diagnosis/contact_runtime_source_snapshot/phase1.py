"""Reproducible corpus validation, throughput evidence, and human-review binding."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

from .contracts import load_json, validate_scene
from .gallery import build_gallery, validate_review
from .importer import import_recorded_map_with_provenance
from .randomness import seeded_rng
from .worldgen import GENERATOR_VERSION, KINDS, generate_scene, geometry_digest


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_geometry(scene):
    """Independent surface-area/volume and spawn checks on every generated case."""
    validate_scene(scene)
    grid = scene["grid"]
    occupied = set(map(tuple, grid["occupied"]))
    res = grid["resolution"]
    nx, ny, nz = grid["shape"]
    shell = ((x, y, z) for x in range(nx) for y in range(ny)
             for z in ((0, nz - 1) if 0 < x < nx - 1 and 0 < y < ny - 1 else range(nz)))
    if any(cell not in occupied for cell in shell):
        raise ValueError("generator: building shell is not enclosed")
    spawn = scene["spawn"]
    index = [math.floor((spawn[i] - grid["origin"][i]) / res) for i in range(3)]
    for x in range(index[0] - 1, index[0] + 2):
        for y in range(index[1] - 1, index[1] + 2):
            for z in range(index[2] - 1, index[2] + 2):
                if (x, y, z) in occupied:
                    distance = sum(max(grid["origin"][i] + cell * res - spawn[i],
                                       spawn[i] - (grid["origin"][i] + (cell + 1) * res), 0) ** 2
                                   for i, cell in enumerate((x, y, z)))
                    if distance <= .15 ** 2:
                        raise ValueError("generator: spawn lacks nominal 0.15m body clearance")
    faces = 0
    for x, y, z in occupied:
        faces += sum(cell not in occupied for cell in ((x-1,y,z),(x+1,y,z),(x,y-1,z),
                                                     (x,y+1,z),(x,y,z-1),(x,y,z+1)))
    vertices = scene["collision_mesh"]["vertices"]
    area, volume = 0.0, 0.0
    for triangle in scene["collision_mesh"]["triangles"]:
        a, b, c = (vertices[i] for i in triangle)
        u, v = [b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)]
        cross = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
        area += math.sqrt(sum(n*n for n in cross)) / 2
        volume += sum(a[i] * cross[i] for i in range(3)) / 6
    if not math.isclose(area, faces * res ** 2, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("generator: collision mesh surface area disagrees with voxels")
    if not math.isclose(volume, len(occupied) * res ** 3, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("generator: collision mesh signed volume disagrees with voxels")


def _case(arguments):
    seed, kind, split = arguments
    started = time.perf_counter()
    try:
        scene = generate_scene(seed, kind, split)
        validate_geometry(scene)
        # Independently record both geometric and full-content identities.
        content = {key: value for key, value in scene.items() if key not in ("scene_id", "seed")}
        return {"seed": seed, "kind": kind, "split": split, "scene_id": scene["scene_id"],
                "valid": True, "sha256": canonical_digest(scene), "content_sha256": canonical_digest(content),
                "geometry_sha256": geometry_digest(scene),
                "rooms": len(scene["rooms"]), "occupied_voxels": len(scene["grid"]["occupied"]),
                "mesh_triangles": len(scene["collision_mesh"]["triangles"]),
                "materials": sorted({s["material"] for s in scene["surfaces"]}),
                "door_states": sorted({d["state"] for d in scene["doors"]}),
                "lighting_states": sorted({l["state"] for l in scene["lighting"]}),
                "seconds": time.perf_counter() - started}
    except (ValueError, TypeError, OverflowError, RuntimeError) as exc:
        return {"seed": seed, "kind": kind, "split": split, "valid": False,
                "error": str(exc), "seconds": time.perf_counter() - started}


def run_phase1(*, count=10000, seed=1741, workers=1, output_dir=Path("reports/phase_1_artifacts"),
               review_path=None, progress=None):
    if type(count) is not int or count < 1 or type(seed) is not int or type(workers) is not int or workers < 1:
        raise ValueError("count/workers must be positive integers and seed an integer")
    cases = [(seed + i, KINDS[i % 3], "test" if i % 10 == 0 else "validation" if i % 10 == 1 else "train")
             for i in range(count)]
    start = time.perf_counter()
    records = []
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        outcomes = executor.map(_case, cases, chunksize=8) if executor else map(_case, cases)
        for record in outcomes:
            records.append(record)
            if progress and (len(records) % 250 == 0 or len(records) == count):
                progress(len(records), count, time.perf_counter() - start)
    finally:
        if executor:
            executor.shutdown()
    elapsed = time.perf_counter() - start
    valid = [item for item in records if item["valid"]]
    content_partitions = {}
    leaked = []
    for record in valid:
        previous = content_partitions.setdefault(record["geometry_sha256"], record["split"])
        if previous != record["split"]:
            leaked.append(record["scene_id"])
    material = sorted({value for item in valid for value in item["materials"]})
    doors = sorted({value for item in valid for value in item["door_states"]})
    lights = sorted({value for item in valid for value in item["lighting_states"]})
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    identity = [{key: value for key, value in record.items() if key != "seconds"} for record in records]
    corpus = {"schema_version": "ares.scene_corpus.v1", "generator_version": GENERATOR_VERSION,
              "seed": seed, "count": count, "cases": identity}
    corpus["corpus_id"] = canonical_digest(corpus)
    (directory / "corpus_manifest.json").write_text(json.dumps(corpus, indent=2) + "\n")
    candidates = [record for record in valid if record["split"] == "train"]
    picked = seeded_rng(seed, "ares.review_selection.v1").sample(candidates, min(50, len(candidates)))
    gallery = None
    human = {"passed": False, "status": "pending_human_review"}
    if picked:
        scenes = [generate_scene(case["seed"], case["kind"], case["split"]) for case in picked]
        gallery = build_gallery(scenes, directory / "review", seed=seed)
        if review_path:
            human = validate_review(load_json(review_path), gallery)
    # Synthetic fixture verifies the adapter end to end; no claim of a real capture.
    from importlib.resources import files
    imported, provenance = import_recorded_map_with_provenance(load_json(
        files("ares") / "data/recorded_maps/synthetic_enclosed_room.json"))
    importer = {"passed": True, "evidence": "synthetic_adapter_fixture_only",
                "scene_id": imported["scene_id"], "provenance": provenance}
    checks = {"all_requested_scenes_valid": len(valid) == count,
              "at_least_10000_valid_scenes": len(valid) >= 10000,
              "at_least_one_scene_per_second": count / elapsed >= 1,
              "all_three_difficulty_levels": {item["kind"] for item in valid} == set(KINDS),
              "all_depth_materials_present": material == ["black", "glass", "mirror", "normal", "shiny"],
              "open_and_closed_doors_present": doors == ["closed", "open"],
              "lit_and_dark_rooms_present": lights == ["dark", "lit"],
              "no_geometry_leak_between_splits": not leaked,
              "recorded_map_adapter_fixture": importer["passed"]}
    software = all(checks.values())
    return {"schema_version": "ares.phase1.v1", "generator_version": GENERATOR_VERSION,
            "seed": seed, "requested_count": count, "valid_count": len(valid), "workers": workers,
            "generation_validation_hashing_seconds": elapsed, "scenes_per_second": count / elapsed,
            "throughput_scope": "generation + full schema, shell, spawn, area/volume validation + hashing; excludes gallery and disk scene serialization",
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "logical_cpus": os.cpu_count()},
            "checks": checks, "software_passed": software, "phase_passed": software and human["passed"],
            "human_review": human, "gallery": gallery, "corpus_id": corpus["corpus_id"],
            "failures": [item for item in records if not item["valid"]], "cross_split_duplicates": leaked,
            "counts_by_kind": {kind: sum(item["kind"] == kind for item in valid) for kind in KINDS},
            "counts_by_split": {split: sum(item["split"] == split for item in valid) for split in ("train", "validation", "test")},
            "importer": importer}
