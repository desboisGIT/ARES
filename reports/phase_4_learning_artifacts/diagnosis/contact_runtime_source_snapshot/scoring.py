"""Authoritative terminal score, separate from any dense training reward."""

from dataclasses import asdict, dataclass
import math

from . import SCORER_VERSION
from .evaluator import point_in_polygon
from .reporting import validate_report, make_report


@dataclass(frozen=True)
class ScoringRules:
    coverage_threshold: float = 0.95
    floor_plan_iou_threshold: float = 0.80
    timeout_s: float = 3600.0
    room_polygon_iou_threshold: float = 0.80

    def __post_init__(self):
        for name in ("coverage_threshold", "floor_plan_iou_threshold", "room_polygon_iou_threshold"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0,1]")
        if type(self.timeout_s) not in (int, float) or not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")


def _iou(truth, claimed):
    union = truth | claimed
    return len(truth & claimed) / len(union) if union else 1.0


def score_episode(world, episode, report, rules=None):
    """world/episode are evaluator-owned; report is the only policy-owned input."""
    validate_report(report)
    rules = rules or ScoringRules()
    reasons = []
    if episode.world_digest != world.digest:
        raise ValueError("episode ground truth belongs to a different scene/profile")
    elapsed = episode.samples[-1].timestamp_s
    if abs(report["end_time_s"] - elapsed) > 1e-9:
        reasons.append("report_time_mismatch")
    if elapsed <= 0:
        reasons.append("no_elapsed_time")
    if elapsed > rules.timeout_s:
        reasons.append("timeout")
    grid = world.scene["grid"]
    aligned = True
    for name, dim in (("voxel_map", 3), ("floor_plan", 2)):
        claimed = report[name]
        if (claimed["shape"] != grid["shape"][:dim]
                or claimed["origin"] != grid["origin"][:dim]
                or claimed["resolution"] != grid["resolution"]):
            aligned = False
            reasons.append(f"{name}_alignment")
    observed = dict(episode.observations)
    claimed = {tuple(cell[:3]): cell[3] for cell in report["voxel_map"]["cells"]} if aligned else {}
    correct = sum(claimed.get(cell) == value and observed.get(cell) == value
                  for cell, value in world.observable.items())
    coverage = correct / len(world.observable) if world.observable else 0.0
    unsupported = sum(observed.get(cell) != value for cell, value in claimed.items())
    if unsupported:
        reasons.append("unsupported_map_claims")
    if coverage < rules.coverage_threshold:
        reasons.append("coverage")
    truth_floor = make_report(world, world.observable, elapsed)["floor_plan"]["cells"]
    floor_cells = report["floor_plan"]["cells"] if aligned else [0] * len(truth_floor)
    iou = sum(_iou({i for i, value in enumerate(truth_floor) if value == label},
                   {i for i, value in enumerate(floor_cells) if value == label})
              for label in (1, 2)) / 2
    if iou < rules.floor_plan_iou_threshold:
        reasons.append("floor_plan_iou")
    # Floor-plan cells must also have trusted support, independent of the
    # voxel_map (a forged floor plan cannot get a free pass).
    nx, ny = grid["shape"][:2]
    height = world.index(world.spawn)[2]
    if aligned and any(value and observed.get((i % nx, i // nx, height)) != value
                       for i, value in enumerate(floor_cells)):
        reasons.append("unsupported_floor_plan_claims")
    reported_rooms = {room["room_id"]: room for room in report["rooms"]}
    # Canonical scene ids are not exposed to a mapper. Match proposed room
    # geometry one-to-one; arbitrary policy-generated room ids are supported.
    candidates = {}
    for room in world.scene["rooms"]:
        rid = room["room_id"]
        if rid not in world.reachable_rooms:
            continue
        footprint = {(x, y) for x in range(nx) for y in range(ny)
                     if point_in_polygon(world.center((x, y, height))[:2], room["polygon"])}
        evidenced = any(value == 1 and (cell[0], cell[1]) in footprint
                         for cell, value in observed.items())
        candidates[rid] = []
        for proposed_id, proposed_room in sorted(reported_rooms.items()):
            proposed = {(x, y) for x in range(nx) for y in range(ny)
                        if point_in_polygon(world.center((x, y, height))[:2], proposed_room["polygon"])}
            if evidenced and _iou(footprint, proposed) >= rules.room_polygon_iou_threshold:
                candidates[rid].append(proposed_id)
    matches = {}

    def match(room_id, visited):
        for proposed_id in candidates[room_id]:
            if proposed_id in visited:
                continue
            visited.add(proposed_id)
            if proposed_id not in matches or match(matches[proposed_id], visited):
                matches[proposed_id] = room_id
                return True
        return False

    for room_id in sorted(candidates):
        match(room_id, set())
    found = sorted(matches.values())
    if set(matches) != set(reported_rooms):
        reasons.append("unmatched_reported_rooms")
    if set(found) != set(world.reachable_rooms):
        reasons.append("rooms_missing")
    if episode.collisions:
        reasons.append("collision")
    if episode.left_bounds:
        reasons.append("left_map")
    if not episode.landed_safely:
        reasons.append("unsafe_landing")
    return {
        "schema_version": "ares.episode_result.v1", "passed": not reasons,
        "time_s": elapsed, "coverage": coverage, "floor_plan_iou": iou,
        "rooms_found": sorted(found), "rooms_required": list(world.reachable_rooms),
        "collisions": episode.collisions, "minimum_clearance_m": episode.minimum_clearance_m,
        "landed_safely": episode.landed_safely, "unsupported_map_cells": unsupported,
        "fail_reasons": reasons, "scorer_version": SCORER_VERSION,
        "scene_version": world.scene["schema_version"], "scene_id": world.scene["scene_id"],
        "profile_version": world.profile["schema_version"], "profile_id": world.profile["profile_id"],
        "seed": episode.seed, "rules": asdict(rules), "scene_profile_sha256": world.digest,
    }
