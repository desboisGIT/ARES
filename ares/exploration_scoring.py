"""Versioned cone-episode scoring, with evaluator-owned target geometry."""

from collections import deque
from types import SimpleNamespace
import numpy as np
from scipy import ndimage

from .contracts import point_in_polygon
from .fast_world import FastEpisode, GroundTruthFastWorld, inflation_structure
from .scoring import ScoringRules, score_episode

EXPLORATION_SCORER_VERSION = "ares.exploration_scorer.v1"


class GroundTruthExplorationReference:
    """Fixed before scoring; no policy receives this target, labels or geometry.

    Target definition retains the six-axis observable voxel union from Phase 0,
    using fast exact configuration-center inflation instead of repeated collision
    queries. Episode evidence now comes only from the forward cone and initial
    physical hull. This intentionally does not change Phase 0 scorer semantics.
    """

    def __init__(self, world):
        self.scene, self.profile, self.digest = world.scene, world.profile, world.digest
        self.shape, self.origin, self.resolution = world.shape, world.origin, world.resolution
        self.spawn = world.home
        occupied = world.volume.astype(bool)
        safe = ~ndimage.binary_dilation(occupied, structure=inflation_structure(world.radius,world.resolution), border_value=1)
        labels,_ = ndimage.label(safe, structure=ndimage.generate_binary_structure(3,1))
        home = self.index(self.spawn)
        if labels[home] == 0:
            raise ValueError("spawn is not a safe configuration-space voxel center")
        self.reachable_mask = labels == labels[home]
        centers = np.argwhere(self.reachable_mask)
        target = self.reachable_mask.copy()
        for axis in range(3):
            for sign in (-1,1):
                rays = centers.copy()
                for _ in range(int(np.floor(world.max_range/world.resolution))):
                    rays[:,axis] += sign
                    rays = rays[np.all((rays>=0)&(rays<self.shape),axis=1)]
                    if not len(rays):
                        break
                    target[tuple(rays.T)] = True
                    rays = rays[~occupied[tuple(rays.T)]]
                    if not len(rays):
                        break
        indices = np.argwhere(target)
        self.observable = {tuple(int(v) for v in cell): int(1+world.volume[tuple(cell)]) for cell in indices}
        xy = np.argwhere(self.reachable_mask.any(axis=2))
        self.reachable_rooms = tuple(sorted(room["room_id"] for room in self.scene["rooms"]
            if any(point_in_polygon(self.center((*cell,home[2]))[:2],room["polygon"]) for cell in xy)))

    def index(self, position):
        return tuple(np.floor((np.asarray(position)-self.origin)/self.resolution).astype(int))

    def center(self, cell):
        return tuple(self.origin+(np.asarray(cell)+.5)*self.resolution)


def build_map_report(observation, rooms=()):
    """Reporter uses only the public observed map and home coordinate."""
    known = np.asarray(observation["known_grid"])
    if known.ndim != 3 or not np.isin(known,[0,1,2]).all():
        raise ValueError("known_grid must be an XYZ 0/1/2 observed map")
    origin = np.asarray(observation["origin"],dtype=float)
    resolution = float(observation["resolution"])
    home_z = int(np.floor((observation["home"][2]-origin[2])/resolution))
    cells = np.argwhere(known>0)
    return {"schema_version": "ares.report.v1",
        "voxel_map": {"resolution": resolution,"origin": origin.tolist(),"shape": list(known.shape),
                      "cells": [[int(v) for v in cell]+[int(known[tuple(cell)])] for cell in cells]},
        "floor_plan": {"resolution": resolution,"origin": origin[:2].tolist(),"shape": list(known.shape[:2]),
                       "cells": known[:,:,home_z].T.ravel().astype(int).tolist()},
        "rooms": list(rooms),"person_markers": [],"end_time_s": float(observation["time_s"])}


def score_exploration(world, episode, report, rules=None):
    if not isinstance(world,GroundTruthFastWorld) or not isinstance(episode,FastEpisode):
        raise ValueError("scoring requires evaluator-owned FastWorld and FastEpisode")
    if episode.world_digest != world.digest or episode.home != tuple(float(v) for v in world.home):
        raise ValueError("episode belongs to a different scene/profile/configuration")
    if not hasattr(world,"_scoring_reference"):
        world._scoring_reference = GroundTruthExplorationReference(world)
    trusted = SimpleNamespace(samples=(SimpleNamespace(timestamp_s=episode.time_s),),
        observations=episode.observations,collisions=episode.collisions,left_bounds=episode.left_bounds,
        minimum_clearance_m=episode.minimum_clearance_m,landed_safely=episode.landed_safely,
        seed=episode.seed,world_digest=episode.world_digest)
    # Reuse frozen metric arithmetic, not its Phase 0 acquisition/trace model.
    result = score_episode(world._scoring_reference,trusted,report,
                           rules or ScoringRules(timeout_s=world.config.max_time_s))
    result.update(schema_version="ares.exploration_result.v1",scorer_version=EXPLORATION_SCORER_VERSION,
                  world_version="ares.fast_world.v1",minimum_clearance_is_lower_bound=episode.minimum_clearance_is_lower_bound,
                  clearance_cap_m=world.pad-world.radius, sensor_model="ideal_forward_voxel_cone.v1")
    return result
