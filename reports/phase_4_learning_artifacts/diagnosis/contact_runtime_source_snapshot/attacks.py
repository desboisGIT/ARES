"""Deliberately invalid policies used to regression-test scorer exploitation."""

from .evaluator import GroundTruthSample
from .reporting import make_report
from .scoring import ScoringRules, score_episode


def run_attacks(world, seed):
    rules = ScoringRules()
    spawn = world.spawn
    cases = {}
    instant = world.record([GroundTruthSample(0, spawn, True)], seed=seed, land=True)
    cases["report_instantly"] = (instant, make_report(world, dict(instant.observations), 0))
    hover = world.record([GroundTruthSample(0, spawn, True),
                          GroundTruthSample(rules.timeout_s + 1, spawn, True)], seed=seed, land=True)
    cases["hover_forever"] = (hover, make_report(world, dict(hover.observations), rules.timeout_s + 1))
    outside = tuple(world.origin[i] - 1 for i in range(3))
    left = world.record([GroundTruthSample(0, spawn, True), GroundTruthSample(1, outside, True)],
                        seed=seed, land=True)
    cases["leave_map"] = (left, make_report(world, dict(left.observations), 1))
    fake = world.record([GroundTruthSample(0, spawn, True), GroundTruthSample(1, spawn, True)],
                        seed=seed, land=True)
    # Deliberately grant this attack the exact answer: even omniscient reporting
    # cannot substitute for evaluator-owned sensing evidence.
    forged = make_report(world, world.observable, 1, world.scene["rooms"])
    cases["fake_map"] = (fake, forged)
    return {name: score_episode(world, episode, report, rules)
            for name, (episode, report) in cases.items()}
