"""Small CPU/mocked harness checks; no generated held-out layouts."""
from dataclasses import asdict
import hashlib
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import scipy
except ImportError:
    raise unittest.SkipTest("Phase 3 requires planning extra")

from ares.fast_world import FastWorldConfig, GroundTruthFastWorld, FAST_WORLD_VERSION
from ares.exploration import extract_rooms
from ares.exploration_scoring import build_map_report, score_exploration, EXPLORATION_SCORER_VERSION
from ares.human_benchmark import summarize_human_logs
from ares.phase3 import benchmark_fast_world, policy_digest, run_phase3, summarize_results
from tests.test_fast_world import analytic_scene, nominal_profile

ROOT = Path(__file__).resolve().parents[1]


class Phase3PipelineTests(unittest.TestCase):
    def test_summary_times_only_passing_and_failures_in_denominator(self):
        result = summarize_results([
            {"passed": True, "time_s": 10., "collisions": 0},
            {"passed": False, "time_s": 1000., "collisions": 2},
            {"passed": True, "time_s": 30., "collisions": 0},
        ])
        self.assertEqual(result["runs"], 3)
        self.assertEqual(result["passed_runs"], 2)
        self.assertAlmostEqual(result["pass_rate"], 2 / 3)
        self.assertEqual(result["median_time_s"], 20.)
        self.assertEqual(result["p90_time_s"], 28.)
        self.assertEqual(result["collision_segments"], 2)
        low, high = result["pass_rate_wilson_95_interval"]
        self.assertLess(low, 2 / 3)
        self.assertGreater(high, 2 / 3)
        self.assertEqual(summarize_results([])["median_time_s"], None)
        failed = summarize_results([{"passed": False, "time_s": 1., "collisions": 1}])
        self.assertEqual(failed["pass_rate"], 0.)
        self.assertIsNone(failed["p90_time_s"])

    def test_benchmark_counts_actual_active_steps_excluding_warmup_and_done(self):
        scene, profile = analytic_scene(), nominal_profile()
        def construct(*args, **kwargs):
            kwargs["config"] = FastWorldConfig(max_time_s=.35, camera_width=4, camera_height=3)
            return GroundTruthFastWorld(*args, **kwargs)
        with patch("ares.phase3.GroundTruthFastWorld", side_effect=construct):
            result = benchmark_fast_world(scene, profile, backend="cpu", num_envs=2, steps=4)
        # Five warmup steps are excluded. The timeout leaves just two actual
        # measured steps per environment, rather than the four requested calls.
        self.assertEqual(result["individual_environment_steps"], 4)
        self.assertEqual(result["done_environments"], 2)
        self.assertFalse(result["all_environments_active"])
        self.assertFalse(result["passed"])
        self.assertGreater(result["wall_seconds"], 0.)
        self.assertAlmostEqual(result["environment_steps_per_second"], 4 / result["wall_seconds"])

    def test_benchmark_small_cpu_active_population_never_claims_gpu_gate(self):
        def construct(*args, **kwargs):
            kwargs["config"] = FastWorldConfig(camera_width=4, camera_height=3)
            return GroundTruthFastWorld(*args, **kwargs)
        with patch("ares.phase3.GroundTruthFastWorld", side_effect=construct):
            result = benchmark_fast_world(analytic_scene(), nominal_profile(), backend="cpu", num_envs=2, steps=2)
        self.assertEqual(result["individual_environment_steps"], 4)
        self.assertTrue(result["all_environments_active"])
        self.assertFalse(result["passed"])

    def test_main_module_import_is_safe_in_spawn_child(self):
        source = textwrap.dedent('''
            import multiprocessing
            import ares.__main__
            def child(queue):
                import ares.__main__
                queue.put("imported")
            if __name__ == "__main__":
                context = multiprocessing.get_context("spawn")
                queue = context.Queue()
                process = context.Process(target=child, args=(queue,))
                process.start()
                process.join(15)
                if process.is_alive():
                    process.terminate()
                    raise RuntimeError("spawn child hung")
                assert process.exitcode == 0, process.exitcode
                assert queue.get(timeout=2) == "imported"
                print("spawn import passed")
        ''')
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "spawn_import_probe.py"
            script.write_text(source)
            process = subprocess.run([sys.executable, str(script)], cwd=ROOT,
                                     env=dict(os.environ, PYTHONPATH=str(ROOT)),
                                     capture_output=True, text=True, timeout=25)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("spawn import passed", process.stdout)

    def test_policy_digest_is_source_hash_and_run_rejects_changed_digest(self):
        expected = hashlib.sha256((ROOT / "ares/exploration.py").read_bytes()).hexdigest()
        self.assertEqual(policy_digest(), expected)
        def worker(arguments):
            seed, split, *_ = arguments
            return {"seed": seed, "scene_sha256": "analytic-fixture", "policy_sha256": expected,
                    "passed": True, "time_s": 2., "collisions": 0}
        with tempfile.TemporaryDirectory() as temporary, \
                patch("ares.phase3.generate_scene", return_value=analytic_scene()), \
                patch("ares.phase3.benchmark_fast_world", return_value={"passed": False}), \
                patch("ares.phase3._episode_worker", side_effect=worker):
            output = run_phase3(output_dir=temporary, episodes=2, start_seed=100, split="train")
            self.assertEqual(output["policy_sha256"], expected)
            self.assertEqual(output["dataset"]["seeds"], [100, 101])
            self.assertFalse(output["classical_apartment_gate"]["passed"])
            self.assertFalse(output["phase_passed"])
            stored = json.loads((Path(temporary) / "baseline_results.json").read_text())
            self.assertEqual(stored["classical"]["pass_rate"], 1.)
            with patch("ares.phase3.policy_digest", side_effect=[expected, "changed"]):
                with self.assertRaisesRegex(ValueError, "policy changed"):
                    run_phase3(output_dir=temporary, episodes=1)


    def test_human_cli_filters_cohort_and_keeps_phase_pending_without_valid_runs(self):
        from ares.cli import main
        from ares.phase3 import _hash
        dataset = {"split": "train", "seeds": [0], "scene_sha256": [_hash(analytic_scene())]}
        baseline = {"schema_version": "ares.phase3.v1", "world_version": FAST_WORLD_VERSION,
                    "scorer_version": EXPLORATION_SCORER_VERSION, "policy_sha256": policy_digest(),
                    "dataset": dataset, "dataset_sha256": _hash(dataset), "profile": nominal_profile(),
                    "world_config": asdict(FastWorldConfig()), "benchmark": {"passed": True},
                    "classical_apartment_gate": {"passed": True}, "phase_passed": False}
        verified = {"summary": summarize_results([]), "records": [],
                    "rejected": [{"path": "ui-test.json", "reason": "actor is ui_test"}],
                    "trust_note": "Test fixture; no human measurement.", "metric_comparison_tolerances": {}}
        with tempfile.TemporaryDirectory() as temporary:
            source, destination = Path(temporary) / "baseline.json", Path(temporary) / "combined.json"
            source.write_text(json.dumps(baseline))
            original = source.read_bytes()
            with patch("ares.human_benchmark.summarize_human_logs", return_value=verified) as aggregate, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(["human-benchmark", "ui-test.json", "--baseline", str(source),
                               "--output", str(destination)])
            self.assertEqual(status, 1)
            aggregate.assert_called_once_with([Path("ui-test.json")],
                expected_scene_sha256=dataset["scene_sha256"], expected_profile=baseline["profile"],
                expected_world_config=baseline["world_config"])
            combined = json.loads(destination.read_text())
            self.assertFalse(combined["phase_passed"])
            self.assertEqual(combined["human"]["status"], "pending_actual_human_runs")
            self.assertEqual(combined["human"]["runs"], 0)
            self.assertIsNone(combined["human"]["median_time_s"])
            self.assertEqual(source.read_bytes(), original)
            baseline["policy_sha256"] = "different-policy"
            source.write_text(json.dumps(baseline))
            with patch("ares.human_benchmark.summarize_human_logs") as aggregate, \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                status = main(["human-benchmark", "ui-test.json", "--baseline", str(source),
                               "--output", str(destination)])
            self.assertEqual(status, 2)
            aggregate.assert_not_called()


class HumanBenchmarkTests(unittest.TestCase):
    """Synthetic attestation fixtures stay in temporary test directories only."""

    def make_log(self):
        scene = analytic_scene([(18, 16, 7)])
        profile = nominal_profile()
        config = FastWorldConfig(dt=.5, camera_width=4, camera_height=3)
        world = GroundTruthFastWorld(scene, profile, config=config, seed=scene["seed"])
        action = [2., 0., 0., 0., 0.]
        world.step(np.array([action]))
        observation = world.observation()
        report = build_map_report(observation, extract_rooms(observation["known_grid"],
                                                            world.resolution, world.origin))
        result = score_exploration(world, world.snapshot_for_scoring(), report)
        self.assertTrue(world.ground_truth_state()["done"])
        self.assertEqual(result["collisions"], 1)
        canonical = json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {"schema_version": "ares.exploration_log.v1", "actor": "human", "started": True,
                "completed": True, "error": None, "scene": scene, "profile": profile,
                "seed": scene["seed"], "scene_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                "world_version": FAST_WORLD_VERSION, "scorer_version": EXPLORATION_SCORER_VERSION,
                "world_config": asdict(config), "backend": "cpu", "actions": [action],
                "report": report, "result": result, "wall_duration_s": .5,
                "sim_wall_ratio": 1., "realtime_eligible": True}

    def write_log(self, directory, name, log):
        path = Path(directory) / name
        payload = json.dumps(log).encode()
        path.write_bytes(gzip.compress(payload) if path.suffix == ".gz" else payload)
        return path

    def test_completed_human_attested_collision_counts_as_failure(self):
        log = self.make_log()
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_log(temporary, "synthetic-attestation.json", log)
            result = summarize_human_logs([path], expected_scene_sha256={log["scene_sha256"]},
                                          expected_profile=log["profile"],
                                          expected_world_config=log["world_config"])
        self.assertEqual(result["summary"]["runs"], 1)
        self.assertEqual(result["summary"]["pass_rate"], 0.)
        self.assertEqual(result["summary"]["collision_segments"], 1)
        self.assertIsNone(result["summary"]["median_time_s"])
        self.assertEqual(result["rejected"], [])
        self.assertIn("unsigned operator attestations", result["trust_note"])

    def test_ui_partial_time_scaled_and_forged_results_rejected(self):
        base = self.make_log()
        variants = []
        for key, value in (("actor", "ui_test"), ("completed", False),
                           ("realtime_eligible", False), ("wall_duration_s", .1),
                           ("scene_sha256", "forged"), ("world_version", "old")):
            log = copy.deepcopy(base)
            log[key] = value
            variants.append(log)
        log = copy.deepcopy(base)
        log["result"]["passed"] = True
        variants.append(log)
        with tempfile.TemporaryDirectory() as temporary:
            paths = [self.write_log(temporary, f"invalid-{i}.json", log) for i, log in enumerate(variants)]
            result = summarize_human_logs(paths)
        self.assertEqual(result["summary"]["runs"], 0)
        self.assertEqual(len(result["rejected"]), len(variants))
        self.assertTrue(all(item["reason"] for item in result["rejected"]))

    def test_duplicate_trace_rejected_and_missing_declared_hash_computed(self):
        log = self.make_log()
        expected = log.pop("scene_sha256")
        with tempfile.TemporaryDirectory() as temporary:
            first = self.write_log(temporary, "attempt.json", log)
            copy_path = self.write_log(temporary, "same-attempt.json.gz", log)
            result = summarize_human_logs([first, copy_path])
        self.assertEqual(result["summary"]["runs"], 1)
        self.assertEqual(result["records"][0]["scene_sha256"], expected)
        self.assertEqual(len(result["rejected"]), 1)
        self.assertIn("duplicate", result["rejected"][0]["reason"])

    def test_wrong_baseline_conditions_and_forged_completion_rejected(self):
        log = self.make_log()
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_log(temporary, "attempt.json", log)
            for kwargs in ({"expected_scene_sha256": {"another-scene"}},
                           {"expected_profile": {}}, {"expected_world_config": {}}):
                result = summarize_human_logs([path], **kwargs)
                self.assertEqual(result["summary"]["runs"], 0)
                self.assertEqual(len(result["rejected"]), 1)
            log["actions"] = [[0., 0., 0., 0., 0.]]
            path = self.write_log(temporary, "incomplete-replay.json", log)
            result = summarize_human_logs([path])
            self.assertEqual(result["summary"]["runs"], 0)
            self.assertIn("incomplete", result["rejected"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
