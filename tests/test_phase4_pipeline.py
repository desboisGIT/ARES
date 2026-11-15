"""Cheap audit tests for paired evaluation, frozen artifacts and honest gates."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import scipy
    import torch
except ImportError:
    raise unittest.SkipTest('optional Phase 4 numpy/scipy/torch unavailable')

from ares import phase4
from ares.flight_policy import FlightPolicy


class Phase4PipelineTests(unittest.TestCase):
    @staticmethod
    def record(success, time_s, collision=False):
        return {'success': success, 'time_s': time_s, 'collision': collision,
                'shield_interventions': 0, 'certified_commands': 0, 'shield_checks': 0,
                'max_speed_mps': 1.2, 'minimum_clearance_m': .1}

    def test_failures_stay_in_denominator_and_out_of_success_time_population(self):
        summary = phase4.summarize_flights([
            self.record(True, 2), self.record(False, 100, True), self.record(True, 4)])
        self.assertEqual(summary['runs'], 3)
        self.assertEqual(summary['successes'], 2)
        self.assertAlmostEqual(summary['success_rate'], 2 / 3)
        self.assertEqual(summary['median_time_s'], 3)
        self.assertAlmostEqual(summary['p90_time_s'], 3.8)
        self.assertEqual(summary['collisions'], 1)
        failed = phase4.summarize_flights([self.record(False, 4)])
        self.assertIsNone(failed['median_time_s'])
        self.assertIsNone(failed['p90_time_s'])
        self.assertIsNone(phase4.summarize_flights([])['success_rate'])

    def test_evaluation_cohort_digest_is_paired_and_covers_endpoint_batches(self):
        class Policy:
            def to(self, device): return self
            def eval(self): return self
            def __call__(self, observation): return torch.zeros(len(observation['features']), 4)

        class Environment:
            def __init__(self, scene, profile, *, batch_size, device, seed, horizon, randomization, dt, guidance_speed_fraction=1.):
                self.seed, self.dt = seed, dt
                self.speed_limit = 1.5
                self.speed_limit_steps = torch.zeros(batch_size, dtype=torch.long)
                self.tasks = [{'positions': [[scene['seed'], 0, 1], [i, 1, 1], [i, 2, 1]]}
                              for i in range(batch_size)]
                self.sampled_profiles = [{'mode': randomization, 'index': i} for i in range(batch_size)]
                self.done = torch.zeros(batch_size, dtype=torch.bool)
                self.success = torch.arange(batch_size) % 2 == 0
                self.collided = ~self.success
                self.episode_steps = torch.ones(batch_size, dtype=torch.long)
                self.maximum_speed = torch.ones(batch_size)
                self.minimum_clearance = torch.full((batch_size,), .2)
                self.minimum_clearance_is_lower_bound = torch.zeros(batch_size, dtype=torch.bool)
            def observation(self): return {'features': torch.zeros(len(self.done), 43)}
            def step(self, action): self.done[:] = True

        with patch.object(phase4, 'FlightEnv', Environment), \
             patch.object(phase4, 'generate_scene', side_effect=lambda seed, kind, split: {'seed': seed, 'split': split}), \
             patch.object(phase4, 'randomization_manifest', return_value={'audited': True}):
            first = phase4.evaluate_flight_policy(Policy(), {}, episodes=5, batch_size=2, max_steps=2)
            second = phase4.evaluate_flight_policy(Policy(), {}, episodes=5, batch_size=2, max_steps=2)
            changed = phase4.evaluate_flight_policy(Policy(), {}, episodes=5, batch_size=2, max_steps=2,
                                                   start_seed=80000)
        self.assertEqual(first['dataset_sha256'], second['dataset_sha256'])
        self.assertNotEqual(first['dataset_sha256'], changed['dataset_sha256'])
        self.assertEqual(first['summary']['runs'], 5)
        self.assertEqual([r['profile_mode'] for r in first['records']], ['low', 'low', 'high', 'high', 'random'])
        self.assertEqual(len(set(first['scene_sha256'])), 3)
        self.assertFalse(first['shield_assumptions_validated'])
        self.assertEqual(first['policy_hz'], 60)
        self.assertIsNone(first['shield_hz'])
        self.assertTrue(all(record['speed_limit_mps'] == 1.5 for record in first['records']))
        self.assertTrue(all(record['speed_limit_exceeding_steps'] == 0 for record in first['records']))
        self.assertTrue(all(abs(record['time_s'] - 1 / 60) < 1e-9 for record in first['records']))

    def mock_run(self, directory, *, rate=1., ppo_rate=None, episodes=1000, split='test', mismatch=False, mutate=False, mutate_runtime=False, residual_acceleration=None, residual_penalty=0., guidance_speed_fraction=1.):
        events, trained, evaluated = [], [], []
        original_save, original_load = phase4.save_checkpoint, phase4.load_checkpoint
        saved_paths = []
        original_sources=phase4.runtime_source_hashes

        def sources():
            hashes=original_sources()
            if mutate_runtime and evaluated:
                hashes['phase4.py']='0'*64
            return hashes

        def train(env, algorithm, config, progress, policy=None):
            events.append('train-' + algorithm)
            policy = policy if policy is not None else FlightPolicy()
            with torch.no_grad(): policy.actor.bias.fill_(len(trained) + 1)
            trained.append(policy)
            return policy, {'training_steps': 4, 'optimizer_updates': 1, 'config': vars(config),
                            'training_curve': [{'loss': 1., 'gradient_norm': .2, 'cnn_gradient_norm': .01}]}

        def save(path, policy, **kwargs):
            events.append('save-' + kwargs['algorithm'])
            saved_paths.append(Path(path))
            return original_save(path, policy, **kwargs)

        def load(path, device):
            events.append('load-' + Path(path).stem)
            return original_load(path, device)

        def evaluate(policy, profile, **kwargs):
            events.append('evaluate')
            evaluated.append(policy)
            if mutate:
                saved_paths[-1 if len(evaluated) > 1 else 0].write_bytes(b'modified after freeze')
            evaluation_rate = ppo_rate if len(evaluated) == 2 and ppo_rate is not None else rate
            return {'summary': {'success_rate': evaluation_rate, 'runs': kwargs['episodes']},
                    'dataset_sha256': str(len(evaluated)) if mismatch else 'paired-fixed-cohort'}

        with ExitStack() as stack:
            stack.enter_context(patch.object(phase4,'runtime_source_hashes',side_effect=sources))
            stack.enter_context(patch.object(phase4, 'CurriculumFlightEnv',
                                             side_effect=lambda *args, **kwargs: SimpleNamespace(scenes=[{}, {}])))
            stack.enter_context(patch.object(phase4, 'train_policy', side_effect=train))
            stack.enter_context(patch.object(phase4, 'save_checkpoint', side_effect=save))
            stack.enter_context(patch.object(phase4, 'load_checkpoint', side_effect=load))
            stack.enter_context(patch.object(phase4, 'evaluate_flight_policy', side_effect=evaluate))
            stack.enter_context(patch.object(phase4, 'FlightEnv', return_value=SimpleNamespace(device=torch.device('cpu'))))
            stack.enter_context(patch.object(phase4, 'generate_scene', return_value={}))
            stack.enter_context(patch.object(phase4, 'inference_benchmark',
                                             return_value={'onboard_gate_passed': False}))
            report = phase4.run_phase4(output_dir=directory, episodes=episodes, eval_split=split,
                                       shield_episodes=0, residual_acceleration=residual_acceleration,
                                       residual_penalty=residual_penalty,
                                       guidance_speed_fraction=guidance_speed_fraction)
        return report, events, trained, evaluated

    def test_guidance_fraction_preserved_for_both_policies_and_default(self):
        with tempfile.TemporaryDirectory() as directory:
            for fraction in (1., .95):
                report, _, trained, evaluated = self.mock_run(directory, guidance_speed_fraction=fraction)
                for name, initial, frozen in zip(('bptt','ppo'), trained, evaluated):
                    self.assertEqual(initial.config.guidance_speed_fraction, fraction)
                    self.assertEqual(frozen.config.guidance_speed_fraction, fraction)
                    self.assertEqual(report['policy_configuration'][name]['guidance_speed_fraction'], fraction)
                    self.assertEqual(initial.config.residual_acceleration, 2.)

    def test_invalid_guidance_fraction_rejected_before_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not-created'
            for fraction in (0., -.1, 1.01, float('inf'), float('nan')):
                with self.subTest(fraction=fraction), self.assertRaisesRegex(ValueError, 'guidance_speed_fraction'):
                    phase4.run_phase4(output_dir=output, guidance_speed_fraction=fraction)
                self.assertFalse(output.exists())

    def test_curriculum_forwards_guidance_fraction_to_environment(self):
        with patch.object(phase4, 'generate_scene', return_value={}), \
             patch.object(phase4, 'FlightEnv') as environment:
            env = phase4.CurriculumFlightEnv({}, device='cpu', batch_size=2, seed=1,
                                            horizon=10, guidance_speed_fraction=.95)
            env.reset()
            self.assertEqual(environment.call_args.kwargs['guidance_speed_fraction'], .95)

    def test_residual_penalty_reaches_both_learners_and_defaults_to_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            for penalty in (0., .15):
                report, _, _, _ = self.mock_run(directory, residual_penalty=penalty)
                for algorithm in ('bptt', 'ppo'):
                    self.assertEqual(report['training'][algorithm]['config']['residual_penalty'], penalty)
                    _, payload = phase4.load_checkpoint(report['checkpoints'][algorithm]['path'])
                    self.assertEqual(payload['metadata']['training_config']['residual_penalty'], penalty)

    def test_invalid_residual_penalty_rejected_before_artifact_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not-created'
            for value in (-.1, float('inf'), float('nan')):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                    phase4.run_phase4(output_dir=output, residual_penalty=value)
                self.assertFalse(output.exists())

    def test_residual_override_reaches_both_learners_and_frozen_config(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _, trained, evaluated = self.mock_run(directory, residual_acceleration=.2)
            for name, initial, frozen in zip(('bptt','ppo'), trained, evaluated):
                self.assertEqual(initial.config.residual_acceleration, .2)
                self.assertEqual(frozen.config.residual_acceleration, .2)
                self.assertEqual(report['policy_configuration'][name]['residual_acceleration'], .2)
            torch.testing.assert_close(trained[0].depth_encoder[0].weight,
                                       trained[1].depth_encoder[0].weight)

    def test_invalid_residual_rejected_before_artifact_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not-created'
            for value in (0., -.1, float('inf'), float('nan')):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite and positive'):
                    phase4.run_phase4(output_dir=output, residual_acceleration=value)
                self.assertFalse(output.exists())

    def test_cli_forwards_optional_residual_authority(self):
        from ares.cli import main
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(phase4, 'run_phase4', return_value={
                 'software_preparation_passed': True, 'phase_passed': False,
                 'waypoint_reliability_gate': {}, 'shield_gate': {},
                 'onboard_inference_gate': {}, 'evaluation': {}}) as run, \
             redirect_stdout(StringIO()):
            main(['phase4', '--residual-acceleration', '.2', '--residual-penalty', '.15', '--guidance-speed-fraction', '.95', '--smoke',
                  '--output', str(Path(directory) / 'report.json')])
            self.assertEqual(run.call_args.kwargs['residual_acceleration'], .2)
            self.assertEqual(run.call_args.kwargs['residual_penalty'], .15)
            self.assertEqual(run.call_args.kwargs['guidance_speed_fraction'], .95)

    def test_frozen_checkpoints_are_evaluation_source_and_training_precedes_heldout(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / 'unrelated.pt'
            sentinel.write_bytes(b'leave existing artifacts intact')
            report, events, trained, evaluated = self.mock_run(directory)
            self.assertEqual(events[:4], ['train-bptt', 'save-bptt', 'train-ppo', 'save-ppo'])
            self.assertTrue(all('train' not in event for event in events[events.index('evaluate'):]))
            for initial, frozen in zip(trained, evaluated):
                self.assertIsNot(initial, frozen)
                torch.testing.assert_close(initial.actor.bias, frozen.actor.bias)
                self.assertFalse(frozen.training)
            self.assertEqual(sentinel.read_bytes(), b'leave existing artifacts intact')
            self.assertTrue((Path(report['artifacts']) / 'result.json').is_file())
            for checkpoint in report['checkpoints'].values():
                _, payload = phase4.load_checkpoint(checkpoint['path'])
                source_hashes = payload['metadata']['runtime_source_sha256']
                for source in ('flight_policy.py', 'flight_env.py', 'flight_training.py',
                               'vendor/diffaero/quadrotor.py','phase4.py','depth.py','worldgen.py'):
                    self.assertIn(source, source_hashes)
                    self.assertEqual(len(source_hashes[source]), 64)

    def test_reliability_requires_test_population_and_hardware_gates_remain_pending(self):
        cases = [(1., 1000, 'test', True), (.989, 1000, 'test', False),
                 (1., 999, 'test', False), (1., 1000, 'validation', False)]
        with tempfile.TemporaryDirectory() as directory:
            for rate, episodes, split, expected in cases:
                with self.subTest(rate=rate, episodes=episodes, split=split):
                    report, _, _, _ = self.mock_run(directory, rate=rate, episodes=episodes, split=split)
                    self.assertEqual(report['waypoint_reliability_gate']['passed'], expected)
                    self.assertFalse(report['shield_gate']['passed'])
                    self.assertFalse(report['onboard_inference_gate']['passed'])
                    self.assertFalse(report['phase_passed'])

    def test_primary_bptt_reliability_does_not_require_comparison_ppo_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _, _, _ = self.mock_run(directory, rate=1., ppo_rate=.5)
        self.assertEqual(report['evaluation']['bptt']['summary']['success_rate'], 1.)
        self.assertEqual(report['evaluation']['ppo']['summary']['success_rate'], .5)
        self.assertTrue(report['waypoint_reliability_gate']['passed'])
        self.assertFalse(report['shield_gate']['passed'])
        self.assertFalse(report['onboard_inference_gate']['passed'])
        self.assertFalse(report['phase_passed'])

    def test_mismatched_comparison_cohorts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'cohorts differ'):
                self.mock_run(directory, mismatch=True)

    def test_checkpoint_mutation_during_evaluation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                self.mock_run(directory, mutate=True)

    def test_runtime_mutation_during_evaluation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'runtime changed'):
                self.mock_run(directory,mutate_runtime=True)


if __name__ == '__main__':
    unittest.main()
