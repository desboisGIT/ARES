"""Frozen-checkpoint CLI audit without executing training or flight rollouts."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
try:
    import torch
except ImportError:
    raise unittest.SkipTest('optional Phase 4 Torch unavailable')

from ares import cli, phase4
from ares.flight_policy import FlightPolicy
from ares.flight_training import save_checkpoint
from ares.phase2 import default_profile


class FlightCLITests(unittest.TestCase):
    def make_checkpoint(self, directory, metadata):
        destination = Path(directory) / 'frozen.pt'
        save_checkpoint(destination, FlightPolicy(), algorithm='bptt', seed=77,
                        training_steps=0, training_curve=[], metadata=metadata)
        return destination

    def invoke(self, checkpoint, output, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = cli.main(['evaluate-flight', str(checkpoint), '--output', str(output),
                               '--device', 'cpu', *extra])
        return result, stdout.getvalue(), stderr.getvalue()

    def test_frozen_profile_forwarded_weights_only_and_checkpoint_unchanged(self):
        profile = default_profile()
        profile['profile_id'] = 'frozen_training_profile_test'
        source = Path(phase4.__file__).parent / 'flight_policy.py'
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(directory, {'profile': profile,
                'runtime_source_sha256': {'flight_policy.py': source_digest}})
            before = checkpoint.read_bytes()
            output = Path(directory) / 'evaluation.json'
            with patch('ares.phase4.evaluate_flight_policy', return_value={'summary': {'runs': 3}}) as evaluate, \
                 patch('ares.flight_training.train_policy') as train, \
                 patch('torch.load', wraps=torch.load) as load:
                code, stdout, stderr = self.invoke(checkpoint, output, '--episodes', '3', '--batch-size', '2',
                    '--max-steps', '5', '--start-seed', '60001', '--split', 'validation', '--shield')
            self.assertEqual(code, 0, stderr)
            self.assertEqual(json.loads(stdout), {'runs': 3})
            train.assert_not_called()
            self.assertTrue(load.call_args.kwargs['weights_only'])
            policy, forwarded = evaluate.call_args.args
            self.assertFalse(policy.training)
            self.assertEqual(forwarded, profile)
            self.assertEqual({key: value for key, value in evaluate.call_args.kwargs.items() if key != 'progress'},
                {'device':'cpu','episodes':3,'batch_size':2,'max_steps':5,'start_seed':60001,
                 'split':'validation','shield':True})
            saved = json.loads(output.read_text())
            self.assertEqual(saved['schema_version'], 'ares.flight_evaluation.v1')
            self.assertEqual(saved['checkpoint_sha256'], hashlib.sha256(before).hexdigest())
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_runtime_mismatch_rejected_before_evaluation(self):
        self.reject({'profile': default_profile(), 'runtime_source_sha256': {'flight_policy.py':'0'*64}},
                    'runtime differs')

    def test_source_path_traversal_rejected_before_evaluation(self):
        for source in ('../README.md', '/etc/passwd', 'vendor/../../README.md'):
            with self.subTest(source=source):
                self.reject({'profile': default_profile(), 'runtime_source_sha256': {source:'0'*64}},
                            'invalid checkpoint source identifier')

    def test_missing_frozen_profile_reports_clean_error(self):
        self.reject({'runtime_source_sha256': {}}, 'no training profile snapshot')

    def reject(self, metadata, expected_error):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.make_checkpoint(directory, metadata)
            before = checkpoint.read_bytes()
            output = Path(directory) / 'must-not-exist.json'
            with patch('ares.phase4.evaluate_flight_policy') as evaluate, \
                 patch('ares.flight_training.train_policy') as train:
                code, stdout, stderr = self.invoke(checkpoint, output)
            self.assertEqual(code, 2)
            self.assertIn('ares: ', stderr)
            self.assertIn(expected_error, stderr)
            self.assertNotIn('Traceback', stderr)
            self.assertEqual(stdout, '')
            evaluate.assert_not_called()
            train.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(checkpoint.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
