"""Portable SIL archive integrity and command protocol; no flight dependency."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from ares.verified_safety import _launcher as launch


def archive(entries):
    stream = io.BytesIO()
    manifest = {}
    with zipfile.ZipFile(stream, 'w') as result:
        for name, payload, mode in entries:
            info = zipfile.ZipInfo(name)
            info.external_attr = mode << 16
            result.writestr(info, payload)
            manifest[name] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    data = stream.getvalue()
    return data, manifest, hashlib.sha256(data).hexdigest()


class VerifiedSafetyPackagingTests(unittest.TestCase):
    def test_complete_shipped_archive_is_checked_without_extraction(self):
        members = launch.load_members()
        self.assertIn(launch.RUNNER, members)
        self.assertIn('reports/phase_4_shield_artifacts/cpu_rpi_certificate_v1/certificate.json', members)
        self.assertTrue(all(name.startswith('reports/') for name in members))
        self.assertFalse(any('__pycache__' in name or name.endswith('.pt') for name in members))

    def test_changed_archive_manifest_or_payload_rejects(self):
        data, manifest, checksum = archive([('reports/example.txt', b'proof', stat.S_IFREG | 0o644)])
        self.assertEqual(launch.checked_members(data, manifest, checksum), {'reports/example.txt': b'proof'})
        with self.assertRaises(ValueError):
            launch.checked_members(data, manifest, '0'*64)
        manifest['reports/example.txt']['sha256'] = '0'*64
        with self.assertRaises(ValueError):
            launch.checked_members(data, manifest, checksum)

    def test_traversal_absolute_symlink_and_duplicate_members_reject(self):
        for name, mode in (('../escape', stat.S_IFREG), ('/reports/escape', stat.S_IFREG),
                           ('reports/../escape', stat.S_IFREG), ('reports/link', stat.S_IFLNK),
                           ('reports/a\\b', stat.S_IFREG)):
            with self.subTest(name=name):
                args = archive([(name, b'x', mode | 0o644)])
                with self.assertRaises(ValueError):
                    launch.checked_members(*args)
        with contextlib.redirect_stderr(io.StringIO()):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                args = archive([('reports/same', b'x', stat.S_IFREG), ('reports/same', b'x', stat.S_IFREG)])
        with self.assertRaises(ValueError):
            launch.checked_members(*args)

    def test_missing_or_extra_member_rejects(self):
        data, manifest, checksum = archive([('reports/example.txt', b'proof', stat.S_IFREG)])
        with self.assertRaises(ValueError):
            launch.checked_members(data, {}, checksum)
        manifest['reports/missing.txt'] = {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()}
        with self.assertRaises(ValueError):
            launch.checked_members(data, manifest, checksum)

    def test_short_validation_protocol_and_non_cpu_device_reject(self):
        for args in (['validate', '--steps', '1'], ['validate', '--cases', '1'],
                     ['smoke', '--device', 'cuda']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                launch.main(args)
            self.assertEqual(error.exception.code, 2)

    def test_functional_smoke_never_relabels_a_timing_failure(self):
        report = {'source_hashes_stable': True, 'run_valid': True, 'cases': [
            {'case': i, 'requested_steps': 1000, 'completed_steps': 1000,
             'functional_behavior_passed': True, 'timing_gate_passed': False}
            for i in range(8)]}
        smoke = launch.report_status(report, mode='smoke', expected_cases=8, expected_steps=1000)
        self.assertTrue(smoke['functional_checks_passed'])
        self.assertTrue(smoke['requested_check_passed'])
        self.assertFalse(smoke['timing_checks_passed'])
        self.assertFalse(smoke['conditional_SIL_passed'])
        self.assertEqual(launch.status_exit_code(smoke), 0)
        strict = launch.report_status(report, mode='validate', expected_cases=8, expected_steps=1000)
        self.assertFalse(strict['requested_check_passed'])
        self.assertEqual(launch.status_exit_code(strict), 1)
        report['cases'][0]['functional_behavior_passed'] = False
        refused = launch.report_status(report, mode='smoke', expected_cases=8, expected_steps=1000)
        self.assertFalse(refused['requested_check_passed'])
        self.assertEqual(launch.status_exit_code(refused), 1)

    def test_existing_outputs_logs_and_broken_symlinks_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            path.write_text('existing result')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launch.main(['smoke', '--output', str(path)])
            self.assertEqual(path.read_text(), 'existing result')
            path.unlink()
            path.with_name(path.name+'.log').write_text('existing log')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launch.main(['smoke', '--output', str(path)])
            self.assertEqual(path.with_name(path.name+'.log').read_text(), 'existing log')
            path.with_name(path.name+'.log').unlink()
            path.symlink_to(Path(directory)/'missing')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launch.main(['smoke', '--output', str(path)])
            self.assertTrue(path.is_symlink())


if __name__ == '__main__':
    unittest.main()
