"""Disk-backed conversion tests; no RealSense SDK or camera required."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

_spec = importlib.util.spec_from_file_location('capture_realsense',
    Path(__file__).resolve().parents[1] / 'scripts' / 'capture_realsense.py')
capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture)


@unittest.skipIf(np is None, 'optional NumPy simulation extra is not installed')
class SpoolTests(unittest.TestCase):
    def test_conversion_is_disk_backed_bounded_and_ignores_unused_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path, output = Path(directory)/'raw.u16', Path(directory)/'normalized.npy'
            shape = (19, 3, 4)
            raw = np.memmap(raw_path, dtype=np.uint16, mode='w+', shape=shape)
            raw[:] = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
            expected = np.array(raw[:17], dtype=np.float32) * .001
            expected[0, 0, 0] = np.nan
            raw.flush()
            raw._mmap.close()
            writes = []
            class TrackingMap(np.memmap):
                def __setitem__(self, key, value):
                    writes.append(value.shape[0])
                    return super().__setitem__(key, value)
            open_map = np.lib.format.open_memmap
            def tracked(*args, **kwargs):
                return open_map(*args, **kwargs).view(TrackingMap)
            with patch.object(np.lib.format, 'open_memmap', side_effect=tracked):
                result = capture.normalize_spool(raw_path, shape, 17, .001, output)
            self.assertIsInstance(result, np.memmap)
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(result.shape, (17, 3, 4))
            np.testing.assert_allclose(result, expected, equal_nan=True)
            self.assertEqual(sum(writes), 17)
            self.assertLessEqual(max(writes), 8)
            result._mmap.close()
            reread = np.load(output, mmap_mode='r')
            np.testing.assert_allclose(reread, expected, equal_nan=True)
            reread._mmap.close()
            self.assertTrue(raw_path.exists())  # converter never deletes source

    def test_rejects_invalid_range_shape_scale_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, output = Path(directory)/'raw', Path(directory)/'out.npy'
            raw.write_bytes(b'\0' * 16)
            for shape, count, scale in [((2, 2), 1, .001), ((2, 2, 2), 0, .001),
                    ((2, 2, 2), 3, .001), ((2, 0, 2), 1, .001),
                    ((2, 2, 2), 1, 0), ((2, 2, 2), 1, float('nan')), ((2, 2, 2), 1, True),
                    ((3, 2, 2), 1, .001)]:
                with self.assertRaises(ValueError):
                    capture.normalize_spool(raw, shape, count, scale, output)
                self.assertFalse(output.exists())
            output.write_bytes(b'preserve')
            with self.assertRaises(ValueError):
                capture.normalize_spool(raw, (2, 2, 2), 1, .001, output)
            self.assertEqual(output.read_bytes(), b'preserve')
            linked = Path(directory) / 'dangling.npy'
            target = Path(directory) / 'outside.npy'
            linked.symlink_to(target)
            with self.assertRaises(ValueError):
                capture.normalize_spool(raw, (2, 2, 2), 1, .001, linked)
            self.assertFalse(target.exists())


class CaptureHelpTests(unittest.TestCase):
    def test_help_without_sdk(self):
        result = subprocess.run([sys.executable, str(Path(capture.__file__)), '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('--bag-reserve-gib', result.stdout)
        self.assertIn('maximum 20000', result.stdout)
        self.assertIn('maximum 600', result.stdout)

    def test_default_duration_and_cap_without_sdk(self):
        with patch.object(sys, 'argv', ['capture', 'fresh-output', '--recording-id', 'id',
                                      '--profile-id', 'profile']), patch.object(capture, 'capture') as run:
            capture.main()
        args = run.call_args.args[0]
        self.assertEqual(args.duration, 60)
        self.assertEqual(args.max_frames, 2000)
        self.assertEqual(args.bag_reserve_gib, 4)


class ClockWarmupTests(unittest.TestCase):
    def test_discards_until_all_configured_streams_are_global(self):
        gate = capture.ClockWarmupGate(['depth:0', 'infrared:1', 'infrared:2', 'accel:0', 'gyro:0'])
        self.assertFalse(gate.observe('depth:0', False, 'hardware_clock'))
        self.assertFalse(gate.observe('depth:0', True, 'global_time'))
        self.assertFalse(gate.observe('depth:0', True, 'global_time'))
        self.assertFalse(gate.ready)
        for key in ('infrared:1', 'infrared:2', 'accel:0', 'gyro:0'):
            self.assertFalse(gate.observe(key, True, 'global_time'))
        self.assertTrue(gate.ready)
        gate.require_ready()
        self.assertTrue(gate.observe('depth:0', True, 'global_time'))
        self.assertEqual(gate.discard_counts['depth:0'], 3)
        self.assertEqual(gate.observed_domains['depth:0'], {'hardware_clock', 'global_time'})

    def test_rejects_post_ready_clock_fallback(self):
        gate = capture.ClockWarmupGate(['depth:0'])
        self.assertFalse(gate.observe('depth:0', True, 'global_time'))
        with self.assertRaisesRegex(RuntimeError, 'after clock readiness'):
            gate.observe('depth:0', False, 'hardware_clock')

    def test_missing_stream_and_pre_ready_fallback_never_pass(self):
        gate = capture.ClockWarmupGate(['depth:0', 'gyro:0'])
        gate.observe('depth:0', True, 'global_time')
        gate.observe('depth:0', False, 'hardware_clock')
        gate.observe('gyro:0', True, 'global_time')
        self.assertFalse(gate.ready)
        with self.assertRaisesRegex(RuntimeError, 'missing streams: depth:0'):
            gate.require_ready()
        with self.assertRaisesRegex(RuntimeError, 'Unexpected capture stream'):
            gate.observe('unknown:0', True, 'global_time')
