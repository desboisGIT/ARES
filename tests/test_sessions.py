"""Viewer session clock, audit and observed-only replay invariants."""
import gzip
import json
from pathlib import Path
import tempfile
import time
import unittest

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest('sessions require optional NumPy')

from ares.fast_world import FastWorldConfig
from ares.sessions import ManualSession, ReplaySession, _realtime_eligible
from ares.recording import write_recording
from tests.test_fast_world import analytic_scene, nominal_profile
from tests.test_recording import example_recording


class ManualSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = ManualSession(analytic_scene(), nominal_profile(), actor='ui_test', log_dir=self.temp.name)
        self.addCleanup(self.session.close)

    def wait_time(self, target, timeout=3):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            state = self.session.state()
            if state['frame']['time_s'] >= target:
                return state
            time.sleep(.01)
        self.fail(f'physics did not reach {target}: {self.session.state()}')

    def test_waiting_state_polling_and_input_never_advance_clock(self):
        self.session.set_input({'velocity_body': [1., 0., 0.], 'yaw_rate': 0., 'land': False})
        for _ in range(50):
            state = self.session.state()
        time.sleep(.08)
        self.assertEqual(self.session.state()['frame']['time_s'], 0.)
        self.assertFalse(state['started'])
        self.assertIsNone(state['result'])
        self.assertNotIn('scene', state)
        self.assertNotIn('coverage', state['frame'])

    def test_wallclock_runs_without_http_and_watchdog_brakes(self):
        self.session.start()
        self.session.set_input({'velocity_body': [1., 0., 0.], 'yaw_rate': 0., 'land': False})
        # Deliberately do not poll state while physics advances.
        time.sleep(.55)
        state = self.session.state()
        self.assertGreaterEqual(state['frame']['time_s'], .45)
        self.assertLess(state['frame']['time_s'], .75)
        np.testing.assert_array_equal(state['frame']['velocity'], [0., 0., 0.])
        position = state['frame']['position']
        later = self.wait_time(state['frame']['time_s']+.1)
        self.assertEqual(position, later['frame']['position'])
        self.assertGreater(later['frame']['time_s'], state['frame']['time_s'])

    def test_reset_preserves_partial_audit_and_returns_waiting(self):
        self.session.start()
        self.wait_time(.1)
        self.session.reset()
        state = self.session.state()
        self.assertFalse(state['started'])
        self.assertEqual(state['frame']['time_s'], 0.)
        logs = list(Path(self.temp.name).glob('*.json.gz'))
        self.assertEqual(len(logs), 1)
        with gzip.open(logs[0], 'rt') as stream:
            log = json.load(stream)
        self.assertFalse(log['completed'])
        self.assertFalse(log['realtime_eligible'])
        self.assertEqual(log['actor'], 'ui_test')
        self.assertIsNone(log['result'])
        self.assertGreater(len(log['actions']), 0)
        self.assertIn('scene_sha256', log)
        self.session.close()
        self.assertEqual(len(list(Path(self.temp.name).glob('*.json.gz'))), 2)
        self.session.close()
        self.assertEqual(len(list(Path(self.temp.name).glob('*.json.gz'))), 2)

    def test_completed_ui_run_is_logged_and_never_human_eligible(self):
        self.session.close()
        self.session = ManualSession(analytic_scene(), nominal_profile(), actor='ui_test',
            config=FastWorldConfig(max_time_s=.1), log_dir=self.temp.name)
        self.session.start()
        self.wait_time(.1)
        state = self.session.state()
        self.assertTrue(state['done'])
        self.assertEqual(state['result']['actor'], 'ui_test')
        with gzip.open(state['log_path'], 'rt') as stream:
            log = json.load(stream)
        self.assertTrue(log['completed'])
        self.assertFalse(log['realtime_eligible'])
        self.assertEqual(len(log['actions']), 2)
        frame = state['frame']
        time.sleep(.08)
        self.assertEqual(self.session.state()['frame'], frame)

    def test_land_pulse_survives_following_zero_before_physics_tick(self):
        self.session.start()
        self.wait_time(1.)
        # Hold the physics lock so both UI commands necessarily precede a tick.
        with self.session._lock:
            self.session.set_input({'velocity_body': [0., 0., 0.], 'yaw_rate': 0., 'land': True})
            self.session.set_input({'velocity_body': [0., 0., 0.], 'yaw_rate': 0., 'land': False})
        state = self.wait_time(1.05)
        self.assertTrue(state['done'])
        self.assertTrue(state['result']['landed_safely'])

    def test_unsafe_early_land_pulse_is_consumed_not_deferred(self):
        self.session.start()
        with self.session._lock:
            self.session.set_input({'velocity_body': [0., 0., 0.], 'yaw_rate': 0., 'land': True})
            self.session.set_input({'velocity_body': [0., 0., 0.], 'yaw_rate': 0., 'land': False})
        self.wait_time(1.05)
        self.assertFalse(self.session.state()['done'])
        self.assertIsNone(self.session._pending_land_wall)

    def test_stale_land_pulse_expires_even_if_zero_inputs_are_fresh(self):
        self.session.start()
        self.wait_time(1.)
        with self.session._lock:
            self.session.set_input({'velocity_body': [0., 0., 0.], 'yaw_rate': 0., 'land': True})
            self.session._pending_land_wall -= self.session.watchdog_s + 1.
            self.session.set_input({'velocity_body': [0., 0., 0.], 'yaw_rate': 0., 'land': False})
        self.wait_time(1.05)
        self.assertFalse(self.session.state()['done'])

    def test_realtime_eligibility_rejects_lag_acceleration_partial_and_automation(self):
        self.assertTrue(_realtime_eligible('human', True, 1., 1.))
        self.assertFalse(_realtime_eligible('human', True, 1., 2.))
        self.assertFalse(_realtime_eligible('human', True, 2., 1.))
        self.assertFalse(_realtime_eligible('human', False, 1., 1.))
        self.assertFalse(_realtime_eligible('ui_test', True, 1., 1.))
        self.assertFalse(_realtime_eligible('human', True, 0., 0.))


class ReplaySessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def write(self, data, compressed=False):
        path = self.base/('log.json.gz' if compressed else 'log.json')
        opener = gzip.open if compressed else open
        with opener(path, 'wt') as stream:
            json.dump(data, stream)
        return path

    def test_exploration_replay_strips_truth_and_never_mutates(self):
        frame = {'time_s': 0., 'position': [1., 2., 3.], 'yaw': .2,
                 'ground_truth': {'secret': 1}, 'coverage': 1.}
        replay = ReplaySession(self.write({'schema_version': 'ares.exploration_log.v1',
            'actor': 'ui_test', 'scene': {'secret': True}, 'frames': [frame, dict(frame, time_s=1.)]}, True))
        first = replay.state()
        self.assertNotIn('ground_truth', first['frame'])
        self.assertNotIn('coverage', first['frame'])
        self.assertNotIn('scene', first)
        self.assertEqual(first['trajectory'], [[1., 2., 3.]])
        first['frame']['position'][0] = 99
        self.assertEqual(replay.state()['frame']['position'][0], 1.)
        replay.seek(1)
        self.assertTrue(replay.state()['done'])
        self.assertEqual(len(replay.state()['trajectory']), 2)
        replay.reset()
        self.assertEqual(replay.state()['frame_index'], 0)
        for index in (-1, 2, True):
            with self.assertRaises(ValueError):
                replay.seek(index)

    def test_raw_recording_never_reads_truth_or_invents_pose(self):
        metadata, arrays = example_recording()
        directory = self.base/'recording'
        write_recording(directory, metadata, arrays)
        (directory/'ground_truth.npz').write_bytes(b'not a valid archive')
        replay = ReplaySession(directory)
        state = replay.state()
        self.assertFalse(state['pose_available'])
        self.assertEqual(state['trajectory'], [])
        self.assertIn('placeholder', state['pose_label'])
        self.assertIsNone(state['frame']['depth_m'][0][0])
        self.assertIsNone(state['frame']['known_floor'])
        self.assertEqual(state['frame_count'], 2)

    def test_sensor_replay_uses_estimated_optical_pose(self):
        pose = np.eye(4)
        pose[:3, 3] = [1., 2., 3.]
        pose[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        replay = ReplaySession(self.write({'schema_version': 'ares.sensor_replay.v1',
            'ground_truth': {'secret': True}, 'trajectory': [{'timestamp_s': .1, 'pose': pose.tolist()}]}))
        self.assertEqual(replay.state()['frame']['position'], [1., 2., 3.])
        self.assertAlmostEqual(replay.state()['frame']['yaw'], 0.)
        self.assertNotIn('ground_truth', replay.state())

    def test_unsupported_and_empty_formats_raise_clear_errors(self):
        for data in ({'schema_version': 'other'}, {'schema_version': 'ares.exploration_log.v1', 'frames': []}):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, 'replay'):
                ReplaySession(self.write(data))

    def test_legacy_flight_log_replays_only_public_pose(self):
        from ares.flight_log import NULL_FIELDS
        data = {'schema_version': 'ares.flight_log.v1', 'scene_id': 'fixture', 'profile_id': 'fixture',
                'seed': 0, 'ground_truth': {'malformed': 'ignored evaluator metadata'},
                'samples': [{'timestamp_s': 0., 'pose_estimate': {
                    'position': [1., 2., 3.], 'quaternion_xyzw': [0., 0., 0., 1.]},
                    **dict.fromkeys(NULL_FIELDS)}]}
        state = ReplaySession(self.write(data)).state()
        self.assertEqual(state['frame']['position'], [1., 2., 3.])
        self.assertIsNone(state['frame']['depth_m'])
        self.assertNotIn('ground_truth', state)
        self.assertNotIn('scene_id', state)
