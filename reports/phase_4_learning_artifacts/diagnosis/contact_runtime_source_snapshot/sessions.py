"""Thread-safe viewer sessions; HTTP polling never advances simulation time."""
from dataclasses import asdict
import copy
import gzip
import json
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from .exploration import extract_rooms
from .exploration_scoring import build_map_report, score_exploration
from .fast_world import FastWorldConfig, GroundTruthFastWorld
from .phase3 import make_frame, _hash
from .fast_world import FAST_WORLD_VERSION
from .exploration_scoring import EXPLORATION_SCORER_VERSION


def _realtime_eligible(actor, completed, sim_duration, wall_duration):
    return bool(actor == 'human' and completed and wall_duration > 0
                and .9 <= sim_duration / wall_duration <= 1.1)


def _depth_json(depth):
    values = np.asarray(depth, dtype=float).astype(object)
    values[~np.isfinite(np.asarray(depth, dtype=float))] = None
    return values.tolist()


class ManualSession:
    """Fixed wall-clock stepping, with an explicit start and stale-input brake."""
    watchdog_s = .4

    def __init__(self, scene, profile, *, backend='cpu', actor='human', config=None,
                 log_dir=Path('reports/manual_logs')):
        if actor not in ('human', 'ui_test'):
            raise ValueError('manual actor must be human or ui_test')
        self.world = GroundTruthFastWorld(scene, profile, backend=backend, config=config,
                                          seed=scene['seed'])
        self.actor, self.log_dir = actor, Path(log_dir)
        self._lock = threading.RLock()
        self._thread = None
        self._stop = threading.Event()
        self._closed = False
        self._initialize()

    def _initialize(self):
        self._started = False
        self._start_wall = self._end_wall = None
        self._input = {'velocity_body': [0., 0., 0.], 'yaw_rate': 0., 'land': False}
        self._input_wall = 0.
        self._pending_land_wall = None
        self._actions, self._frames = [], [make_frame(self.world.observation())]
        self._result = self._report = self._log_path = self._error = None
        self._saved = False
        self._done = False

    def start(self):
        with self._lock:
            if self._closed:
                raise ValueError('session is closed')
            if self._started or self._done:
                return
            self._started = True
            self._pending_land_wall = None
            self._start_wall = time.monotonic()
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name='ares-manual-physics', daemon=True)
            self._thread.start()

    def set_input(self, controls):
        if (not isinstance(controls, dict) or set(controls) != {'velocity_body', 'yaw_rate', 'land'}
                or not isinstance(controls['velocity_body'], list) or len(controls['velocity_body']) != 3
                or type(controls['land']) is not bool):
            raise ValueError('expected velocity_body[3], yaw_rate and boolean land')
        numbers = [*controls['velocity_body'], controls['yaw_rate']]
        if not all(type(v) in (int, float) and math.isfinite(v) for v in numbers):
            raise ValueError('controls must be finite numbers')
        with self._lock:
            if self._closed:
                raise ValueError('session is closed')
            clipped = copy.deepcopy(controls)
            velocity = np.asarray(controls['velocity_body'], dtype=float)
            largest = float(np.max(np.abs(velocity)))
            if largest:
                unit = velocity / largest
                velocity = unit * min(largest, self.world.config.max_speed / float(np.linalg.norm(unit)))
            clipped['velocity_body'] = velocity.tolist()
            clipped['yaw_rate'] = float(np.clip(controls['yaw_rate'], -self.world.config.max_yaw_rate, self.world.config.max_yaw_rate))
            self._input, self._input_wall = clipped, time.monotonic()
            if controls['land'] and self._started and not self._done:
                self._pending_land_wall = self._input_wall

    def _run(self):
        dt = self.world.config.dt
        deadline = self._start_wall + dt
        try:
            while not self._stop.wait(max(0., deadline-time.monotonic())):
                with self._lock:
                    if self._closed or self._done:
                        return
                    now = time.monotonic()
                    controls = self._input if now-self._input_wall <= self.watchdog_s else {
                        'velocity_body': [0., 0., 0.], 'yaw_rate': 0., 'land': False}
                    # A UI button pulse may be followed by a zero HTTP command
                    # before the next tick. Consume it once, with its own expiry.
                    land = (self._pending_land_wall is not None
                            and now-self._pending_land_wall <= self.watchdog_s)
                    self._pending_land_wall = None
                    yaw = self.world.observation()['yaw']
                    vx, vy, vz = controls['velocity_body']
                    c, s = math.cos(yaw), math.sin(yaw)
                    action = [c*vx-s*vy, s*vx+c*vy, vz, controls['yaw_rate'], float(land)]
                    self.world.step(np.asarray([action], dtype=np.float32))
                    self._actions.append(action)
                    truth = self.world.ground_truth_state()
                    self._done = truth['done']
                    if len(self._actions) % max(1, round(.5/dt)) == 0 or self._done:
                        self._frames.append(make_frame(self.world.observation()))
                    if self._done:
                        self._end_wall = time.monotonic()
                        observed = self.world.observation()
                        rooms = extract_rooms(observed['known_grid'], observed['resolution'], observed['origin'])
                        self._report = build_map_report(observed, rooms)
                        self._result = score_exploration(self.world, self.world.snapshot_for_scoring(), self._report)
                        self._result['actor'] = self.actor
                        self._persist()
                        return
                deadline += dt
        except Exception as exc:
            with self._lock:
                self._error = f'{type(exc).__name__}: {exc}'
                self._end_wall = time.monotonic()
                self._persist()

    def _persist(self):
        if self._saved:
            return
        final = make_frame(self.world.observation())
        if self._frames[-1]['time_s'] != final['time_s']:
            self._frames.append(final)
        wall = 0. if self._start_wall is None else (self._end_wall or time.monotonic())-self._start_wall
        ratio = final['time_s']/wall if wall > 0 else None
        completed = bool(self._done and self._result is not None)
        eligible = _realtime_eligible(self.actor, completed, final['time_s'], wall)
        log = {'schema_version': 'ares.exploration_log.v1', 'actor': self.actor,
               'scene': self.world.scene, 'profile': self.world.profile, 'seed': self.world.seed,
               'scene_sha256': _hash(self.world.scene), 'world_version': FAST_WORLD_VERSION,
               'scorer_version': EXPLORATION_SCORER_VERSION,
               'world_config': asdict(self.world.config), 'backend': self.world.backend,
               'actions': self._actions, 'frames': self._frames, 'report': self._report,
               'result': self._result, 'completed': completed, 'started': self._started,
               'wall_duration_s': wall, 'sim_wall_ratio': ratio, 'realtime_eligible': eligible,
               'error': self._error, 'note': 'Actual human input required; ui_test is excluded from human baseline.'}
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f'{self.actor}-{uuid.uuid4().hex}.json.gz'
        with path.open('xb') as output:
            with gzip.GzipFile(fileobj=output, mode='wb', mtime=0) as compressed:
                compressed.write(json.dumps(log, allow_nan=False, separators=(',', ':')).encode())
        self._log_path, self._saved = str(path), True

    def state(self):
        with self._lock:
            return {'mode': 'live', 'actor': self.actor, 'started': self._started,
                    'done': self._done, 'frame': make_frame(self.world.observation()),
                    'result': copy.deepcopy(self._result) if self._done else None,
                    'controls': {'max_speed': self.world.config.max_speed,
                                 'max_yaw_rate': self.world.config.max_yaw_rate},
                    'frame_index': len(self._frames)-1, 'frame_count': len(self._frames),
                    'log_path': self._log_path, 'error': self._error, 'pose_available': True}

    def _stop_worker(self):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        self._thread = None

    def reset(self):
        self._stop_worker()
        with self._lock:
            if self._closed:
                raise ValueError('session is closed')
            self._persist()
            self.world.reset()
            self._initialize()

    def close(self):
        self._stop_worker()
        with self._lock:
            if not self._closed:
                self._end_wall = self._end_wall or time.monotonic()
                self._persist()
                self._closed = True


class ReplaySession:
    """Observed-only replay adapter; raw depth never acquires a fabricated pose."""
    def __init__(self, path):
        from .flight_log import validate_flight_log
        from .recording import read_recording
        path = Path(path)
        self._lock = threading.RLock()
        self._index = 0
        self.actor, self.result = 'recorded', None
        self.pose_available = True
        self.label = 'Recorded pose'
        self._frames = []
        if path.is_dir():
            _, arrays = read_recording(path)  # Deliberately never open ground_truth.npz.
            self.pose_available = False
            self.label = 'Raw depth only; pose unavailable (display placeholder)'
            self._frames = [self._base_frame(float(t), depth=_depth_json(d))
                            for t, d in zip(arrays['timestamps_s'], arrays['depth_m'])]
        else:
            try:
                opener = gzip.open if path.suffix == '.gz' else open
                with opener(path, 'rt', encoding='utf-8') as stream:
                    data = json.load(stream, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
                version = data.get('schema_version')
                if version == 'ares.exploration_log.v1':
                    self.actor = data.get('actor', 'recorded')
                    self._frames = [self._public_frame(f) for f in data['frames']]
                    self.result = copy.deepcopy(data.get('result'))
                elif version == 'ares.flight_log.v1':
                    validate_flight_log({key: value for key, value in data.items() if key != 'ground_truth'})
                    self._frames = [self._base_frame(f['timestamp_s'], position=f['pose_estimate']['position'],
                        yaw=self._quaternion_yaw(f['pose_estimate']['quaternion_xyzw'])) for f in data['samples']]
                    self.label = 'Phase 0 idealized harness pose; no recorded depth'
                elif version == 'ares.sensor_replay.v1':
                    for record in data['trajectory']:
                        pose = np.asarray(record['pose'], dtype=float)
                        if pose.shape != (4, 4) or not np.isfinite(pose).all():
                            raise ValueError('invalid replay pose')
                        self._frames.append(self._base_frame(record['timestamp_s'], position=pose[:3, 3].tolist(),
                            yaw=math.atan2(float(pose[1, 2]), float(pose[0, 2]))))
                    self.label = 'Estimated sensor-replay pose; depth not embedded'
                else:
                    raise ValueError(f'unsupported replay format: {version}')
            except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
                raise ValueError('invalid or unsupported replay format') from exc
        if not self._frames:
            raise ValueError('replay requires at least one frame')

    @staticmethod
    def _quaternion_yaw(q):
        x, y, z, w = q
        return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))

    @staticmethod
    def _base_frame(timestamp, position=None, yaw=0., depth=None):
        if (type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0
                or type(yaw) not in (int, float) or not math.isfinite(yaw)):
            raise ValueError('invalid replay frame time or yaw')
        if position is not None and (not isinstance(position, list) or len(position) != 3
                or not all(type(v) in (int, float) and math.isfinite(v) for v in position)):
            raise ValueError('invalid replay frame position')
        return {'time_s': float(timestamp), 'position': position or [0., 0., 0.],
                'velocity': [0., 0., 0.], 'home': None, 'yaw': float(yaw),
                'depth_m': depth, 'known_floor': None, 'observed_voxels': 0}

    @classmethod
    def _public_frame(cls, frame):
        allowed = {'position', 'velocity', 'home', 'yaw', 'time_s', 'depth_m', 'known_floor', 'observed_voxels'}
        if not isinstance(frame, dict) or not {'position', 'yaw', 'time_s'} <= frame.keys():
            raise ValueError('invalid exploration replay frame')
        output = cls._base_frame(frame['time_s'], position=frame['position'], yaw=frame['yaw'])
        output.update({key: copy.deepcopy(value) for key, value in frame.items() if key in allowed})
        return output

    def state(self):
        with self._lock:
            return {'mode': 'replay', 'actor': self.actor, 'frame': copy.deepcopy(self._frames[self._index]),
                    'frame_index': self._index, 'frame_count': len(self._frames),
                    'done': self._index == len(self._frames)-1, 'result': copy.deepcopy(self.result) if self._index == len(self._frames)-1 else None,
                    'pose_available': self.pose_available, 'pose_label': self.label,
                    'trajectory': [copy.deepcopy(frame['position']) for frame in self._frames[:self._index+1]]
                                  if self.pose_available else []}

    def seek(self, frame_index):
        if type(frame_index) is not int or not 0 <= frame_index < len(self._frames):
            raise ValueError('frame index out of range')
        with self._lock:
            self._index = frame_index

    def reset(self):
        self.seek(0)

    def start(self):
        pass

    def set_input(self, controls):
        raise ValueError('replay has no live controls')

    def close(self):
        pass
