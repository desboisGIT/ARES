#!/usr/bin/env python3
"""Bounded RealSense capture; optional SDK is imported only by main()."""
from __future__ import annotations
import argparse
import json
import math
import shutil
from pathlib import Path
import sys
import threading


class ClockWarmupGate:
    """Discard startup frames until every configured stream reports global time."""
    def __init__(self, expected_keys):
        self.expected = frozenset(expected_keys)
        if not self.expected:
            raise ValueError('clock gate needs configured stream keys')
        self.global_seen = set()
        self.discard_counts = {}
        self.observed_domains = {}
        self.ready = False

    def observe(self, key, is_global, domain):
        if key not in self.expected:
            raise RuntimeError('Unexpected capture stream: ' + key)
        self.observed_domains.setdefault(key, set()).add(domain)
        if self.ready:
            if not is_global:
                raise RuntimeError('Timestamp left global-time domain after clock readiness: ' + key)
            return True
        self.discard_counts[key] = self.discard_counts.get(key, 0) + 1
        if is_global:
            self.global_seen.add(key)
        else:
            self.global_seen.discard(key)
        self.ready = self.global_seen == self.expected
        # The readiness-transition frame is discarded too: no sample from
        # before all streams demonstrated readiness enters normalized arrays.
        return False

    def require_ready(self):
        if not self.ready:
            missing = ', '.join(sorted(self.expected - self.global_seen))
            raise RuntimeError('Global clock warmup did not complete; missing streams: ' + missing)


def normalize_spool(raw_path, shape, count, scale, output_path):
    """Convert a uint16 disk spool to float32 .npy using at most eight frames.

    The caller owns the returned memmap and must close it after saving. Failed
    conversions retain their files for inspection; existing output is refused.
    """
    import numpy as np
    if (len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape)
            or type(count) is not int or not 0 < count <= shape[0]):
        raise ValueError('shape must contain three positive integers; count must fit its frame cap')
    if isinstance(scale, (bool, np.bool_)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('depth scale must be finite and positive')
    if Path(raw_path).stat().st_size != math.prod(shape) * 2:
        raise ValueError('raw spool size does not match uint16 shape')
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise ValueError('normalized spool already exists')
    raw = np.memmap(raw_path, dtype=np.uint16, mode='r', shape=shape)
    normalized = None
    try:
        normalized = np.lib.format.open_memmap(output_path, mode='w+', dtype=np.float32,
                                               shape=(count, shape[1], shape[2]))
        for start in range(0, count, 8):
            end = min(start + 8, count)
            block = raw[start:end].astype(np.float32)
            block *= scale
            block[raw[start:end] == 0] = np.nan
            if np.isinf(block).any():
                raise ValueError('depth scale overflows float32 metres')
            normalized[start:end] = block
        normalized.flush()
        return normalized
    except BaseException:
        if normalized is not None:
            normalized._mmap.close()
        raise
    finally:
        raw._mmap.close()


def capture(args):
    existed = args.output.exists()
    try:
        return _capture(args)
    except BaseException as exc:
        if not existed and args.output.is_dir():
            report_path = args.output / 'capture_report.json'
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            report.update(status='failed', failure=str(exc),
                          recovery='Raw bag and any depth spool files retained; output is incomplete.')
            report_path.write_text(json.dumps(report, indent=2) + '\n')
        raise


def _capture(args):
    try:
        import numpy as np
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError('Capture requires numpy and the optional pyrealsense2 SDK') from exc
    if args.output.exists():
        raise ValueError('output already exists; choose a fresh directory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shape = (args.max_frames, args.height, args.width)
    # Raw uint16 + normalized float32 + conservatively uncompressed NPZ, plus
    # separately reserved native-bag space (actual bag size depends on streams).
    required_bytes = math.prod(shape) * 10 + args.bag_reserve_gib * 1024**3
    if shutil.disk_usage(args.output.parent).free < required_bytes:
        raise ValueError('Insufficient disk for depth spools, normalized NPZ and native-bag reserve')
    if not rs.context().query_devices():
        raise RuntimeError('No RealSense device found; connect a camera with depth, stereo IR and IMU')
    args.output.mkdir()
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    for index in (1, 2):
        config.enable_stream(rs.stream.infrared, index, args.width, args.height, rs.format.y8, args.fps)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    config.enable_record_to_file(str(args.output / 'capture.bag'))
    resolved = config.resolve(rs.pipeline_wrapper(pipeline))
    clock_gate = ClockWarmupGate(str(p.stream_type()) + ':' + str(p.stream_index())
                                for p in resolved.get_streams())
    depth_profile = resolved.get_stream(rs.stream.depth)
    intr = depth_profile.as_video_stream_profile().get_intrinsics()
    if any(abs(c) > 1e-12 for c in intr.coeffs):
        raise RuntimeError("Depth has nonzero distortion; normalized pinhole recording requires rectified depth")
    for device_sensor in resolved.get_device().sensors:
        if not device_sensor.supports(rs.option.global_time_enabled):
            raise RuntimeError('All device sensors must support global_time_enabled for a common clock')
        device_sensor.set_option(rs.option.global_time_enabled, 1)
    sensor = resolved.get_device().first_depth_sensor()
    scale = sensor.get_depth_scale()
    if args.emitter == 'alternate':
        option = rs.option.emitter_on_off
        if not sensor.supports(option):
            raise RuntimeError('This device/firmware does not expose emitter_on_off')
        sensor.set_option(option, 1)
        if sensor.get_option(option) != 1:
            raise RuntimeError('Alternating emitter option did not read back enabled')
    else:
        if not sensor.supports(rs.option.emitter_enabled):
            raise RuntimeError('This device does not expose emitter_enabled')
        if sensor.supports(rs.option.emitter_on_off):
            sensor.set_option(rs.option.emitter_on_off, 0)
        sensor.set_option(rs.option.emitter_enabled, int(args.emitter == 'on'))
    # SDK extrinsics are column-major rotations from motion stream to depth.
    basis = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    rotations = {}
    for kind in (rs.stream.accel, rs.stream.gyro):
        extr = resolved.get_stream(kind).get_extrinsics_to(depth_profile)
        rotations[str(kind)] = basis @ np.array(extr.rotation).reshape(3, 3, order='F')
    raw_path = args.output / '.depth_raw.u16'
    normalized_path = args.output / '.depth_normalized.npy'
    raw_depth = np.memmap(raw_path, dtype=np.uint16, mode='w+', shape=shape)
    depth_times, motion = [], {str(rs.stream.accel): [], str(rs.stream.gyro): []}
    last, gaps, emitter_frames, domains, errors = {}, {}, [], set(), []
    lock = threading.Lock()
    ready, done = threading.Event(), threading.Event()
    def callback(frame):
        if not ready.is_set() or done.is_set():
            return
        try:
            frames = list(frame.as_frameset()) if frame.is_frameset() else [frame]
            with lock:
                for f in frames:
                    kind = f.get_profile().stream_type()
                    key = str(kind) + ':' + str(f.get_profile().stream_index())
                    domain = str(f.get_frame_timestamp_domain())
                    if not clock_gate.observe(key, f.get_frame_timestamp_domain() == rs.timestamp_domain.global_time, domain):
                        continue
                    stamp = f.get_timestamp() / 1000.
                    domains.add(domain)
                    if len(domains) > 1:
                        raise RuntimeError('Mixed timestamp domains; capture rejected (no host-arrival alignment)')
                    number = f.get_frame_number()
                    if key in last:
                        old_t, old_n = last[key]
                        if number == old_n and stamp == old_t:
                            continue  # a pipeline frameset may reuse an unchanged frame
                        if stamp <= old_t or number <= old_n:
                            raise RuntimeError('Nonmonotonic timestamp/frame counter: ' + key)
                        gaps[key] = gaps.get(key, 0) + max(0, number-old_n-1)
                    last[key] = (stamp, number)
                    if kind == rs.stream.depth:
                        if len(depth_times) >= args.max_frames:
                            done.set()
                            break
                        raw_depth[len(depth_times)] = np.asanyarray(f.get_data())
                        depth_times.append(stamp)
                    if kind in (rs.stream.depth, rs.stream.infrared):
                        meta = rs.frame_metadata_value.frame_emitter_mode
                        mode = int(f.get_frame_metadata(meta)) if f.supports_frame_metadata(meta) else None
                        emitter_frames.append({'stream': key, 'timestamp_s': stamp,
                                               'frame_number': number, 'emitter_mode': mode})
                        if args.emitter == 'alternate' and mode is None:
                            raise RuntimeError('Alternation requested but frame emitter metadata unavailable')
                    if kind in (rs.stream.accel, rs.stream.gyro):
                        samples = motion[str(kind)]
                        if len(samples) >= 300000:
                            raise RuntimeError('IMU sample cap exceeded')
                        v = f.as_motion_frame().get_motion_data()
                        samples.append((stamp, rotations[str(kind)] @ np.array([v.x, v.y, v.z])))
        except Exception as exc:
            errors.append(str(exc))
            done.set()
    started = False
    try:
        pipeline.start(config, callback)
        started = True
        ready.set()
        done.wait(args.duration)
    finally:
        done.set()
        try:
            if started:
                pipeline.stop()
        finally:
            try:
                raw_depth.flush()
            finally:
                raw_depth._mmap.close()
    try:
        clock_gate.require_ready()
    except RuntimeError as exc:
        errors.append(str(exc))
    report = {'clock_ready': clock_gate.ready,
              'clock_warmup_discard_counts': clock_gate.discard_counts,
              'clock_observed_domains': {k: sorted(v) for k, v in clock_gate.observed_domains.items()},
              'timestamp_domains': sorted(domains), 'frame_counter_gaps': gaps,
              'emitter_requested': args.emitter, 'emitter_frames': emitter_frames,
              'motion_to_body_rotation': {k: v.tolist() for k, v in rotations.items()},
              'errors': errors, 'raw_file': 'capture.bag', 'depth_scale_m': scale}
    (args.output / 'capture_report.json').write_text(json.dumps(report, indent=2) + '\n')
    if errors:
        raise RuntimeError('; '.join(errors))
    accel, gyro = motion[str(rs.stream.accel)], motion[str(rs.stream.gyro)]
    if not depth_times or len(accel) < 2 or len(gyro) < 2:
        raise RuntimeError('Insufficient depth/IMU samples; raw bag retained, no normalized recording written')
    at, gt = np.array([x[0] for x in accel]), np.array([x[0] for x in gyro])
    times = np.unique(np.concatenate((at, gt)))
    times = times[(times >= max(at[0], gt[0])) & (times <= min(at[-1], gt[-1]))]
    if len(times) < 2:
        raise RuntimeError('IMU streams have no overlapping time interval')
    origin = min(depth_times[0], times[0])
    def interpolate(samples, ts):
        values = np.array([x[1] for x in samples])
        return np.column_stack([np.interp(times, ts, values[:, i]) for i in range(3)])
    depth = normalize_spool(raw_path, shape, len(depth_times), scale, normalized_path)
    arrays = {'timestamps_s': np.array(depth_times)-origin, 'depth_m': depth,
              'imu_timestamps_s': times-origin, 'imu_accel_mps2': interpolate(accel, at),
              'imu_gyro_rps': interpolate(gyro, gt)}
    # Source checkout invocation works without requiring an editable installation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ares.recording import write_recording
    metadata = {'schema_version': 'ares.sensor_recording.v1', 'recording_id': args.recording_id,
                'source': 'real', 'seed': args.seed, 'profile_id': args.profile_id,
                'camera': {'width': intr.width, 'height': intr.height, 'fx': intr.fx,
                           'fy': intr.fy, 'cx': intr.ppx, 'cy': intr.ppy}, 'frames_file': 'frames.npz'}
    try:
        write_recording(args.output, metadata, arrays)
    finally:
        depth._mmap.close()
    # Only delete our own two temporary files after normalized save succeeds.
    raw_path.unlink()
    normalized_path.unlink()
    report.update(status='complete', depth_frames=len(depth_times))
    (args.output / 'capture_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'depth_frames': len(depth_times),
                      'imu_samples': len(times), 'frame_counter_gaps': gaps}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--recording-id', required=True)
    parser.add_argument('--profile-id', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--duration', type=float, default=60, help='capture duration in seconds (maximum 600)')
    parser.add_argument('--max-frames', type=int, default=2000, help='depth frame cap (maximum 20000)')
    parser.add_argument('--bag-reserve-gib', type=float, default=4,
                        help='extra free disk reserved for native bag and overhead (default 4 GiB)')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--emitter', choices=('on', 'off', 'alternate'), default='on')
    args = parser.parse_args()
    if (not 0 < args.duration <= 600 or not 0 < args.max_frames <= 20000
            or not math.isfinite(args.bag_reserve_gib) or args.bag_reserve_gib < 1
            or not all(v > 0 for v in (args.max_frames, args.width, args.height, args.fps))):
        parser.error('duration must be in (0,600], cap in [1,20000], dimensions/fps positive, bag reserve >=1 GiB')
    try:
        capture(args)
    except (RuntimeError, ValueError, AttributeError, OSError) as exc:
        parser.exit(1, f'Capture failed: {exc}\n')


if __name__ == '__main__':
    main()
