# Sensor recording v1

`ares.recording` is the CPU replay boundary. It requires the optional NumPy
dependency. Existing Phase 0/1 code remains usable without NumPy.

A directory contains `metadata.json` and `frames.npz`. Native device recordings
such as `capture.bag` may also be retained. Runtime replay opens only the two
specified files. Synthetic validation ground truth belongs in the separate
`ground_truth.npz` file; `read_recording` never opens or discovers that file.

The metadata object has exactly these fields (unknown fields are rejected):

```json
{
  "schema_version": "ares.sensor_recording.v1",
  "recording_id": "example",
  "source": "synthetic",
  "seed": 0,
  "profile_id": "drone_nominal",
  "camera": {
    "width": 320,
    "height": 240,
    "fx": 170.0,
    "fy": 170.0,
    "cx": 159.5,
    "cy": 119.5
  },
  "frames_file": "frames.npz"
}
```

`source` is `synthetic` or `real`; `seed` is an integer (use a documented capture
seed such as zero for a real recording). Identifiers are nonempty strings. Camera
dimensions are positive integers; focal lengths are positive finite numbers, and
the finite principal point lies inside the image. `frames_file` must equal
`frames.npz`, and an escaping symlink is rejected. JSON duplicate keys and
nonstandard NaN/Infinity constants are rejected.

The NPZ archive contains exactly these required arrays:

| Array | Shape | Meaning |
| --- | --- | --- |
| `timestamps_s` | `(F,)` | Depth frame timestamps, seconds |
| `depth_m` | `(F,H,W)` | `float32` camera optical Z depth, metres |
| `imu_timestamps_s` | `(M,)` | IMU sample timestamps, seconds |
| `imu_accel_mps2` | `(M,3)` | Specific acceleration in body axes, m/s² |
| `imu_gyro_rps` | `(M,3)` | Body angular velocity, rad/s |

`F` must be positive; `M` may be zero for a depth-only recording. All times are
finite and strictly increasing within their stream, with a common clock origin.
Streams need not have equal lengths or matching sample times. Numeric arrays are
required; object arrays and pickle loading are forbidden. All values except depth
holes must be finite. Valid depth is strictly positive; holes are NaN (never zero
or infinity). Depth is **optical Z, not Euclidean range**: pixel `(u,v)` becomes
`[(u-cx)*Z/fx, (v-cy)*Z/fy, Z]`. Therefore a slanted ray's length exceeds its Z.

An optional flow stream must provide all three arrays together:

| Array | Shape | Meaning |
| --- | --- | --- |
| `flow_timestamps_s` | `(K,)` | Flow timestamps, seconds |
| `flow_velocity_body_mps` | `(K,2)` | Body forward and left velocities, m/s |
| `flow_range_m` | `(K,)` | Positive downward range, metres |

The current odometry baseline does not fuse flow. IR left/right and color streams
are reserved for a future version and are not supported NPZ fields in v1. Retain
the native bag when these original streams or asynchronous IMU samples matter;
do not silently place them into unknown NPZ fields.

World axes are right-handed with Z up. Body axes are X forward, Y left, Z up.
Optical camera axes are X right, Y down, Z forward. Poses are 4×4 camera-to-world
transforms. The default level camera-to-body rotation is:

```text
[[ 0, 0, 1],
 [-1, 0, 0],
 [ 0,-1, 0]]
```

`CameraIntrinsics.from_profile(profile)` (also the module-level `from_profile`)
derives centered pinhole intrinsics from the nominal depth dimensions and fields
of view. It supports profile versions carrying the same parameter names. Measured
real-device intrinsics should instead be stored directly in the recording.

`write_recording(path, metadata, arrays)` validates before writing and accepts an
existing directory containing native capture files. It refuses to overwrite
either existing contract file. `read_recording(path)` returns `(metadata, arrays)`
and validates the same shape, timestamp and value rules. Validation errors use
`ares.contracts.ContractError`; intrinsics construction errors use `ValueError`.
