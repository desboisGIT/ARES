# Phase 2 primary implementation references

Verified against official vendor documentation/source on 2026-11-12. APIs and option availability must also be checked against the installed SDK and attached hardware.

- [Python pipeline bindings](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/pyrs_pipeline.cpp): configuration plus callback capture; callback mode cannot be mixed with polling/waiting. The recorder copies frames in the callback and receives asynchronous motion rather than retaining only the latest motion sample per depth frame.
- [Official depth conversion Python example](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py): configured stream intrinsics/depth sensor scale and NumPy depth access. ARES retains the native depth projection; color alignment is not requested.
- [Frame API and metadata definitions](https://github.com/realsenseai/librealsense/blob/master/include/librealsense2/h/rs_frame.h): timestamp-domain meanings, per-stream frame counters and emitter-mode metadata. Global time converts camera time to the system-clock reference. Emitter mode zero means all emitters disabled; missing metadata cannot establish an emitter state.
- [Sensor option definitions](https://github.com/realsenseai/librealsense/blob/master/include/librealsense2/h/rs_option.h): runtime-tested `emitter_on_off`, `emitter_enabled` and `global_time_enabled`. Option enumeration does not establish support on every device or firmware.
- [Projection, coordinate systems and extrinsics](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0): optical axes, intrinsics and transformations between stream coordinate systems. ARES applies factory motion-to-depth rotations before its documented body basis.
- [Record and playback API](https://github.com/realsenseai/librealsense/tree/master/examples/record-playback): native bag capture and playback preserve camera streams separately from the normalized ARES arrays.

The implementation uses the sensor SDK for capture, not a stereo-VIO library. Native bag support, emitter alternation, motion extrinsics, clock conversion and capture rates have not been exercised against physical hardware in this workspace. No unverified camera performance specifications or compatibility guarantees are used as measured Phase 2 evidence.

## Optional runtime packages and packaging

The base install remains standard-library-only. `.[simulation]` adds NumPy for CPU sensor/replay checks; `.[cuda]` adds the CUDA 13 CuPy wheel and its toolkit extra for desktop benchmarks; `.[recording]` adds NumPy and the official RealSense Python wrapper. CI and the CPU Docker image install only `.[simulation]` before the Phase 2 smoke check. They do not measure GPU or onboard performance. Run `make phase2 PYTHON=.venv/bin/python` for the explicitly selected CUDA environment.

- [NumPy source license](https://github.com/numpy/numpy/blob/main/doc/source/license.rst) is BSD-3-Clause; installed wheel metadata also lists bundled permissive components. This does not replace reviewing its exact wheel notices.
- [CuPy license](https://github.com/cupy/cupy/blob/main/LICENSE) is MIT. The installed `cupy-cuda13x` 14.2.0 package requests `cuda-pathfinder` and, through `[ctk]`, CUDA toolkit libraries. [NVIDIA cuda-pathfinder source license](https://github.com/NVIDIA/cuda-python/blob/main/cuda_pathfinder/LICENSE) is Apache-2.0.
- [CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html) governs NVIDIA toolkit components. Installed NVIDIA wheels identify proprietary licenses; their exact redistribution terms remain pending. They are recorded as desktop-only dependencies, not approved components of the permissive deployed stack.
- [Official RealSense Python wrapper instructions](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/readme.md) point to [the official package files](https://pypi.org/project/pyrealsense2/#files). The recording extra uses `pyrealsense2>=2.59.1`, the current stable family checked during implementation. Availability depends on Python ABI, OS and architecture; Windows Python 3.14 documentation does not establish Linux or ARM wheel availability. Confirm the exact target wheel or follow vendor source-build instructions. No recording SDK or hardware compatibility test was performed here.

`LICENSES.yaml` records the actually installed NumPy/CuPy/CUDA versions and individual NVIDIA distributions observed in the workspace virtual environment. The dependency constraints are not a lockfile or a deployment license approval.


## Global-clock startup behavior

[The official SDK global timestamp reader implementation](https://github.com/realsenseai/librealsense/blob/master/src/global_timestamp_reader.cpp) reports global-time domain only when the option is enabled and its time-difference keeper has become ready (`_ts_is_ready`). Before then, frames can retain the hardware-clock domain. ARES therefore uses an all-configured-stream readiness gate rather than rejecting ordinary startup: discard normalized samples during warmup, retain native bag frames, and enforce global time strictly after readiness. This gate is bounded by the existing total capture duration and is tested without an SDK. Physical camera timing remains untested.
