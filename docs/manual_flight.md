# Local manual flight and replay

Phase 3 provides a browser viewer for the simulator and normalized flight logs. It runs on loopback, uses no external assets, and controls a simulated point-mass drone. It does not connect to a flight device.

Install the CUDA extra for GPU simulation, then launch a human session or inspect a saved log:

```sh
.venv/bin/python -m pip install -e '.[cuda]'
.venv/bin/python -m ares fly --seed 100 --split train --backend cuda --actor human --log-dir logs/manual
.venv/bin/python -m ares view-log LOG_PATH
```

Use `--backend cpu` with the simulation extra for a CPU manual session. Replace `LOG_PATH` with the log to replay.

Open the local URL printed by the CLI in a browser. Click **Start controls** when ready: this begins the mission clock. **Brake / zero controls**, Space, switching windows, or hiding the browser zeros commands. After starting, braking does not stop the mission clock; thinking and reaction time count. Reset returns to a fresh run waiting for Start. **Land** explicitly requests landing; the viewer never lands automatically.

## Controls

| Input | Motion |
| --- | --- |
| W / S | Forward / backward in body coordinates |
| A / D | Left / right in body coordinates |
| Q / E | Up / down |
| Left / right arrow | Turn left / right |
| Space | Brake and disable controls until Start |
| L or Land button | Request landing |
| Gamepad left stick | Forward/back and left/right |
| Gamepad right stick | Horizontal: yaw; vertical: up/down |

The gamepad has a 10% dead zone. Use the visible Land button with a gamepad. The keyboard remains available without a gamepad. Browsers may require a button press on the controller before exposing it. Movement magnitude is capped at the session's speed limit; yaw uses its yaw-rate limit. Input and state requests run at up to 20 Hz, with at most one outstanding request of each kind. The session advances physics on its own fixed wall-clock schedule, independently of browser requests.

## Reading the display

The large first-person image colours each forward-facing depth pixel by optical depth and camera-ray position. It is a depth visualization, not a colour camera or photoreal rendering. The small map shows only observed floor cells and the current pose; a home marker appears when the session supplies one. Elapsed time, yaw, and observed-cell count are available during flight. Ground-truth scoring results appear only when the run is complete.

An actor marked **UI TEST** is excluded from human baseline measurements. Opening the viewer or exercising it automatically does not establish a human benchmark. Real human runs must be recorded with their human actor metadata by the session/CLI and included in the benchmark report explicitly.

For the measured benchmark cohort, fly with `--split test --seed 50000` (or
another seed through 50099). The train example above is for practice. After real
human runs, replay-verify and aggregate the logs without rerunning the classical
benchmark:

```sh
.venv/bin/python -m ares human-benchmark logs/manual/*.json.gz --baseline reports/phase_3.json --output reports/phase_3_human.json
```

Only completed, compatible human runs at real-time speed qualify. Partial logs
are retained for audit and rejected with reasons. Completed collisions/timeouts
count as failed attempts. Metrics are independently recomputed from actions;
actor identity and wall duration remain unsigned operator attestations.

Replay uses the same display with a frame slider and Play/Pause. Legacy frames without depth show “No depth recorded for this frame”. Raw recordings without an estimated pose show **Pose unavailable** and omit pose and home markers; they do not claim a zero position. Pose-only logs show XYZ coordinates in metres and an XY trail when recorded poses are available; this does not infer occupancy. The timeline remains available. Play advances through recorded frames at 10 frames/second; use the slider for exact frame inspection. The displayed elapsed time remains the recorded timestamp, not playback duration.

## Embedding and HTTP interface

`ares.viewer.build_server(session, host='127.0.0.1', port=8766)` returns a standard-library `ThreadingHTTPServer`. Use `port=0` for an ephemeral test port. Its caller owns shutdown, server closure, and session lifetime. `serve_viewer(session, host=..., port=...)` blocks until Ctrl-C, then closes the server and session.

A session supplies thread-safe `state()`, `set_input(action)`, `reset()`, and `close()` methods; live sessions additionally supply `start()`, and replay sessions supply `seek(frame_index)`. State has `mode`, `actor`, `frame`, `frame_count`, `frame_index`, `done`, `result`, and `controls`; live state also supplies `started` so the viewer distinguishes waiting from braking after launch. Frames may flag `pose_available: false` when no pose estimate exists. Frames include `position`, `yaw`, `time_s`, `depth_m`, and `known_floor`. Floor shape is `[nx, ny]`, with dense row-major cells at `cells[y * nx + x]`, origin `[x, y]`, and a resolution in metres. Cell values are 0 unknown, 1 free, and 2 occupied. Controls supply `max_speed` and `max_yaw_rate`.

| Route | Payload / response |
| --- | --- |
| `GET /` or `/index.html` | Packaged HTML viewer |
| `GET /api/state` | Normalized session JSON |
| `POST /api/start` | `{}` |
| `POST /api/reset` | `{}` |
| `POST /api/action` | `{"velocity_body": [forward, left, up], "yaw_rate": radians_per_second, "land": false}` |
| `POST /api/seek` | `{"frame_index": nonnegative_integer}` |

Action velocities are metres per second. POST schemas reject extra fields, nonfinite values, booleans in numeric fields, invalid types, and bodies above 16 KiB. The HTTP layer validates shape and types; the session enforces simulation limits and replay index bounds. The server accepts only loopback bindings and loopback Host headers, checks browser POST origins, provides no CORS access, and exposes only the listed routes. It does not serve a filesystem directory. No browser action writes a score or invents a human timing record; persistence belongs to the session.
