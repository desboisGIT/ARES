# Build plan v3: autonomous drone that maps unknown buildings at speed (€1,000 budget)

This plan supersedes `build_plan.md` and `sim_build_prompt.md`. It is standalone. Coding agents build the software; phases marked **human task** need a person.

## 1. Goal

A small quadrotor is placed at the entrance of a home or building it has never seen. It takes off, flies through every reachable room as fast as it safely can, and produces a live 3D map and a room-level floor plan. It never needs a radio link to fly. Marking people on the map is an add-on module, built last.

Primary metric: **time to map**. An episode passes when the map covers at least 95% of the observable space, the floor plan matches the true one (IoU at least 0.80 on a 20 cm grid), every reachable room is found, there are zero collisions, and the drone lands safely. Passing runs are ranked by time.

## 2. Design decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Main sensor | Active stereo depth camera (Intel RealSense D435i; Orbbec Gemini 335 as the cheaper alternative) | Direct range measurements, works on blank walls and in the dark thanks to its infrared projector, 90 frames per second, built-in IMU, about 72 g. Simulated depth transfers to reality far better than camera images |
| Compute | Onboard single-board computer (8-core ARM, 8 GB) | No radio dependence and low latency. The desktop is for training and viewing only |
| Flight policy | Small neural network on low-resolution depth, trained with differentiable physics | Published results reach high speed with zero-shot transfer, on a computer far weaker than ours |
| Position tracking | Visual-inertial odometry from the camera's stereo pair and IMU, backed by a downward optical-flow and range sensor | Standard, cheap, runs on a small board |
| Mapping | Classical voxel map | Accurate and well understood; no learning needed |
| Exploration | Learned policy on top of a classical planner | The classical planner guarantees a working system; learning is added only where it measurably saves time |
| Gaze | The drone's yaw is controlled as its "eyes" | The camera sees about 87° wide, so the policy must choose where to look as well as where to fly |
| Safety | Classical shield that caps speed to what can stop inside known free space | Learned parts can never command a crash the shield can prevent |
| Firmware | PX4 or ArduPilot in offboard mode (thrust and body rates) | Whichever supports the chosen board best; record the license |
| Photoreal simulation | None for the core system | Not needed when flight uses depth |

Reference work to study before coding: "Learning vision-based agile flight via differentiable physics" (training method), DiffAero (simulator), FUEL and FALCON (exploration planners), SUPER (the safety principle of always keeping a stopping path in known free space).

Expected performance, to be measured: the camera's depth is most reliable to about 3–4 m, so safe indoor speeds of roughly 3–6 m/s are realistic. In a home that is close to what a skilled FPV pilot flies.

## 3. Hardware (human task, bought in two stages)

Estimates only; verify every price and spec before ordering.

**Stage A, bought at Phase 2 (no flight risk): the sensor rig**

| Part | Approx. cost |
| --- | --- |
| RealSense D435i depth camera | €360 |
| Onboard computer: 8-core ARM board, 8 GB RAM, cooling, storage | €120 |

**Stage B, bought at Phase 7: the airframe**

| Part | Approx. cost |
| --- | --- |
| 3.5–4 inch frame with full prop guards, motors, props | €100 |
| Flight controller and ESC | €100 |
| Downward optical-flow and range sensor | €26 |
| Radio handset and receiver (independent safety link) | €80 |
| 3 batteries and charger | €95 |
| 5 V power regulator, Wi-Fi adapter, cables, mounts | €40 |
| Spares | €70 |

Total about €990. Target takeoff weight about 500 g with a thrust-to-weight ratio of at least 4, ideally 5, so it has the power for sharp maneuvers. Pick motors and battery for this before anything else on the airframe. Choosing the Orbbec Gemini 335 instead frees roughly €100 for spares, at the cost of a lower frame rate.

## 4. Onboard software architecture

| Layer | Rate | Input | Output |
| --- | --- | --- | --- |
| Flight controller | 500+ Hz | Thrust and body-rate commands | Motor commands; hover and land failsafes |
| Safety shield | 100 Hz | Local map, velocity, commanded action | Speed cap or override so the drone can always stop in known free space; blocks motion into unseen space |
| Flight policy (learned) | 60–90 Hz | Low-resolution depth image (start at 24 × 32), a coarse ring of ranges from the local map for directions outside the camera's view, velocity, attitude, next two waypoints, gaze target, speed limit | Thrust and body rates, including yaw |
| Odometry | 100–200 Hz | Stereo images, IMU, optical flow, range | Pose and velocity |
| Mapper | 15–30 Hz | Depth and pose | Voxel occupancy map, frontiers, floor plan |
| Exploration policy | 2–5 Hz | Map channels and drone state | Next two waypoints, gaze target, speed limit; classical planner as fallback |
| Reporter | 1–5 Hz | Map | Live 3D map and floor plan streamed to the desktop when a link exists |

Everything that runs on the drone must also run unchanged against the simulator. The drone must complete a mission with no link at all.

Notes for implementers:

- The depth camera's infrared dot pattern disturbs feature tracking. Alternate the projector on and off between frames, or track on a stream without the pattern.
- License rule: the deployed stack uses permissive licenses (Apache, MIT, BSD). Several well-known odometry packages and person detectors are GPL or AGPL. Record every dependency in `LICENSES.yaml` with `commercial_ok: true|false`; if unclear, mark false and report.

## 5. Shared formats (frozen contracts; version every change)

Conventions: SI units; right-handed, z up; all randomness from explicit seeds.

- **Scene:** collision mesh and voxel grid; room polygons, labels, doors and their state; per-surface depth-sensing material (normal, shiny, black, glass, mirror); lighting (lit or dark); spawn point; license flags; generator seed.
- **Drone profile:** one config file with every physical and sensor parameter (mass, inertia, thrust, motor lag, drag, battery, depth camera field of view, noise and frame rate, IMU noise, odometry drift, compute latency, body radius), each with a nominal value and a randomization range. Simulator, randomization, and training read only from it.
- **Report:** voxel map, floor plan grid (0 unknown, 1 free, 2 occupied) with resolution and origin, room polygons, optional person markers, end time.
- **Episode result:** pass or fail, time, coverage, floor-plan IoU, rooms found, collisions, minimum clearance, landed safely, fail reasons; scorer, scene, and profile versions; seed.
- **Flight log:** timestamped depth and camera frames, IMU, pose estimate, commands, policy outputs, shield interventions, map snapshots. Simulator ground truth goes in a separate block that real logs omit.

## 6. Phases

A phase is done only when every gate item passes in CI or in a written report (`reports/phase_N.md`). Numeric targets are initial; measure and report real numbers.

### Phase 0: Foundations

**Build:** repository, containers, CI, deterministic seeding; scene schema and validator; drone profile; scorer with a separate dense training reward; scripted "attack" policies (report instantly, hover forever, leave the map, fake a map) that must all fail.

**Gate:** one-command build from a fresh clone; hand-built three-room scene validates; broken scenes rejected for the right reason; attack policies fail; same seed gives identical results.

### Phase 1: World generator

**Build:** procedural buildings as meshes and voxels, growing in difficulty: single rooms, apartments, building floors with corridors and many units. Realistic dimensions, furniture, door states, glass, mirrors, shiny and black surfaces, lit and dark rooms. Importer that turns a real recorded map into a scene.

**Gate:** 10,000 valid scenes with no human fixes; at least one scene per second; a human finds 50 random floor plans plausible.

### Phase 2: Depth simulation and the handheld rig

**Build:**

- GPU depth simulation by ray casting: the camera's field of view and frame rate, noise that grows with distance, holes on shiny, black, and glass surfaces, thick edges, maximum-range dropout.
- IMU, optical-flow, and odometry-drift simulation from the drone profile.
- Onboard odometry and mapper running on recorded data.
- **Human task:** buy Stage A hardware; walk through real homes and buildings holding the rig at flight height; record logs.

**Gate:**

- Odometry and mapper produce clean maps from handheld logs of at least three real places; loop error reported.
- The depth noise model is fitted to the real logs; simulated depth of an imported real room matches real depth of it (error statistics reported).
- Depth simulation runs for thousands of parallel drones; rate reported.
- Odometry, mapper, and a dummy policy all meet their rates on the onboard computer. If they do not, report before any airframe is bought.

### Phase 3: Fast world, classical baseline, human baseline

**Build:**

- GPU-vectorized voxel world with a forward-facing sensor cone and a point-mass drone with yaw, supporting many drones per scene.
- Classical explorer in the style of FUEL or FALCON that produces a scored report.
- Benchmark harness: held-out scenes, fixed seeds, table of median time, 90th-percentile time, pass rate, collisions.
- Gamepad mode with a first-person view so a human can fly the same scenes; replay viewer for any log.

**Gate:** at least 100,000 steps per second with 4,096 environments; classical baseline passes at least 90% of apartments; human times recorded in the table.

### Phase 4: Flight policy

**Build:**

- DiffAero wrapped with the scenes, drone profile, and a model of the flight controller's rate loop.
- Flight policy trained by backpropagating through the differentiable simulator. Loss terms: closeness to the target velocity toward the waypoint, distance to obstacles, control smoothness, pointing the camera along the direction of travel or at the gaze target, respect for the speed limit.
- Input includes the next two waypoints, so the policy takes racing lines through doorways instead of stopping at each one.
- Keep the depth input low resolution; the reference work found that this helps transfer to reality.
- Randomize every drone profile parameter, compute latency, and sensor noise each episode.
- Safety shield as described in section 4.
- A PPO-trained policy with the same interface, for comparison.

**Gate:**

- Reaches random waypoint pairs in cluttered scenes with at least 99% success across the full randomization range.
- Speed, clearance, and shield-intervention statistics reported; the shield never fails to prevent a collision that was preventable under its assumptions.
- Inference under 5 ms on the onboard computer.

### Phase 4b: Agile maneuver skills

**Build:**

- Skills trained in the dynamics simulator with the same randomization as the flight policy:
  - gap pass: through an opening only slightly wider than the drone
  - hard stop: from full speed to a hover a set distance from a wall
  - slalom around furniture
  - snap turn: 180° reversal in the shortest distance
  - flip and roll
  - precision landing on a marked pad
- Clearance envelope per skill: the volume the drone can occupy while performing it, taken from thousands of randomized simulated runs.
- Skill gate in the safety shield: a skill may start only if its whole envelope lies inside space the map already shows as free. During a flip the camera sees floor and ceiling, so the map's memory is the only protection.
- Attitude through flips comes from the flight controller's gyro. Model odometry dropout during and after flips, and train the recovery to a stable hover.
- These skills stay in the mapping pilot because they are useful during exploration. Freestyle flying (racing lines, chained tricks, the full show) is the job of the acro pilot in Track B (section 7), which reuses the flip work and envelopes from this phase.

**Gate:**

- Each skill succeeds in at least 99% of randomized runs; envelopes reported.
- Odometry recovers after flips within a measured time, in every test.
- Precision landing within 10 cm and hard-stop distance error within 15 cm in simulation.
- The shield refuses every skill whose envelope touches unknown or occupied space (tested with adversarial scenes).

### Phase 5: Exploration policy (key checkpoint)

**Build:**

- Layout predictor: a network that guesses the building beyond the current map (trained on generated buildings), used to estimate how much each candidate viewpoint and gaze direction would reveal.
- Exploration policy: reads map channels (free, occupied, unknown, frontier, predicted layout, visit count), outputs the next two waypoints, a gaze target, and a speed limit. Train by imitating the classical explorer, then PPO in the fast world.
- Automatic curriculum: generate more of the scene types where the policy currently fails.
- Combined system in the dynamics simulator: exploration policy, flight policy, shield.

**Gate:** on held-out buildings, the learned system passes at least as often as the classical explorer with a clearly shorter median time (report the improvement with uncertainty intervals), and is compared against the human gamepad times.

**If the gate fails:** ship the classical explorer with the learned flight policy, report, and continue. The project does not depend on this phase succeeding.

### Phase 6: Full stack in the loop

**Build:**

- The complete onboard stack: odometry, mapper, both policies exported for fast inference, shield, flight controller interface, logger, reporter.
- Desktop ground station: live 3D map and floor plan viewer.
- Software in the loop: the flight controller's own simulator plus the depth simulation feeding the real stack.
- Hardware in the loop: the same test with the stack running on the real onboard computer.

**Gate:** full missions pass in the loop with no simulator ground truth used anywhere (enforced by a test); every layer meets its rate on the onboard computer; end-to-end latency from depth frame to motor command measured and written into the drone profile.

### Phase 7: Build and measure the drone (human task)

**Build (agents):** system identification scripts, camera-to-body calibration tools, failsafe checklist, one-command retraining from a measured drone profile.

**Human task:** buy Stage B hardware; assemble; bench tests; fly by hand with the stack running in **shadow mode** (it computes commands but does not fly).

**Gate:**

- Simulated drone replaying real stick logs follows nearly the same path.
- Shadow-mode commands are sensible throughout manual flights (reviewed in the replay viewer).
- Hover and land failsafes work in every test with the link off.
- Policies retrained on the measured profile still pass Phases 4 and 5.

### Phase 8: Real flight ladder (human task)

Rules for every flight: prop guards, safety pilot with a kill switch on an independent radio, nobody else in the space, eye protection, failsafe tested before each session.

| Step | What flies | Where | Speed cap |
| --- | --- | --- | --- |
| 1 | Autonomous hover and simple moves | Largest empty room available | 0.5 m/s |
| 2 | Flight policy to waypoints around soft obstacles | Same, with foam or cardboard | 1.5 m/s |
| 3 | Flight policy through doorways | Two connected rooms | 2 m/s |
| 4 | Full exploration | The house | 1.5 m/s |
| 5 | Full exploration, speed raised step by step | The house | Up to the shield's limit |
| 6 | Maneuver skills, one at a time, flips last | Largest empty room, over soft flooring | Per skill |
| 7 | Hand over to the acro pilot (Track B, Phase B5) | The house | Up to the shield's limit |

**Build (agents):** tools that replay each real flight through the scorer, compare it with the same place in simulation, and add every flown place and every failure to the scene library.

**Gate:** each step is clean for at least 20 flights before the next.

### Phase 9: Time trial

The drone maps the whole house from the front door and lands. Compare with a human flying the same task, same scorer, at least 10 runs each.

### Phase 10: Extensions (after the core works)

- **People:** pretrained person detector on the camera's colour stream, detections placed in 3D using depth, markers added to the map and scorer. A small thermal camera for dark rooms.
- **Photoreal twin:** Gaussian splat built after the flight from the recorded frames.
- **360° LiDAR version:** with a budget near €2,500, a Livox Mid-360 class sensor sees a whole room from the doorway and supports much higher speeds. The software above is designed so only the sensor layer changes.
- **Multiple drones:** shared map, split exploration, radio relaying.

## 7. Track B: the acro pilot (second model)

A second, separate model whose only job is to look impressive: racing lines, sharp turns, rapid direction changes, flips and rolls. It runs on the same drone and never explores. The mapping pilot (Track A, the phases above) maps the space first; the acro pilot then performs inside that map.

Track B depends on Phases 0–4 (simulator, drone profile, shield) and can be built in parallel with Phases 5 and 6.

### Design

| Aspect | Choice | Why |
| --- | --- | --- |
| Inputs | Velocity, attitude, body rates, the next three virtual gates, the active trick, and distances to obstacles sampled from the known map | The space is already mapped, so the policy flies from its state and the map. During a flip the camera sees nothing useful |
| Outputs | Thrust and body rates at 100 Hz or more | Same interface as Track A |
| Training | Reinforcement learning (PPO) in the dynamics simulator, full drone-profile randomization | State-based racing policies trained this way have beaten champion human pilots; training takes hours on one GPU |
| Course | Generated automatically from the map | Every mapped building becomes a race track |
| Safety | Same shield, plus clearance envelopes that grow with speed and with position uncertainty | Acrobatics only inside space known to be free |

### Phase B1: Course generator

**Build:** from any map (generated scene or real recorded map), place virtual gates at doorways and through open areas to form a closed course; mark trick zones where the free volume is large enough for each trick; a difficulty setting that tightens lines and adds tricks.

**Gate:** valid courses generated for at least 95% of apartments and all larger scenes; every gate and trick zone lies in free space with the required clearance.

### Phase B2: Racing policy

**Build:** policy rewarded for progress along the course and for passing gates, penalized for low clearance, with collisions ending the episode. Curriculum from wide slow courses to tight fast ones. Lap-time table against human gamepad laps on the same courses.

**Gate:** completes at least 99% of laps on held-out courses across the full randomization range; median lap time beats the human gamepad laps.

### Phase B3: Tricks

**Build:**

- Trick set: flip, roll, power loop, split-S, snap 180° turn, fast yaw spin.
- The policy receives the requested trick when it enters a trick zone. Reward: complete the rotation, stay inside the trick's envelope, and exit back on the racing line at speed.
- Optional style input: imitate freestyle flights recorded by a human in gamepad mode.

**Gate:** each trick succeeds in at least 99% of randomized runs and stays inside its envelope; the exit speed and position error after each trick are reported.

### Phase B4: Position tracking under acrobatics

**Build:**

- Model of odometry failure at high rotation rates, fitted later to real logs.
- Short-term dead reckoning from the IMU through each trick.
- Relocalization against the known map by aligning depth frames to it as soon as the camera sees structure again.
- Position-uncertainty estimate fed to the shield, which widens margins or cancels the next trick when uncertainty is high.

**Gate:** in the loop with simulated dropouts, the drone always recovers a position fix after a trick within a measured time, and the shield cancels tricks whenever uncertainty exceeds the clearance available.

### Phase B5: Real flights (human task)

Same safety rules as Phase 8. Start in the largest empty space available, over soft flooring.

| Step | What flies | Speed cap |
| --- | --- | --- |
| 1 | Racing policy on a wide course | 2 m/s |
| 2 | Racing policy, speed raised step by step | Up to the shield's limit |
| 3 | Single tricks in the open, one at a time | Per trick |
| 4 | Full course with tricks | Up to the shield's limit |
| 5 | The house: Track A maps it, Track B races it | Up to the shield's limit |

**Gate:** each step is clean for at least 20 flights before the next.

## 8. Standing rules

- Tests are written with the code; every bug fix adds a regression test.
- Ground truth access is explicit and named; nothing uses it silently.
- Held-out scenes are never used for training or tuning.
- Benchmarks run in CI; a slowdown above 10% must be justified.
- Every checkpoint is stored with its drone profile, scene set, scorer version, and seed.
- When this plan is ambiguous or wrong, stop and report with a proposed fix.
