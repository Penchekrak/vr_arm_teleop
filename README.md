# vr_arm_teleop

Dashboard-first control of a robot arm + Aero Hand using fused
multi-camera depth, tabletop cube detection, PyBullet preview, and
optional VR teleoperation. The desktop web dashboard is the source of
truth for setup and hardware motion: it owns workspace editing, cube
selection, simulation preview, approval, and execution. VR/WebXR remains
available as an auxiliary view/input path.

---

## What works today

The end-to-end mock loop runs on a single laptop: launch the server
with a sim robot, open the dashboard, inspect the point cloud/cubes,
adjust the workspace, simulate a grasp plan, then explicitly approve
the movement.

Implemented:

- **`teleop_core/`** — types, oriented gripper workspace box with
  `contains` / `clamp`, dashboard command approval gate, finger
  calibration FSM (6-step prompt flow), `CartesianTracker` (anchor +
  delta math for optional VR), WebSocket message dataclasses + JSON
  codec, and `TeleopServer` orchestration.
- **Point cloud backend** — `MockPointCloudSource` (synthetic
  animated cloud), plus hardware capture/fusion plumbing for mixed
  RealSense and ZED 2i camera configs. Real hardware clouds are gated
  from AR display until camera-to-robot calibration is marked valid.
- **Robot backends** — `NoopRobotDriver` (logs commands),
  `PybulletRobotDriver` (full 6-DoF arm via `calculateInverseKinematics`
  + tendon-coupled hand fingers, URDF in `urdf_rc5_right_hand/`),
  `PybulletPlanSimulator` (separate mock copy for dashboard approval),
  `FloatingWristDriver` (6-DoF floating wrist for development without
  IK).
- **Dashboard control gate** — `/api/control/enable` must be called
  before dashboard motion can be planned. Cube selection creates a
  pending Aero grasp plan and runs it through a PyBullet mirror; only a
  matching `/api/grasp/approve` sends the stored commands to the robot
  driver.
- **WebXR frontend** (`webxr_app/static/`) — Three.js scene + WebXR
  session, per-frame input reader, tracked-hand visualization,
  controller models, point-cloud renderer bound to the binary stream,
  workspace wireframe, head-locked overlay panels (with word-wrap and
  wider geometry so long prompts fit), calibration state machine,
  finger-curl + thumb-abduction math.
- **Ghost hand** — a translucent cyan FBX hand rendered as a
  re-engage / alignment target. Two modes: (a) *starting ghost* drawn
  the first time we hit `ready` — wrist position follows the user,
  orientation + finger curls mirror the robot's current state via a
  one-shot `RobotEchoMsg`; (b) *re-engage ghost* captured on trigger
  release and held in place until the user returns to that pose. The
  engage path gates on a position + orientation tolerance.
- **Thumb abduction end-to-end** — calibration captures min/max raw
  abduction, server applies the remap, `RobotCommand.target_thumb_abduction`
  carries the normalized value, and both pybullet drivers drive
  `right_thumb_cmc_abd` into a tightened band (URDF allows 100°, we
  cap to a more realistic ~6°–63°). Calibration prompt order +
  MIN/MAX bounds were aligned with the joint's natural direction.
- **Client-side workspace-exit warning** — when the user wrist leaves
  the workspace box the warning panel lights up and the workspace
  wireframe flashes red. Server-side `SafetyMonitor` (lag + stale-state
  detection) is still pending.
- **CLI wiring** (`webxr_app/__main__.py`) — picks pc/robot backends,
  derives the workspace box from the robot's home pose (or reads it
  from `--workspace path.json`), starts the HTTPS+WS server.

Not implemented yet — see [ROADMAP.md](ROADMAP.md):

- Camera-to-robot calibration for physically aligned hardware point clouds
- `PybulletPointCloudSource` (depth render from sim as a fake sensor)
- Hardware validation/tuning of `AeroArmDriver` and dashboard-approved plans
- `SafetyMonitor.step` + a couple of state-transition hooks in
  `TeleopServer`

---

## Quick start

```bash
git clone https://github.com/Akhunzianov/vr_arm_teleop.git
cd vr_arm_teleop
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate a self-signed cert so the Quest browser will let you enter
WebXR (WebXR requires HTTPS on LAN, or `localhost`):

```bash
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem \
  -out certs/cert.pem -days 365 -nodes -subj "/CN=$(hostname -I | awk '{print $1}')"
```

Run the smallest end-to-end demo — mock point cloud + sim robot:

```bash
python -m webxr_app \
  --pc-backend mock \
  --robot-backend pybullet \
  --cert certs/cert.pem --key certs/key.pem
```

Open the dashboard at `https://<your-LAN-ip>:8001` (or
`http://localhost:8001` without TLS). The Quest/WebXR page at port
`8000` is optional.

### Dashboard workflow

The server starts the desktop dashboard on `http://<host>:8001` by
default. It renders the configured URDF, fused point cloud, gripper
workspace, detected cubes, and Quest head/right-wrist markers when VR is
connected.

Primary motion flow:

1. Click **Adjust Workspace** to translate, rotate, or resize the
   oriented gripper box with Three.js transform controls; click **Save**
   to replace the runtime safety workspace.
2. Click **Enable Control**. This disables VR trigger-commanded motion.
3. Click a detected cube, then **Simulate Grasp**. The server creates an
   Aero thumb/index pinch plan, validates every wrist target against the
   workspace, and runs the command sequence on a separate PyBullet copy.
4. Inspect the pending simulation result, then click **Approve Move**.
   Only this matching approval sends the stored commands to the robot.

### Wired alternative (no cert needed)

```bash
adb reverse tcp:8000 tcp:8000
python -m webxr_app --pc-backend mock --robot-backend pybullet
# Then in the Quest browser: http://localhost:8000
```

### Optional VR operation

VR teleoperation is auxiliary. It is blocked while dashboard control mode
is enabled.

1. **Finger calibration** — head-locked panel walks through 6 poses;
   press **X** on the left controller to advance each step.
2. **Align to the ghost** — once calibration finishes, a translucent
   cyan hand appears at your wrist with the robot's current
   orientation and finger pose. Twist your wrist + curl your fingers
   to match it before pulling the trigger.
3. **Engage tracking** — hold the **left trigger**. The server
   snapshots `(user_wrist_now, robot_wrist_now)` as the anchor pair.
4. **Track** — move your right hand; the robot wrist follows by
   `target = robot_anchor + (user_wrist_now - user_anchor)`, clamped
   to the workspace box. Finger curls + thumb abduction stream
   through normalised against the calibration. Stepping outside the
   workspace pops a warning and the box flashes red.
5. **Disengage** — release the trigger. A ghost hand stays where you
   let go; you must return roughly to that pose to re-engage.
6. **Quit** — press **Y**.

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--pc-backend` | `mock` | `mock` / `hardware` / `realsense` / `pybullet` (stub) |
| `--robot-backend` | `pybullet` | `noop` / `pybullet` / `floating` / `aero` (stub) |
| `--urdf` | shipped URDF | URDF path for pybullet/floating |
| `--pybullet-gui` | off | Show pybullet GUI window |
| `--home-joints` | derived | 6 comma-separated radians, e.g. `0,-2.0,1.8,-1.4,1.57,0` |
| `--cameras` | – | camera config JSON for `--pc-backend hardware` or `realsense` |
| `--workspace` | derived from home | Workspace JSON; supports legacy `{"min":[...],"max":[...]}` or oriented `{"center":[...],"half_extents":[...],"orientation":[x,y,z,w]}` |
| `--port` | `8000` | HTTP/HTTPS port |
| `--dashboard-port` | `8001` | Desktop control dashboard port |
| `--cert` / `--key` | – | TLS cert + key (required for non-localhost Quest) |

### Hardware camera config

Use `--pc-backend hardware --cameras config/hardware_cameras.json` to
read any enabled mix of RealSense and ZED 2i cameras. The legacy
`--pc-backend realsense` path uses the same config parser but rejects
non-RealSense cameras.

Each camera entry includes `type` (`realsense` or `zed2i`),
`serial`, stream settings, `z_min` / `z_max`, `downsample`,
`calibrated`, and a 4x4 `world_from_camera` matrix. Disabled cameras
are ignored.
For RealSense calibration, keep both depth `width` / `height` and
`color_width` / `color_height` at least `640x480`. The reader queries
each camera's SDK profile list by serial and uses the nearest supported
profile when the exact target is unavailable.

If any enabled camera has `calibrated: false`, the backend still opens
the cameras for diagnostics but returns no point-cloud frames to the AR
client. This is intentional: without camera-to-robot calibration, the
cloud cannot be displayed in a physically meaningful robot/world frame.
Use [config/hardware_cameras.example.json](config/hardware_cameras.example.json)
as the starting shape.

Run the continuous ChArUco calibration tool to solve those transforms:

```bash
python scripts/calibrate_two_cameras_charuco.py \
  --cameras config/hardware_cameras.json \
  --squares-x 7 --squares-y 5 \
  --square-length 0.035 --marker-length 0.026
```

The tool opens every enabled camera in the config, picks the FK-trusted
anchor from `--anchor-camera` or the camera whose `urdf_link` matches
`--camera-link`, serves the same Three.js dashboard on
`--dashboard-port`, and atomically autosaves the input config after all
non-anchor cameras have stable inlier solutions. The first autosave
creates a `*.bak` copy beside the config.

---

## Architecture

Three layers, one direction of dependency:

```
teleop_core/         pure interfaces + types + orchestration
   ↑
teleop_backends/     concrete implementations (cameras, robot)
   ↑
webxr_app/           CLI that wires backends together
                     + JS frontend served from static/
```

**Hard rule:** `teleop_core` never imports from `teleop_backends` or
`webxr_app`. If you're tempted, make the interface richer instead.

### Key interfaces

- **`teleop_core/point_cloud.py`** — `PointCloudSource.grab()` →
  `PointCloudFrame` (XYZ + RGB, world frame).
- **`teleop_core/robot.py`** — `RobotDriver.{start, stop, send,
  get_state, home_pose}`.
- **`teleop_core/messages.py`** — every JSON message that crosses the
  WebSocket. Frontend `modules/comms.js` mirrors this.
- **`teleop_core/dashboard_control.py`** — dashboard control-mode state,
  workspace updates, cube-plan simulation gate, and approval-only
  execution.

Pure logic modules — extend / fix in place, don't subclass:

- `workspace.py` — oriented gripper box, `contains` + `clamp`.
- `calibration.py` — `FingerCalibrationFSM` and the captured record.
- `tracking.py` — `CartesianTracker` (anchor + delta math).
- `safety.py` — `SafetyMonitor` (lag detection, workspace exit — stub).
- `server.py` — `TeleopServer` orchestrator (the four async loops).

### Coordinate frames

| frame | origin | when used |
|---|---|---|
| `world` | robot base | point cloud, robot, workspace |
| `play_space` | where the headset booted (WebXR `local-floor`) | user wrist samples |
| `view` | head, moves with user | head-locked text overlays |

For optional VR teleop, because we drive the robot via *deltas from an
anchor*, no explicit `play_space → world` transform is needed for the
tracking math — the anchor pair captured at trigger-down implicitly
defines it.

For rendering the point cloud and workspace box in the user's view,
v1 cheats: they're placed at a fixed offset in `local-floor` space.
The operator chooses to stand somewhere that makes the geometry feel
right. A proper recenter step is on the roadmap.

---

## Wire protocol

### Control channel (JSON text frames)

Each message has a `type` discriminator matching one of the
dataclasses in `teleop_core/messages.py`. The frontend has a 1:1
mirror in `modules/comms.js` + `modules/state_machine.js`.

Client → Server:
- `HandStateMsg` — streamed at ~30 Hz; wrist position + orientation
  in play_space, finger curls, raw abduction.
- `ButtonMsg` — edge events (X to advance calibration, Y to quit).
- `TriggerMsg` — analog value; edges drive engage/disengage.

Server → Client:
- `PhaseMsg` — `'idle' | 'finger_cal' | 'ready' | 'tracking' | 'fault'`.
- `PromptMsg` — head-locked text panel content + severity.
- `WorkspaceMsg` — one-time announcement of the workspace box bounds.
- `RobotEchoMsg` — live robot pose for HUD.
- `SafetyMsg` — discrete safety event.

### Point cloud channel (binary frames, same WebSocket)

```
[uint32 N][uint32 reserved=0]
[N × int16 x_mm][N × int16 y_mm][N × int16 z_mm]   # world frame, mm
[N × uint8 r][N × uint8 g][N × uint8 b]
```

~9 bytes/point. With WebSocket `permessage-deflate` and a workspace
crop, a 3-camera fused cloud (~10k points) compresses to ~30 KB/frame.
At 15 Hz ≈ 450 KB/s — trivial over Wi-Fi or USB.

---

## File map

```
teleop_core/
  types.py            Pose, Vec3
  point_cloud.py      PointCloudFrame, PointCloudSource, encode_frame
  robot.py            RobotState, RobotCommand, RobotDriver
  workspace.py        Workspace (oriented gripper box)
  dashboard_control.py Dashboard command gate + pending plans
  calibration.py      FingerCalibrationFSM, CalibrationRecord, steps
  tracking.py         CartesianTracker, TrackingResult, WristAnchor
  safety.py           SafetyMonitor (stub), SafetyEvent
  messages.py         WebSocket message dataclasses + JSON codec
  server.py           TeleopServer orchestrator (four async loops)

teleop_backends/
  pointcloud/
    mock.py              synthetic animated cloud
    hardware.py          mixed RealSense/ZED 2i capture + fusion
    realsense_multi.py   legacy RealSense-only wrapper
    pybullet_render.py   pybullet depth-render as a fake sensor [stub]
  robot/
    noop.py                  logs commands, never moves
    pybullet_driver.py       sim 6-DoF arm + tendon hand via pybullet IK
    floating_wrist_driver.py 6-DoF floating wrist (no IK), for dev
    aero_arm.py              real Aero hand + (TBD arm) [stub]

webxr_app/
  __main__.py         CLI + backend wiring (the ONLY file that imports
                      from both teleop_core AND teleop_backends)
  static/
    index.html / style.css / app.js
    modules/
      comms.js             WebSocket client (JSON + binary)
      scene.js             three.js + XR session
      input_reader.js      per-frame WebXR input snapshot
      hand_view.js         tracked-hand visualization
      controller_view.js   controller models
      pointcloud_view.js   THREE.Points bound to the binary stream
      workspace_view.js    workspace box wireframe
      overlay.js           head-locked text panels (prompt + warning)
      state_machine.js     reflects server phase
      hand_math.js         finger curl + thumb abduction (pure)

urdf_rc5_right_hand/  RC5 arm + Aero right hand URDF + meshes
config/               example workspace.json
scripts/              standalone smoke tests
```

---

## What this project deliberately does *not* do

- Joint-by-joint copying of arm pose. Cartesian-only.
- Encode every pixel of the cloud. We crop + quantize, and defer
  delta encoding until measured bandwidth becomes a problem.
- Treat the pybullet driver as the production target. It's for
  development; production runs on the real arm via `AeroArmDriver`
  once that hardware ships.
