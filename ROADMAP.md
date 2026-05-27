# Roadmap

Ordered punchlist of remaining work. Each item lists the files
you'll touch and a rough effort. Pick one numbered item, read the
relevant interface files, implement it.

## 1. Safety monitor ⏱ 1h

Workspace-exit feedback and dashboard workspace clamping are wired, but
the dashboard approval gate still needs the server-side safety loop for
live execution faults.

- `teleop_core/safety.SafetyMonitor.step` — lag detection
  (commanded vs actual wrist) and stale-state detection.
- `TeleopServer._safety_loop` — call `step`, broadcast `SafetyMsg`,
  pause the tracker when severity escalates.
- Remaining `TeleopServer` state-transition hooks
  (`_enter_finger_cal`, `_finish_finger_cal`, `_engage_tracking`,
  `_disengage_tracking`, `_fault`) — currently stubs raising
  `NotImplementedError`.

Done when: yanking the robot driver offline pops a `fault` overlay
that the user has to acknowledge.

## 2. Dashboard-first hardware validation ⏱ hardware session

The dashboard now owns workspace adjustment, cube selection, PyBullet
preview, and approval-only execution. Validate this on the physical
robot before treating it as production-safe.

- Confirm the oriented workspace transform controls match the gripper
  frame and block out-of-box targets.
- Tune PyBullet preview tolerance against RC5 actual final pose error.
- Add stale-plan invalidation for cube pose drift, robot motion, and
  workspace edits between simulation and approval.
- Add a visible emergency freeze/hold action that maps to the real arm
  SDK, not only dashboard control disable.

Done when: selecting a cube from the dashboard simulates the Aero pinch,
requires explicit approval, and executes the same motion on hardware
without any VR trigger input.

## 3. Hardware point-cloud calibration ✅ implemented, needs hardware validation

Hardware capture/fusion plumbing exists for mixed RealSense and ZED 2i
configs. Continuous camera-to-robot calibration now opens the configured
cameras, trusts the arm-mounted camera FK as the anchor, optimizes
non-anchor extrinsics from overlapping ChArUco detections, shows the
provisional world-frame cloud in the dashboard, and autosaves stable
solutions back into the hardware camera config.

- Validate the workflow with the physical D405 + D435i camera set.
- Tune board size, inlier thresholds, and stability window on hardware.

Done when: with the configured cameras on the workspace, the fused
cloud aligns with the robot/workspace frame in AR.

## 4. Pybullet point-cloud source ⏱ 2h

Useful for developing the pipeline without real cameras pointed at
something interesting.

- `teleop_backends/pointcloud/pybullet_render.PybulletPointCloudSource`
  — render a depth image from one or more virtual viewpoints inside
  the same pybullet sim used by the robot driver, convert to a cloud
  in world frame.

## 5. Phase 2 frame alignment ⏱ 1h

Replace the "fixed offset in local-floor" cheat with a one-time
recenter step:

- New `RecenterMsg` in the wire protocol.
- Operator stands at a known position relative to the robot, presses
  a button to capture the play_space → world transform.
- Server persists the transform and applies it to the point cloud and
  workspace renderings.

## 6. Real arm driver — *needs hardware validation*

`AeroArmDriver` contains the RC5/Aero SDK wiring, but still needs a
hardware validation session:

- Confirm controller state recovery in `start()` and before commands.
- Tune RC5 waypoint speed/acceleration for dashboard-approved plans.
- Verify `get_state()` timestamps stay fresh enough for safety checks.
- Validate Aero actuator-space commands against the real hand limits.

`TeleopServer` does not change — same interface in, different
hardware out.

---

# Contributing — agent-facing guidance

If you are an AI agent picking this up, the rules of engagement:

1. **Pick one numbered item.** Don't try to do several at once. The
   interfaces let you commit progress without breaking the rest of
   the project.
2. **Implement against the interface, never against a concrete
   class.** If you're tempted to `import pybullet` from inside
   `teleop_core`, stop — add a method to the relevant ABC instead.
3. **Honor the wire format.** If you add a new control message,
   define the dataclass in `teleop_core/messages.py` *and* extend the
   frontend's `comms.js` switch. Do both in the same change.
4. **Don't bypass the dependency direction.** No imports from
   `teleop_core` into `teleop_backends`. No imports from anywhere
   else into `webxr_app`.
5. **No mutable globals in `teleop_core`.** All state lives on
   instance attributes of `TeleopServer`.
6. **Async-first.** All I/O is awaitable. Anything that has to block
   (`rs.pipeline.wait_for_frames`, `pybullet.stepSimulation`) goes
   inside `asyncio.to_thread(...)`.
7. **Match existing port code.** When you port logic from
   `../vr_tendon_arm_teleop`, keep the numerics identical so the two
   projects produce the same calibrations.
8. **Don't add a new top-level dependency without updating
   `requirements.txt`** and noting which backend needs it. Core has
   only `aiohttp` + `numpy`.
9. **Comment why, not what.** The `what` is read in the code; the
   `why` is what an agent six months from now needs.
