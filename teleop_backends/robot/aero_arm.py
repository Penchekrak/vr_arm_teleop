"""Real-hardware driver: RC5 arm + Aero Hand fingers.

Translates the RobotCommand interface (Pose quaternion + normalised finger
curls) into:
  - RC5 Cartesian waypoints via RobotApi (blocking SDK → asyncio.to_thread)
  - Aero Hand 7-joint compact form via AeroHand SDK

RC5 orientation convention: rotation vector in degrees [rx, ry, rz].
This matches teleop_oculus_v1_by_controllers.py; scipy.Rotation converts
between that and the quaternion stored in Pose.

Aero Hand 7-joint layout (from teleop_fingers_aero.py):
    slot 0  thumb_cmc_abd      ← target_thumb_abduction
    slot 1  thumb_cmc_flex     ← target_finger_curls[0]  (thumb)
    slot 2  thumb_mcp (+ip)    ← target_finger_curls[0]  (thumb)
    slot 3  index_mcp          ← target_finger_curls[1]
    slot 4  middle_mcp         ← target_finger_curls[2]
    slot 5  ring_mcp           ← target_finger_curls[3]
    slot 6  pinky_mcp          ← target_finger_curls[4]

Values are interpolated between the hardware lower/upper joint limits
(degrees) read at start(), using the indices 0,1,2,4,7,10,13 into the
16-joint limit array that the SDK exposes.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from teleop_core.robot import RobotCommand, RobotDriver, RobotState
from teleop_core.aero_hand_model import clamp_actuator_degrees
from teleop_core.types import Pose
from .rc5_state import (
    load_rc5_robot_api,
    read_rc5_named_joint_angles,
)


# RC5 SDK path — override with RC5_API_PATH env var if the repo lives
# somewhere else on the development machine.
_DEFAULT_RC5_API_PATH = "/home/echerepanov/spiridonov/rc5_python_api"

# Indices into the AeroHand 16-joint limit arrays for the 7-joint compact form.
# Matches the mapping in teleop_fingers_aero.py / AeroHand.convert_seven_joints_to_sixteen.
_AERO_SLOT_IDX = (0, 1, 2, 4, 7, 10, 13)

# Keep fingers away from the hard mechanical limits during teleop.
_CURL_GAIN = 0.95
_ABD_GAIN  = 1.0

# RC5 canonical home (from teleop_oculus_v1_by_controllers.py)
_HOME_POS_M   = (0.250, 0.175, 0.400)
_HOME_ROT_DEG = (-90.0,   0.0,   0.0)   # rotation vector, degrees


# ── coordinate-conversion helpers ────────────────────────────────────────────

def _rotvec_deg_to_quat(rotvec_deg: np.ndarray) -> np.ndarray:
    """Rotation-vector in degrees → quaternion (x, y, z, w)."""
    return R.from_rotvec(np.radians(rotvec_deg)).as_quat()


def _quat_to_rotvec_deg(quat_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) → rotation-vector in degrees."""
    return np.degrees(R.from_quat(quat_xyzw).as_rotvec())


def _read_rc5_tcp_pose(robot) -> Pose | None:
    cart = robot.motion.linear.get_actual_position(orientation_units="deg")
    if cart is None:
        return None
    pos = np.asarray(cart[:3], dtype=np.float64)
    orn = _rotvec_deg_to_quat(np.asarray(cart[3:], dtype=np.float64))
    return Pose(position=pos, orientation=orn, frame="world")


# ── Aero Hand mapping ─────────────────────────────────────────────────────────

def _curls_to_aero7(
    curls:   np.ndarray,           # (5,) normalised 0..1 [thumb, idx, mid, ring, little]
    abd_norm: float,               # normalised 0..1 thumb CMC abduction
    lower:   tuple[float, ...],    # 16-joint lower limits, degrees
    upper:   tuple[float, ...],    # 16-joint upper limits, degrees
) -> list[float]:
    """Map normalised VR signals to the Aero Hand 7-joint compact form (degrees)."""

    def lerp(slot_idx: int, t: float) -> float:
        lo, hi = lower[slot_idx], upper[slot_idx]
        return lo + float(np.clip(t, 0.0, 1.0)) * (hi - lo)

    c = np.asarray(curls, dtype=np.float32)
    return [
        lerp(_AERO_SLOT_IDX[0], float(abd_norm)  * _ABD_GAIN),
        lerp(_AERO_SLOT_IDX[1], float(c[0])       * _CURL_GAIN),
        lerp(_AERO_SLOT_IDX[2], float(c[0])       * _CURL_GAIN),
        lerp(_AERO_SLOT_IDX[3], float(c[1])       * _CURL_GAIN),
        lerp(_AERO_SLOT_IDX[4], float(c[2])       * _CURL_GAIN),
        lerp(_AERO_SLOT_IDX[5], float(c[3])       * _CURL_GAIN),
        lerp(_AERO_SLOT_IDX[6], float(c[4])       * _CURL_GAIN),
    ]


# ── Driver ────────────────────────────────────────────────────────────────────

class AeroArmDriver(RobotDriver):
    """RC5 arm + Aero Hand fingers on real hardware.

    All blocking SDK calls are dispatched via asyncio.to_thread so the
    event loop remains free. A threading.Lock serialises concurrent calls
    within each SDK (RC5 for arm, AeroHand for fingers) in case the server
    issues overlapping get_state / send requests.
    """

    def __init__(
        self,
        arm_ip:         str            = "10.10.10.10",
        aero_port:      Optional[str]  = None,
        arm_speed:      float          = 3.0,
        arm_accel:      float          = 3.0,
        rc5_api_path:   str            = _DEFAULT_RC5_API_PATH,
    ) -> None:
        self._arm_ip       = arm_ip
        self._aero_port    = aero_port
        self._arm_speed    = arm_speed
        self._arm_accel    = arm_accel
        self._rc5_api_path = os.environ.get("RC5_API_PATH", rc5_api_path)

        self._robot = None      # RobotApi, set in start()
        self._hand  = None      # AeroHand, set in start()

        self._arm_lock  = threading.Lock()
        self._hand_lock = threading.Lock()

        self._joint_lower: Optional[tuple[float, ...]] = None
        self._joint_upper: Optional[tuple[float, ...]] = None
        self._actuation_lower: Optional[tuple[float, ...]] = None
        self._actuation_upper: Optional[tuple[float, ...]] = None

        self._home_pose:  Optional[Pose] = None
        self._last_curls: np.ndarray     = np.zeros(5, dtype=np.float32)
        self._last_aero_actuators: Optional[np.ndarray] = None

    # --- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        self._counter = 0
        if self._robot is not None:
            return  # idempotent

        def _connect() -> None:
            # ── RC5 ──────────────────────────────────────────────────────────
            RobotApi = load_rc5_robot_api(self._rc5_api_path)
            from API.source.models.classes.enum_classes.state_classes import (  # noqa: PLC0415
                InComingControllerState as Ics,
                InComingSafetyStatus    as Iss,
            )

            robot = RobotApi(self._arm_ip, show_std_traceback=True)
            # Clear any latched fault before enabling the controller.
            cs_init = robot.controller_state.get()
            print(f"[arm] startup controller_state={cs_init}", flush=True)
            if (
                robot.safety_status.get()    == Iss.fault.name
                or cs_init == Ics.failure.name
            ):
                robot.controller_state.set("off")
            robot.controller_state.set("run", await_sec=120)
            print(f"[arm] after set('run'): controller_state={robot.controller_state.get()}", flush=True)
            self._robot = robot

            # ── Aero Hand ────────────────────────────────────────────────────
            from aero_open_sdk.aero_hand import AeroHand  # noqa: PLC0415

            hand = AeroHand(port=self._aero_port) if self._aero_port else AeroHand()
            self._joint_lower = hand.joint_lower_limits
            self._joint_upper = hand.joint_upper_limits
            self._actuation_lower = hand.actuation_lower_limits
            self._actuation_upper = hand.actuation_upper_limits
            self._hand = hand

            # ── Home pose from actual RC5 position ───────────────────────────
            pose = _read_rc5_tcp_pose(robot)
            if pose is not None:
                pos = pose.position
                orn = pose.orientation
            else:
                # Robot not responding at start — use the canonical home pose.
                pos = np.asarray(_HOME_POS_M,   dtype=np.float64)
                orn = _rotvec_deg_to_quat(np.asarray(_HOME_ROT_DEG, dtype=np.float64))

            self._home_pose = Pose(position=pos, orientation=orn, frame="world")

        await asyncio.to_thread(_connect)

    async def stop(self) -> None:
        def _disconnect() -> None:
            if self._robot is not None:
                try:
                    with self._arm_lock:
                        self._robot.motion.mode.set("hold")
                except Exception as exc:
                    print(f"[AeroArmDriver] RC5 hold on stop: {exc!r}")
                self._robot = None

            if self._hand is not None:
                try:
                    with self._hand_lock:
                        if self._joint_lower and self._joint_upper:
                            # Move to open/safe pose: fingers straight, thumb spread.
                            open7 = _curls_to_aero7(
                                np.zeros(5, dtype=np.float32), 1.0,
                                self._joint_lower, self._joint_upper,
                            )
                        else:
                            open7 = [0.0] * 7
                        self._hand.set_joint_positions(open7)
                        time.sleep(0.1)
                        self._hand.close()
                except Exception as exc:
                    print(f"[AeroArmDriver] AeroHand close on stop: {exc!r}")
                self._hand = None

        await asyncio.to_thread(_disconnect)

    # --- command + state -----------------------------------------------------

    async def send(self, cmd: RobotCommand) -> None:
        if self._robot is None or self._hand is None:
            raise RuntimeError("AeroArmDriver.send called before start()")

        def _send_arm() -> None:
            if cmd.target_wrist_pose is None:
                return
            pose = cmd.target_wrist_pose
            pos = tuple(float(v) for v in pose.position)
            rot = tuple(
                float(v) for v in
                _quat_to_rotvec_deg(np.asarray(pose.orientation, dtype=np.float64))
            )
            with self._arm_lock:
                # Re-enable the controller if it dropped out of "run" state
                # (e.g. inactivity timeout between startup and first command).
                cs = self._robot.controller_state.get()
                if cs != "run":
                    print(f"[arm] controller in '{cs}', re-enabling run...", flush=True)
                    self._robot.controller_state.set("run", await_sec=30)
                
                if self._counter % 10 == 0:
                    self._robot.motion.linear.add_new_waypoint(
                        pos + rot,
                        speed=self._arm_speed,
                        accel=self._arm_accel,
                    )
                self._counter += 1
                # Only transition to "move" when not already executing:
                # re-sending "move" while already moving blocks the thread
                # for up to 5 s if the confirmation times out.
                if self._robot.motion.mode.get() != "move":
                    self._robot.motion.mode.set("move")

        def _send_hand() -> None:
            if (
                cmd.target_finger_curls is None
                and cmd.target_thumb_abduction is None
                and cmd.target_aero_actuator_degrees is None
            ):
                return
            if cmd.target_aero_actuator_degrees is not None:
                actuators = np.asarray(cmd.target_aero_actuator_degrees, dtype=np.float32).reshape(-1)
                if actuators.shape[0] != 7:
                    raise ValueError("target_aero_actuator_degrees must have length 7")
                if self._actuation_lower is not None and self._actuation_upper is not None:
                    values = clamp_actuator_degrees(
                        actuators,
                        self._actuation_lower,
                        self._actuation_upper,
                    )
                else:
                    values = tuple(float(v) for v in actuators)
                with self._hand_lock:
                    self._hand.set_actuations(list(values))
                self._last_aero_actuators = np.asarray(values, dtype=np.float32)
                return

            c = (
                np.asarray(cmd.target_finger_curls, dtype=np.float32)
                if cmd.target_finger_curls is not None
                else self._last_curls
            )
            if c.shape[0] != 5:
                raise ValueError("target_finger_curls must have length 5")
            a = float(np.clip(cmd.target_thumb_abduction, 0.0, 1.0)) \
                if cmd.target_thumb_abduction is not None else 0.0
            joints7 = _curls_to_aero7(c, a, self._joint_lower, self._joint_upper)
            with self._hand_lock:
                self._hand.set_joint_positions(joints7)
            if cmd.target_finger_curls is not None:
                self._last_curls = c.copy()

        # RC5 and AeroHand are independent buses — dispatch both in parallel.
        await asyncio.gather(
            asyncio.to_thread(_send_arm),
            asyncio.to_thread(_send_hand),
        )

    async def get_state(self) -> RobotState:
        if self._robot is None:
            raise RuntimeError("AeroArmDriver.get_state called before start()")

        def _read() -> RobotState:
            with self._arm_lock:
                pose = _read_rc5_tcp_pose(self._robot)
                named_joints = read_rc5_named_joint_angles(self._robot)
            if pose is not None:
                wrist_pose = pose
            else:
                # Robot temporarily unresponsive — echo the home pose so the
                # safety monitor doesn't false-positive on stale state.
                assert self._home_pose is not None
                wrist_pose = self._home_pose
            joint_names = tuple(named_joints)
            joint_angles = np.asarray(
                [named_joints[name] for name in joint_names],
                dtype=np.float32,
            )

            return RobotState(
                wrist_pose=wrist_pose,
                joint_angles=joint_angles,
                finger_curls=self._last_curls.copy(),
                timestamp=time.monotonic(),
                joint_names=joint_names,
            )

        return await asyncio.to_thread(_read)

    @property
    def home_pose(self) -> Pose:
        if self._home_pose is None:
            raise RuntimeError("home_pose available only after start()")
        return self._home_pose
