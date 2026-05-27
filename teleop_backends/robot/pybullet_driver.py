"""Pybullet-backed robot driver.

Holds a pybullet client, loads the arm+hand URDF, solves IK against the
commanded wrist pose, applies finger curls. Used as the day-to-day
development backend before / instead of the real robot.

The physical hand has 5 tendons (one per finger) but the URDF exposes
~16 finger DOFs; we couple them in software by mapping one normalised
scalar per finger to all of that finger's joints. Thumb opposition
(``cmc_abd``) is held at a fixed neutral angle and is NOT part of the
curl signal (matches the old vr_tendon_arm_teleop project so calibration
records port over without re-tuning).

Everything that touches pybullet goes through ``asyncio.to_thread`` --
pybullet's API is not thread-safe and not async, and we don't want IK
solves to block the event loop. A single background task steps the sim
at ``sim_hz`` so position-control targets are actually tracked.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pybullet as p

from teleop_core.aero_hand_model import (
    AERO_ACTUATION_LOWER_LIMITS_DEG,
    AERO_ACTUATION_UPPER_LIMITS_DEG,
    clamp_actuator_degrees,
)
from teleop_core.robot import RobotCommand, RobotDriver, RobotState
from teleop_core.types import Pose


# Defaults match the rc5_aero_hand URDF copied from vr_tendon_arm_teleop.
# Override via the constructor if a different URDF lands.
_DEFAULT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "joint0", "joint1", "joint2", "joint3", "joint4", "joint5",
)

# Index convention: 0 thumb, 1 index, 2 middle, 3 ring, 4 little
# (matches the order used by FingerCalibrationFSM / RobotCommand.target_finger_curls).
_DEFAULT_FINGER_JOINT_GROUPS: dict[int, tuple[str, ...]] = {
    0: ("right_thumb_cmc_flex", "right_thumb_mcp", "right_thumb_ip"),
    1: ("right_index_mcp_flex", "right_index_pip", "right_index_dip"),
    2: ("right_middle_mcp_flex", "right_middle_pip", "right_middle_dip"),
    3: ("right_ring_mcp_flex", "right_ring_pip", "right_ring_dip"),
    4: ("right_pinky_mcp_flex", "right_pinky_pip", "right_pinky_dip"),
}

_THUMB_OPPOSITION_JOINT = "right_thumb_cmc_abd"
_THUMB_OPPOSITION_DEFAULT_RAD = 0.5
# See FloatingWristDriver for rationale: narrower than the URDF limits.
_THUMB_ABD_MIN_RAD = 0.2
_THUMB_ABD_MAX_RAD = 0.85

_DEFAULT_EE_LINK_NAME = "right_tcp_link"

# Home pose: arm centred (j0=0), forearm raised so the TCP sits roughly
# in front of the base, wrist rolled 90° so the palm faces up. Tweak via
# the home_joint_angles constructor argument when iterating in the GUI.
_ARM_HOME_Q: tuple[float, ...] = (3.14, -2.148, 2.424, -0.264, 1.52, 1.454)


@dataclass(frozen=True)
class _JointInfo:
    index: int
    lower: float
    upper: float


class PybulletRobotDriver(RobotDriver):
    """Pybullet sim robot. IK target -> joint goals -> stepSimulation()."""

    def __init__(
        self,
        urdf_path: Path,
        ee_link_name: str = _DEFAULT_EE_LINK_NAME,
        finger_joint_groups: dict[int, tuple[str, ...]] | None = None,
        arm_joint_names: tuple[str, ...] = _DEFAULT_ARM_JOINT_NAMES,
        home_joint_angles: tuple[float, ...] | None = None,
        client_id: int | None = None,
        gui: bool = False,
        sim_hz: float = 240.0,
    ) -> None:
        self._urdf_path = Path(urdf_path)
        self._ee_link_name = ee_link_name
        self._finger_groups = dict(finger_joint_groups or _DEFAULT_FINGER_JOINT_GROUPS)
        self._arm_joint_names = arm_joint_names
        self._home_q = tuple(home_joint_angles) if home_joint_angles is not None \
            else _ARM_HOME_Q
        self._client_id: Optional[int] = client_id
        self._gui = gui
        self._sim_dt = 1.0 / float(sim_hz)
        self._owns_client = client_id is None

        self._body_id: Optional[int] = None
        self._arm_joints: list[_JointInfo] = []
        self._finger_joints: dict[int, list[_JointInfo]] = {}
        self._thumb_opp: Optional[_JointInfo] = None
        self._ee_link_index: int = -1
        self._movable_joint_indices: list[int] = []

        # Serialises every pybullet call. Held inside the worker thread,
        # but acquired before to_thread dispatches so we never have two
        # threads inside pybullet at once.
        self._lock = asyncio.Lock()
        self._step_task: Optional[asyncio.Task] = None
        self._stopping = False

        self._home_pose: Optional[Pose] = None
        self._last_curls = np.zeros(5, dtype=np.float32)

    # --- lifecycle -----------------------------------------------------

    async def start(self) -> None:
        if self._body_id is not None:
            return  # already started; keep idempotent per the ABC

        def _connect_and_load() -> None:
            if self._owns_client:
                mode = p.GUI if self._gui else p.DIRECT
                self._client_id = p.connect(mode)
            assert self._client_id is not None
            cid = self._client_id

            p.setGravity(0, 0, -9.81, physicsClientId=cid)
            p.setTimeStep(self._sim_dt, physicsClientId=cid)
            p.setRealTimeSimulation(0, physicsClientId=cid)

            self._body_id = p.loadURDF(
                str(self._urdf_path),
                useFixedBase=True,
                physicsClientId=cid,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
            )

            joint_by_name: dict[str, _JointInfo] = {}
            link_index_by_name: dict[str, int] = {}
            n_joints = p.getNumJoints(self._body_id, physicsClientId=cid)
            for i in range(n_joints):
                info = p.getJointInfo(self._body_id, i, physicsClientId=cid)
                name = info[1].decode()
                jtype = info[2]
                lower, upper = info[8], info[9]
                child_link_name = info[12].decode()
                joint_by_name[name] = _JointInfo(index=i, lower=lower, upper=upper)
                link_index_by_name[child_link_name] = i
                if jtype != p.JOINT_FIXED:
                    self._movable_joint_indices.append(i)

            self._arm_joints = [joint_by_name[n] for n in self._arm_joint_names]
            self._finger_joints = {
                f: [joint_by_name[n] for n in names]
                for f, names in self._finger_groups.items()
            }
            self._thumb_opp = joint_by_name[_THUMB_OPPOSITION_JOINT]
            self._ee_link_index = link_index_by_name[self._ee_link_name]

            # Seed home pose: snap arm to a known config, hold thumb
            # opposition, freeze fingers open.
            for joint, q in zip(self._arm_joints, self._home_q):
                p.resetJointState(self._body_id, joint.index, q, physicsClientId=cid)
            p.resetJointState(
                self._body_id,
                self._thumb_opp.index,
                _THUMB_OPPOSITION_DEFAULT_RAD,
                physicsClientId=cid,
            )
            for joint, q in zip(self._arm_joints, self._home_q):
                self._motor_target(joint, float(q))
            self._motor_target(self._thumb_opp, _THUMB_OPPOSITION_DEFAULT_RAD)
            for finger, joints in self._finger_joints.items():
                for j in joints:
                    self._motor_target(j, j.lower)

            # One step so the home pose is reflected in getLinkState before
            # the first IK / get_state call.
            p.stepSimulation(physicsClientId=cid)
            pos, orn = self._read_ee_pose()
            self._home_pose = Pose(
                position=np.asarray(pos, dtype=np.float64),
                orientation=np.asarray(orn, dtype=np.float64),
                frame="world",
            )

        await asyncio.to_thread(_connect_and_load)
        self._stopping = False
        self._step_task = asyncio.create_task(self._step_loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._step_task is not None:
            self._step_task.cancel()
            try:
                await self._step_task
            except (asyncio.CancelledError, Exception):
                pass
            self._step_task = None
        if self._owns_client and self._client_id is not None:
            cid = self._client_id
            await asyncio.to_thread(p.disconnect, physicsClientId=cid)
        self._body_id = None
        self._client_id = None if self._owns_client else self._client_id

    # --- command + state ----------------------------------------------

    async def send(self, cmd: RobotCommand) -> None:
        if self._body_id is None:
            raise RuntimeError("PybulletRobotDriver.send called before start()")
        target_pose = cmd.target_wrist_pose
        target_curls = cmd.target_finger_curls
        target_abd = cmd.target_thumb_abduction
        target_actuators = cmd.target_aero_actuator_degrees

        def _apply() -> None:
            if target_pose is not None:
                arm_q = self._solve_arm_ik(
                    np.asarray(target_pose.position, dtype=np.float64),
                    np.asarray(target_pose.orientation, dtype=np.float64),
                )
                for joint, q in zip(self._arm_joints, arm_q):
                    self._motor_target(joint, float(q))
            if target_curls is not None:
                curls = np.asarray(target_curls, dtype=np.float32).reshape(-1)
                if curls.shape[0] != 5:
                    raise ValueError("target_finger_curls must have length 5")
                for finger, c in enumerate(curls):
                    self._set_finger_curl(finger, float(c))
                self._last_curls = curls.astype(np.float32)
            if target_abd is not None and self._thumb_opp is not None:
                a = float(np.clip(target_abd, 0.0, 1.0))
                lo = max(self._thumb_opp.lower, _THUMB_ABD_MIN_RAD)
                hi = min(self._thumb_opp.upper, _THUMB_ABD_MAX_RAD)
                rad = lo + a * (hi - lo)
                self._motor_target(self._thumb_opp, rad)
            if target_actuators is not None:
                self._set_aero_actuators(target_actuators)

        async with self._lock:
            await asyncio.to_thread(_apply)

    async def get_state(self) -> RobotState:
        if self._body_id is None:
            raise RuntimeError("PybulletRobotDriver.get_state called before start()")

        def _read() -> RobotState:
            pos, orn = self._read_ee_pose()
            states = p.getJointStates(
                self._body_id,
                [j.index for j in self._arm_joints],
                physicsClientId=self._client_id,
            )
            arm_q = np.array([s[0] for s in states], dtype=np.float32)
            return RobotState(
                wrist_pose=Pose(
                    position=np.asarray(pos, dtype=np.float64),
                    orientation=np.asarray(orn, dtype=np.float64),
                    frame="world",
                ),
                joint_angles=arm_q,
                finger_curls=self._last_curls.copy(),
                timestamp=time.monotonic(),
            )

        async with self._lock:
            return await asyncio.to_thread(_read)

    @property
    def home_pose(self) -> Pose:
        if self._home_pose is None:
            raise RuntimeError("home_pose available only after start()")
        return self._home_pose

    # --- internals -----------------------------------------------------

    async def _step_loop(self) -> None:
        """Fixed-rate sim stepping. Keeps position-controlled joints tracking."""
        next_t = time.monotonic()
        while not self._stopping:
            next_t += self._sim_dt
            async with self._lock:
                await asyncio.to_thread(
                    p.stepSimulation, physicsClientId=self._client_id
                )
            sleep = next_t - time.monotonic()
            if sleep > 0:
                await asyncio.sleep(sleep)
            else:
                # We've fallen behind; resync the schedule rather than burn
                # CPU chasing an unreachable deadline.
                next_t = time.monotonic()

    def _read_ee_pose(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        state = p.getLinkState(
            self._body_id, self._ee_link_index, physicsClientId=self._client_id
        )
        return state[4], state[5]

    def _solve_arm_ik(
        self,
        target_position: np.ndarray,
        target_orientation_xyzw: np.ndarray,
    ) -> np.ndarray:
        solution = p.calculateInverseKinematics(
            self._body_id,
            self._ee_link_index,
            tuple(float(x) for x in target_position),
            targetOrientation=tuple(float(x) for x in target_orientation_xyzw),
            physicsClientId=self._client_id,
        )
        # IK returns one entry per movable joint, in declaration order.
        position_in_solution = {
            joint_idx: i for i, joint_idx in enumerate(self._movable_joint_indices)
        }
        return np.array(
            [solution[position_in_solution[j.index]] for j in self._arm_joints],
            dtype=np.float32,
        )

    def _set_finger_curl(self, finger: int, curl: float) -> None:
        c = float(np.clip(curl, 0.0, 1.0))
        joints = self._finger_joints.get(finger)
        if not joints:
            return
        for joint in joints:
            target = joint.lower + c * (joint.upper - joint.lower)
            self._motor_target(joint, target)

    def _set_aero_actuators(self, actuator_degrees) -> None:
        values = np.asarray(
            clamp_actuator_degrees(
                np.asarray(actuator_degrees, dtype=np.float32).reshape(-1),
                AERO_ACTUATION_LOWER_LIMITS_DEG,
                AERO_ACTUATION_UPPER_LIMITS_DEG,
            ),
            dtype=np.float32,
        )
        lo = np.asarray(AERO_ACTUATION_LOWER_LIMITS_DEG, dtype=np.float32)
        hi = np.asarray(AERO_ACTUATION_UPPER_LIMITS_DEG, dtype=np.float32)
        denom = np.maximum(hi - lo, 1e-6)
        normalized = np.clip((values - lo) / denom, 0.0, 1.0)

        if self._thumb_opp is not None:
            self._motor_target(self._thumb_opp, np.deg2rad(float(values[0])))
        curls = np.array(
            [
                max(float(normalized[1]), float(normalized[2])),
                float(normalized[3]),
                float(normalized[4]),
                float(normalized[5]),
                float(normalized[6]),
            ],
            dtype=np.float32,
        )
        for finger, curl in enumerate(curls):
            self._set_finger_curl(finger, float(curl))
        self._last_curls = curls

    def _motor_target(self, joint: _JointInfo, target: float) -> None:
        if joint.upper > joint.lower:
            target = float(np.clip(target, joint.lower, joint.upper))
        p.setJointMotorControl2(
            self._body_id,
            joint.index,
            p.POSITION_CONTROL,
            targetPosition=float(target),
            physicsClientId=self._client_id,
        )
