"""Robot driver interface.

Hides the difference between "drive a sim arm in pybullet" and "drive a
real arm over the network / SDK". The orchestrator only ever talks to
:class:`RobotDriver`; the rest of the code never imports pybullet or any
hardware SDK directly.

A driver is responsible for:
- Solving inverse kinematics (or otherwise) to reach the commanded
  Cartesian wrist target.
- Returning the **actual** wrist pose so the safety layer can compute
  command-vs-actual lag.
- Reporting its own home pose so the tracker can anchor against it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import Pose


@dataclass(frozen=True)
class RobotState:
    """Snapshot of the robot at a point in time. Used for safety + HUD."""

    wrist_pose: Pose          # current end-effector pose, world frame
    joint_angles: np.ndarray  # all joint angles (radians), for logging
    finger_curls: np.ndarray  # (5,) normalized 0..1; for HUD echo
    timestamp: float          # time.monotonic() seconds
    joint_names: tuple[str, ...] = ()

    @property
    def named_joint_angles(self) -> dict[str, float]:
        """Return URDF-name keyed joint angles when a driver provides names."""
        if len(self.joint_names) != int(self.joint_angles.shape[0]):
            return {}
        return {
            name: float(angle)
            for name, angle in zip(self.joint_names, self.joint_angles)
        }


@dataclass(frozen=True)
class RobotCommand:
    """A target for the driver to track. None fields = no change."""

    target_wrist_pose: Optional[Pose] = None
    target_finger_curls: Optional[np.ndarray] = None  # (5,) normalized
    target_thumb_abduction: Optional[float] = None    # normalized 0..1
    target_aero_actuator_degrees: Optional[np.ndarray] = None  # (7,) Aero actuator degrees
    timestamp: float = 0.0


class RobotDriver(abc.ABC):
    """Bidirectional interface to a robot.

    Implementations: pybullet sim, real Aero hand (+ later, a real arm),
    a no-op logger for dry-run development, a recorded-trajectory player
    for regression tests.
    """

    @abc.abstractmethod
    async def start(self) -> None:
        """Connect / spin up sim / open serial. Idempotent."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Release resources. Safe to call multiple times."""

    @abc.abstractmethod
    async def send(self, cmd: RobotCommand) -> None:
        """Issue a target. Implementation chooses how to track (IK, async)."""

    @abc.abstractmethod
    async def get_state(self) -> RobotState:
        """Return the latest known state. Cheap to call every frame."""

    @property
    @abc.abstractmethod
    def home_pose(self) -> Pose:
        """Neutral wrist pose. Used as the robot-side anchor on engage."""
