"""Cartesian wrist tracking with anchoring.

Engage: capture (user_wrist_world, robot_wrist) as the anchor pair.
Per frame: target_robot_wrist = anchor_robot_wrist + (user_wrist_world
                                                     - anchor_user_wrist).
Disengage: forget the anchor; robot freezes at its last target.

The tracker is allowed to clamp targets against the :class:`Workspace`;
that's where the "robot stops at the border" behaviour comes from.

Orientation: a delta quaternion from the user wrist anchor is applied
to the robot wrist anchor in the world frame. Assumes play_space and
world are roughly co-oriented (the v1 frame-alignment cheat described
in the README); a proper recenter step will land in phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import Pose
from .workspace import Workspace


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def _quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion (its conjugate, within numerical tol)."""
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float64)


# Change-of-basis from helmet frame (x=right, y=up, z=back) to robot frame
# (x=right, y=forward, z=up). Equivalent to a +90 deg rotation about X:
# (x_h, y_h, z_h) -> (x_h, -z_h, y_h).
_HELMET_TO_ROBOT_Q = np.array(
    [np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)], dtype=np.float64,
)
_ROBOT_TO_HELMET_Q = _quat_inv(_HELMET_TO_ROBOT_Q)


def _helmet_to_robot_pose(pose: Pose) -> Pose:
    """Re-express a pose given in helmet axes into the robot frame."""
    p = np.asarray(pose.position, dtype=np.float64)
    new_pos = np.array([p[0], -p[2], p[1]], dtype=np.float64)
    q = np.asarray(pose.orientation, dtype=np.float64)
    # Conjugate the orientation by the change-of-basis quaternion.
    new_q = _quat_mul(_HELMET_TO_ROBOT_Q, _quat_mul(q, _ROBOT_TO_HELMET_Q))
    new_q /= max(np.linalg.norm(new_q), 1e-9)
    return Pose(position=new_pos, orientation=new_q, frame=pose.frame)


@dataclass(frozen=True)
class WristAnchor:
    """The pair of poses captured when the operator engages tracking."""
    user_wrist: Pose
    robot_wrist: Pose
    timestamp: float


@dataclass(frozen=True)
class TrackingResult:
    """Per-frame output of the tracker."""

    target: Pose                # robot wrist target (post-clamp)
    raw_target: Pose            # before workspace clamping; for safety/HUD
    in_workspace: bool          # True iff raw target was inside the box
    engaged: bool


class CartesianTracker:
    """Maps user-wrist motion onto robot-wrist commands.

    Single instance per session. Holds the (possibly null) anchor; the
    server feeds it user poses each frame and gets back commands.
    """

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._anchor: Optional[WristAnchor] = None
        self._last_target: Optional[Pose] = None

    @property
    def is_engaged(self) -> bool:
        return self._anchor is not None

    def set_workspace(self, workspace: Workspace) -> None:
        """Replace the clamp volume after a dashboard workspace edit."""
        self._workspace = workspace

    def engage(self, user_wrist: Pose, robot_wrist: Pose, t: float) -> None:
        """Start tracking. Subsequent updates produce non-None targets."""
        self._anchor = WristAnchor(
            user_wrist=_helmet_to_robot_pose(user_wrist),
            robot_wrist=robot_wrist,
            timestamp=t,
        )
        # Initialise the frozen-at-disengage memory to the engage point
        # so a falling edge immediately after engage keeps the robot still
        # rather than reverting to something stale.
        self._last_target = robot_wrist

    def disengage(self) -> None:
        """Stop tracking. The robot should *freeze* at its current pose."""
        self._anchor = None
        # _last_target stays put -- the command loop will keep resending
        # it so the driver holds position instead of drifting back to home.

    def update(self, user_wrist: Pose, t: float) -> TrackingResult:
        """Compute the robot target for the current user wrist."""
        if self._anchor is None:
            # Not engaged: hand back the frozen pose if we have one, else
            # an identity pose. Either way, the caller checks `engaged`
            # before sending anything to the robot.
            frozen = self._last_target if self._last_target is not None \
                else Pose.identity(frame="world")
            return TrackingResult(
                target=frozen, raw_target=frozen,
                in_workspace=True, engaged=False,
            )

        # Re-express the incoming helmet-frame pose in the robot frame so
        # the delta below is in the same coordinates as the robot anchor.
        user_wrist_r = _helmet_to_robot_pose(user_wrist)

        # Cartesian delta from the user's wrist anchor, applied to the
        # robot's wrist anchor. Position is a simple subtraction; orientation
        # is a world-frame delta quaternion (user_now * inv(user_anchor))
        # left-multiplied onto the robot's anchor orientation.
        delta = np.asarray(user_wrist_r.position, dtype=np.float64) \
            - np.asarray(self._anchor.user_wrist.position, dtype=np.float64)
        raw_pos = np.asarray(self._anchor.robot_wrist.position, dtype=np.float64) + delta

        user_q = np.asarray(user_wrist_r.orientation, dtype=np.float64)
        user_anchor_q = np.asarray(self._anchor.user_wrist.orientation, dtype=np.float64)
        robot_anchor_q = np.asarray(self._anchor.robot_wrist.orientation, dtype=np.float64)
        delta_q = _quat_mul(user_q, _quat_inv(user_anchor_q))
        raw_q = _quat_mul(delta_q, robot_anchor_q)
        raw_q /= max(np.linalg.norm(raw_q), 1e-9)  # renormalise against drift

        raw_target = Pose(
            position=raw_pos,
            orientation=raw_q,
            frame=self._anchor.robot_wrist.frame,
        )

        clamped_pos, was_outside = self._workspace.clamp(raw_pos)
        target = Pose(
            position=clamped_pos,
            orientation=raw_q,
            frame=raw_target.frame,
        )
        self._last_target = target
        return TrackingResult(
            target=target, raw_target=raw_target,
            in_workspace=not was_outside, engaged=True,
        )
