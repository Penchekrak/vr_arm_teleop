"""Open-loop two-finger cube grasp planning for Aero Hand Open."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation as R

from teleop_backends.camera_calibration import UrdfKinematicTree
from teleop_core.aero_hand_model import (
    AERO_ACTUATION_LOWER_LIMITS_DEG,
    AERO_ACTUATION_UPPER_LIMITS_DEG,
    AeroActuatorGoal,
    AeroThumbJointPose,
    actuator_goal_from_thumb_and_index,
)
from teleop_core.cube_detection import DetectedCube
from teleop_core.robot import RobotCommand
from teleop_core.types import Pose


@dataclass(frozen=True)
class AeroCubePinchConfig:
    """Tunable constants for the first open-loop cube pinch."""

    clearance_m: float = 0.012
    squeeze_margin_m: float = 0.004
    min_index_theta_rad: float = 0.0
    max_index_theta_rad: float = math.radians(65.0)
    pregrasp_thumb: AeroThumbJointPose = field(
        default_factory=lambda: AeroThumbJointPose(
            cmc_abd_deg=45.0,
            cmc_flex_deg=25.0,
            mcp_deg=20.0,
            ip_deg=20.0,
        )
    )
    close_thumb: AeroThumbJointPose = field(
        default_factory=lambda: AeroThumbJointPose(
            cmc_abd_deg=45.0,
            cmc_flex_deg=35.0,
            mcp_deg=35.0,
            ip_deg=35.0,
        )
    )
    base_link: str = "right_base_link"
    index_pad_link: str = "right_index_distal_link"
    thumb_pad_link: str = "right_thumb_distal_link"
    index_pad_local_m: tuple[float, float, float] = (0.0, 0.003512, 0.026776)
    thumb_pad_local_m: tuple[float, float, float] = (0.0, 0.0059474, 0.031823)
    # Convention: world_from_tcp = world_from_pinch @ pinch_from_tcp.
    pinch_from_tcp: tuple[tuple[float, float, float, float], ...] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    actuation_lower_limits_deg: tuple[float, ...] = AERO_ACTUATION_LOWER_LIMITS_DEG
    actuation_upper_limits_deg: tuple[float, ...] = AERO_ACTUATION_UPPER_LIMITS_DEG


@dataclass(frozen=True)
class AeroCubePinchPlan:
    cube_id: str
    pregrasp_wrist_pose: Pose
    close_wrist_pose: Pose
    pregrasp_actuators: AeroActuatorGoal
    close_actuators: AeroActuatorGoal
    pregrasp_index_theta_rad: float
    close_index_theta_rad: float
    pregrasp_aperture_m: float
    close_aperture_m: float

    def as_robot_commands(self) -> tuple[RobotCommand, RobotCommand]:
        """Return the open-loop pregrasp and close commands."""

        return (
            RobotCommand(
                target_wrist_pose=self.pregrasp_wrist_pose,
                target_aero_actuator_degrees=np.asarray(
                    self.pregrasp_actuators.values,
                    dtype=np.float32,
                ),
            ),
            RobotCommand(
                target_wrist_pose=self.close_wrist_pose,
                target_aero_actuator_degrees=np.asarray(
                    self.close_actuators.values,
                    dtype=np.float32,
                ),
            ),
        )


class AeroCubePinchPlanner:
    """Plan a two-finger thumb/index pinch on Aero's actuator manifold."""

    def __init__(
        self,
        kinematic_tree: UrdfKinematicTree,
        config: AeroCubePinchConfig | None = None,
    ) -> None:
        self._tree = kinematic_tree
        self.config = config or AeroCubePinchConfig()

    @classmethod
    def from_urdf(
        cls,
        urdf_path: Path,
        config: AeroCubePinchConfig | None = None,
    ) -> "AeroCubePinchPlanner":
        return cls(UrdfKinematicTree.from_file(Path(urdf_path)), config)

    def plan(self, cube: DetectedCube) -> AeroCubePinchPlan:
        edge = float(cube.edge_m)
        pregrasp_aperture = edge + float(self.config.clearance_m)
        close_aperture = max(0.0, edge - float(self.config.squeeze_margin_m))

        pre_theta = self.solve_index_theta_for_aperture(
            pregrasp_aperture,
            thumb_pose=self.config.pregrasp_thumb,
        )
        close_theta = self.solve_index_theta_for_aperture(
            close_aperture,
            thumb_pose=self.config.close_thumb,
        )
        pre_goal = actuator_goal_from_thumb_and_index(
            self.config.pregrasp_thumb,
            pre_theta,
        ).clamped(
            self.config.actuation_lower_limits_deg,
            self.config.actuation_upper_limits_deg,
        )
        close_goal = actuator_goal_from_thumb_and_index(
            self.config.close_thumb,
            close_theta,
        ).clamped(
            self.config.actuation_lower_limits_deg,
            self.config.actuation_upper_limits_deg,
        )
        pose = self._wrist_pose_for_cube(cube)

        return AeroCubePinchPlan(
            cube_id=str(cube.id),
            pregrasp_wrist_pose=pose,
            close_wrist_pose=pose,
            pregrasp_actuators=pre_goal,
            close_actuators=close_goal,
            pregrasp_index_theta_rad=pre_theta,
            close_index_theta_rad=close_theta,
            pregrasp_aperture_m=self.aperture_for_index_theta(
                pre_theta,
                thumb_pose=self.config.pregrasp_thumb,
            ),
            close_aperture_m=self.aperture_for_index_theta(
                close_theta,
                thumb_pose=self.config.close_thumb,
            ),
        )

    def solve_index_theta_for_aperture(
        self,
        aperture_m: float,
        *,
        thumb_pose: AeroThumbJointPose,
    ) -> float:
        """Solve on the coupled index manifold for the closest aperture."""

        target = float(aperture_m)
        lo = float(self.config.min_index_theta_rad)
        hi = float(self.config.max_index_theta_rad)
        samples = np.linspace(lo, hi, 181)
        apertures = np.asarray(
            [
                self.aperture_for_index_theta(float(theta), thumb_pose=thumb_pose)
                for theta in samples
            ],
            dtype=np.float64,
        )
        best_i = int(np.argmin(np.abs(apertures - target)))

        # Refine around the best sample without assuming global monotonicity.
        left = max(0, best_i - 2)
        right = min(len(samples) - 1, best_i + 2)
        fine = np.linspace(float(samples[left]), float(samples[right]), 101)
        fine_apertures = np.asarray(
            [
                self.aperture_for_index_theta(float(theta), thumb_pose=thumb_pose)
                for theta in fine
            ],
            dtype=np.float64,
        )
        return float(fine[int(np.argmin(np.abs(fine_apertures - target)))])

    def aperture_for_index_theta(
        self,
        theta_rad: float,
        *,
        thumb_pose: AeroThumbJointPose,
    ) -> float:
        joints = self._joint_positions(float(theta_rad), thumb_pose)
        index = self._pad_position(
            self.config.index_pad_link,
            self.config.index_pad_local_m,
            joints,
        )
        thumb = self._pad_position(
            self.config.thumb_pad_link,
            self.config.thumb_pad_local_m,
            joints,
        )
        return float(np.linalg.norm(index - thumb))

    def _pad_position(
        self,
        link_name: str,
        local_pos_m: Iterable[float],
        joint_positions: dict[str, float],
    ) -> np.ndarray:
        link_from_base = self._tree.transform(
            self.config.base_link,
            link_name,
            joint_positions,
        )
        local = np.ones(4, dtype=np.float64)
        local[:3] = np.asarray(tuple(local_pos_m), dtype=np.float64)
        return (link_from_base @ local)[:3]

    def _joint_positions(
        self,
        index_theta_rad: float,
        thumb_pose: AeroThumbJointPose,
    ) -> dict[str, float]:
        theta = float(np.clip(
            index_theta_rad,
            self.config.min_index_theta_rad,
            self.config.max_index_theta_rad,
        ))
        return {
            "right_thumb_cmc_abd": math.radians(thumb_pose.cmc_abd_deg),
            "right_thumb_cmc_flex": math.radians(thumb_pose.cmc_flex_deg),
            "right_thumb_mcp": math.radians(thumb_pose.mcp_deg),
            "right_thumb_ip": math.radians(thumb_pose.ip_deg),
            "right_index_mcp_flex": theta,
            "right_index_pip": theta,
            "right_index_dip": theta,
        }

    def _wrist_pose_for_cube(self, cube: DetectedCube) -> Pose:
        world_from_pinch = np.eye(4, dtype=np.float64)
        world_from_pinch[:3, :3] = R.from_euler("z", float(cube.yaw_rad)).as_matrix()
        world_from_pinch[:3, 3] = np.asarray(cube.center_m, dtype=np.float64).reshape(3)

        pinch_from_tcp = np.asarray(
            self.config.pinch_from_tcp,
            dtype=np.float64,
        ).reshape(4, 4)
        world_from_tcp = world_from_pinch @ pinch_from_tcp
        return Pose(
            position=world_from_tcp[:3, 3].copy(),
            orientation=R.from_matrix(world_from_tcp[:3, :3]).as_quat(),
            frame="world",
        )
