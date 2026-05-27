import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from teleop_backends.robot.aero_grasp import AeroCubePinchPlanner
from teleop_core.cube_detection import DetectedCube


REPO_ROOT = Path(__file__).resolve().parents[1]
URDF = REPO_ROOT / "urdf_rc5_right_hand" / "urdf_with_simple_collisions.urdf"


def _cube(yaw: float = 0.0, edge: float = 0.038) -> DetectedCube:
    return DetectedCube(
        id="cube-0",
        center_m=(0.2, 0.1, edge / 2.0),
        yaw_rad=yaw,
        edge_m=edge,
        points=120,
        mean_abs_sdf_m=0.001,
        confidence=0.9,
    )


def test_cube_pinch_planner_returns_pregrasp_and_close_actuator_goals():
    planner = AeroCubePinchPlanner.from_urdf(URDF)

    plan = planner.plan(_cube())

    assert len(plan.pregrasp_actuators.values) == 7
    assert len(plan.close_actuators.values) == 7
    assert plan.close_actuators.values[3] > plan.pregrasp_actuators.values[3]
    assert plan.pregrasp_actuators.values[4:] == pytest.approx([0.0, 0.0, 0.0])
    assert plan.close_actuators.values[4:] == pytest.approx([0.0, 0.0, 0.0])
    assert all(
        lo <= value <= hi
        for value, lo, hi in zip(
            plan.close_actuators.values,
            planner.config.actuation_lower_limits_deg,
            planner.config.actuation_upper_limits_deg,
        )
    )


def test_cube_yaw_rotates_wrist_pose_without_changing_edge_based_actuators():
    planner = AeroCubePinchPlanner.from_urdf(URDF)

    plan0 = planner.plan(_cube(yaw=0.0))
    plan45 = planner.plan(_cube(yaw=math.radians(45.0)))

    assert plan45.pregrasp_actuators.values == pytest.approx(
        plan0.pregrasp_actuators.values
    )
    assert plan45.close_actuators.values == pytest.approx(plan0.close_actuators.values)

    rot0 = R.from_quat(plan0.pregrasp_wrist_pose.orientation)
    rot45 = R.from_quat(plan45.pregrasp_wrist_pose.orientation)
    delta = rot45 * rot0.inv()

    assert delta.as_euler("zyx", degrees=True)[0] == pytest.approx(45.0, abs=1e-4)


def test_index_aperture_lookup_crosses_requested_cube_width():
    planner = AeroCubePinchPlanner.from_urdf(URDF)

    open_aperture = planner.aperture_for_index_theta(
        0.0,
        thumb_pose=planner.config.close_thumb,
    )
    closed_aperture = planner.aperture_for_index_theta(
        planner.config.max_index_theta_rad,
        thumb_pose=planner.config.close_thumb,
    )
    target = _cube().edge_m - planner.config.squeeze_margin_m

    assert open_aperture > target
    assert closed_aperture < target

    theta = planner.solve_index_theta_for_aperture(
        target,
        thumb_pose=planner.config.close_thumb,
    )

    assert 0.0 < theta < math.radians(90.0)
    assert planner.aperture_for_index_theta(
        theta,
        thumb_pose=planner.config.close_thumb,
    ) == pytest.approx(target, abs=0.002)


def test_pinch_wrist_pose_tracks_cube_center_position():
    planner = AeroCubePinchPlanner.from_urdf(URDF)

    plan = planner.plan(_cube())

    assert np.allclose(
        plan.pregrasp_wrist_pose.position,
        [0.2, 0.1, 0.019],
        atol=1e-9,
    )


def test_pinch_plan_exports_pregrasp_and_close_robot_commands():
    planner = AeroCubePinchPlanner.from_urdf(URDF)
    plan = planner.plan(_cube())

    pregrasp_cmd, close_cmd = plan.as_robot_commands()

    assert pregrasp_cmd.target_wrist_pose == plan.pregrasp_wrist_pose
    assert close_cmd.target_wrist_pose == plan.close_wrist_pose
    assert pregrasp_cmd.target_aero_actuator_degrees.tolist() == pytest.approx(
        plan.pregrasp_actuators.values
    )
    assert close_cmd.target_aero_actuator_degrees.tolist() == pytest.approx(
        plan.close_actuators.values
    )
