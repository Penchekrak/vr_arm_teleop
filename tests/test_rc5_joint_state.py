import asyncio
import time

import numpy as np
import pytest

from teleop_core.robot import RobotCommand
from teleop_core.types import Pose
from teleop_backends.robot.aero_arm import AeroArmDriver
from teleop_backends.robot.rc5_state import (
    RC5_ARM_JOINT_NAMES,
    RC5JointStateReader,
    read_rc5_named_joint_angles,
)


class FakeJointMotion:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def get_actual_position(self, *, units):
        self.calls.append({"units": units})
        return self.values


class FakeMotion:
    def __init__(self, values):
        self.joint = FakeJointMotion(values)


class FakeRobot:
    def __init__(self, values):
        self.motion = FakeMotion(values)


def test_read_rc5_named_joint_angles_requests_radians_and_maps_urdf_names():
    robot = FakeRobot([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])

    named = read_rc5_named_joint_angles(robot)

    assert robot.motion.joint.calls == [{"units": "rad"}]
    assert named == {
        "joint0": 0.1,
        "joint1": -0.2,
        "joint2": 0.3,
        "joint3": -0.4,
        "joint4": 0.5,
        "joint5": -0.6,
    }


def test_read_rc5_named_joint_angles_rejects_wrong_joint_count():
    robot = FakeRobot([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="expected 6 RC5 joint angles"):
        read_rc5_named_joint_angles(robot)


def test_read_rc5_named_joint_angles_rejects_non_finite_values():
    robot = FakeRobot([0.1, -0.2, float("nan"), -0.4, 0.5, -0.6])

    with pytest.raises(ValueError, match="non-finite"):
        read_rc5_named_joint_angles(robot)


def test_rc5_joint_state_reader_connects_read_only_and_disconnects():
    calls = []

    class FakeRobotApi:
        def __init__(self, *, ip, read_only, show_std_traceback):
            calls.append(("init", ip, read_only, show_std_traceback))
            self._robot = FakeRobot([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            self.motion = self._robot.motion

        def disconnect(self):
            calls.append(("disconnect",))

    reader = RC5JointStateReader(
        arm_ip="10.10.10.20",
        robot_api_factory=FakeRobotApi,
    )

    assert reader.read_joint_state() == {
        "joint0": 0.0,
        "joint1": 0.1,
        "joint2": 0.2,
        "joint3": 0.3,
        "joint4": 0.4,
        "joint5": 0.5,
    }
    assert calls == [
        ("init", "10.10.10.20", True, True),
        ("disconnect",),
    ]


def test_aero_arm_driver_get_state_populates_named_rc5_joints(monkeypatch):
    driver = AeroArmDriver()
    driver._robot = object()
    driver._home_pose = Pose.identity(frame="world")

    def fake_tcp_pose(_robot):
        return Pose(
            position=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            frame="world",
        )

    def fake_named_joints(_robot):
        return {name: float(i) for i, name in enumerate(RC5_ARM_JOINT_NAMES)}

    monkeypatch.setattr(
        "teleop_backends.robot.aero_arm._read_rc5_tcp_pose",
        fake_tcp_pose,
    )
    monkeypatch.setattr(
        "teleop_backends.robot.aero_arm.read_rc5_named_joint_angles",
        fake_named_joints,
    )
    monkeypatch.setattr(time, "monotonic", lambda: 123.0)

    state = asyncio.run(driver.get_state())

    assert state.joint_names == RC5_ARM_JOINT_NAMES
    assert np.allclose(state.joint_angles, [0, 1, 2, 3, 4, 5])
    assert state.timestamp == 123.0


def test_aero_arm_driver_sends_direct_actuator_goal_without_compact_joint_mapping():
    class FakeHand:
        def __init__(self):
            self.actuations = []
            self.joint_positions = []

        def set_actuations(self, values):
            self.actuations.append(list(values))

        def set_joint_positions(self, values):
            self.joint_positions.append(list(values))

    driver = AeroArmDriver()
    driver._robot = object()
    driver._hand = FakeHand()
    driver._joint_lower = tuple([0.0] * 16)
    driver._joint_upper = tuple([90.0] * 16)
    driver._actuation_lower = tuple([0.0] * 7)
    driver._actuation_upper = tuple([300.0] * 7)

    goal = np.array([45.0, 20.0, 30.0, 120.0, 0.0, 0.0, 0.0], dtype=np.float32)

    asyncio.run(driver.send(RobotCommand(target_aero_actuator_degrees=goal)))

    assert driver._hand.actuations == [pytest.approx(goal.tolist())]
    assert driver._hand.joint_positions == []
