import asyncio
import json
import time

import numpy as np
import pytest

from teleop_core.cube_detection import CubeDetectionResult, DetectedCube
from teleop_core.dashboard_control import (
    DashboardControlError,
    DashboardControlService,
    PlanSimulationResult,
    PlanSimulationStep,
)
from teleop_core.robot import RobotCommand, RobotState
from teleop_core.server import ServerConfig, TeleopServer
from teleop_core.types import Pose
from teleop_core.workspace import Workspace


def _pose(x=0.2, y=0.1, z=0.15):
    return Pose(
        position=np.array([x, y, z], dtype=np.float64),
        orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        frame="world",
    )


def _cube_result(sequence=7):
    return CubeDetectionResult(
        sequence=sequence,
        timestamp=time.monotonic(),
        cubes=(
            DetectedCube(
                id="cube-0",
                center_m=(0.2, 0.1, 0.019),
                yaw_rad=0.0,
                edge_m=0.038,
                points=120,
                mean_abs_sdf_m=0.001,
                confidence=0.9,
            ),
        ),
    )


class RecordingRobot:
    def __init__(self):
        self.sent = []

    async def start(self):
        pass

    async def stop(self):
        pass

    async def send(self, cmd):
        self.sent.append(cmd)

    async def get_state(self):
        return RobotState(
            wrist_pose=_pose(0.1, 0.0, 0.2),
            joint_angles=np.zeros(0, dtype=np.float32),
            finger_curls=np.zeros(5, dtype=np.float32),
            timestamp=time.monotonic(),
        )

    @property
    def home_pose(self):
        return _pose(0.1, 0.0, 0.2)


class EmptyPointCloud:
    async def start(self):
        pass

    async def stop(self):
        pass

    async def grab(self):
        return None


class FakePlan:
    def __init__(self):
        self.commands = (
            RobotCommand(target_wrist_pose=_pose(), timestamp=1.0),
            RobotCommand(target_wrist_pose=_pose(z=0.12), timestamp=2.0),
        )

    def as_robot_commands(self):
        return self.commands


class FakePlanner:
    def __init__(self):
        self.cubes = []

    def plan(self, cube):
        self.cubes.append(cube)
        return FakePlan()


class FakeSimulator:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = 0

    async def simulate(self, commands, start_state=None):
        self.calls += 1
        return PlanSimulationResult(
            ok=self.ok,
            message="ok" if self.ok else "unreachable",
            steps=(
                PlanSimulationStep(
                    name="pregrasp",
                    target_position_m=(0.2, 0.1, 0.15),
                    actual_position_m=(0.2, 0.1, 0.15),
                    position_error_m=0.0,
                    reached=self.ok,
                ),
            ),
        )


def _workspace():
    return Workspace(
        min_corner=np.array([0.0, -0.2, 0.0], dtype=np.float32),
        max_corner=np.array([0.5, 0.3, 0.4], dtype=np.float32),
    )


def _service(robot=None, simulator=None):
    workspace_box = _workspace()
    current_workspace = {"value": workspace_box}

    def set_workspace(value):
        current_workspace["value"] = value

    service = DashboardControlService(
        robot_driver=robot or RecordingRobot(),
        workspace_getter=lambda: current_workspace["value"],
        workspace_setter=set_workspace,
        cube_provider=_cube_result,
        grasp_planner=FakePlanner(),
        plan_simulator=simulator or FakeSimulator(),
    )
    return service, current_workspace


def test_dashboard_plan_requires_control_mode_and_does_not_move_robot_before_approval():
    robot = RecordingRobot()
    service, _ = _service(robot=robot)

    async def run():
        with pytest.raises(DashboardControlError) as exc:
            await service.plan_grasp("cube-0")
        assert exc.value.status == 409

        await service.enable()
        pending = await service.plan_grasp("cube-0")

        assert pending.cube_id == "cube-0"
        assert pending.simulation.ok is True
        assert robot.sent == []

    asyncio.run(run())


def test_dashboard_approval_executes_only_the_matching_pending_plan():
    robot = RecordingRobot()
    service, _ = _service(robot=robot)

    async def run():
        await service.enable()
        pending = await service.plan_grasp("cube-0")

        with pytest.raises(DashboardControlError) as exc:
            await service.approve("wrong-plan")
        assert exc.value.status == 409
        assert robot.sent == []

        executed = await service.approve(pending.plan_id)

        assert executed["executed"] is True
        assert [cmd.timestamp for cmd in robot.sent] == [1.0, 2.0]
        assert service.snapshot()["pending_plan"] is None

    asyncio.run(run())


def test_simulation_failure_blocks_approval_and_keeps_robot_stationary():
    robot = RecordingRobot()
    service, _ = _service(robot=robot, simulator=FakeSimulator(ok=False))

    async def run():
        await service.enable()
        with pytest.raises(DashboardControlError) as exc:
            await service.plan_grasp("cube-0")

        assert exc.value.status == 422
        assert "unreachable" in exc.value.reason
        assert service.snapshot()["pending_plan"] is None
        assert robot.sent == []

    asyncio.run(run())


def test_workspace_update_clears_pending_plan_and_replaces_runtime_workspace():
    service, current_workspace = _service()

    async def run():
        await service.enable()
        await service.plan_grasp("cube-0")

        updated = await service.update_workspace({
            "center": [0.2, 0.1, 0.2],
            "half_extents": [0.2, 0.2, 0.15],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "frame": "world",
        })

        assert updated["workspace"]["center"] == pytest.approx([0.2, 0.1, 0.2])
        assert current_workspace["value"].contains(np.array([0.2, 0.1, 0.2]))
        assert service.snapshot()["pending_plan"] is None

    asyncio.run(run())


def test_dashboard_workspace_update_persists_configured_workspace_file(tmp_path):
    workspace_path = tmp_path / "workspace.json"
    workspace_path.write_text(json.dumps(_workspace().as_dict()))
    server = TeleopServer(
        point_cloud_source=EmptyPointCloud(),
        robot_driver=RecordingRobot(),
        workspace=Workspace.from_dict(json.loads(workspace_path.read_text())),
        config=ServerConfig(workspace_path=workspace_path),
    )

    async def run():
        await server._dashboard_control.update_workspace({
            "center": [0.25, 0.05, 0.2],
            "half_extents": [0.2, 0.15, 0.1],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "frame": "world",
        })

    asyncio.run(run())

    written = json.loads(workspace_path.read_text())
    assert written["center"] == pytest.approx([0.25, 0.05, 0.2])
    assert written["half_extents"] == pytest.approx([0.2, 0.15, 0.1])
    assert written["orientation"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert written["min"] == pytest.approx([0.05, -0.1, 0.1])
    assert written["max"] == pytest.approx([0.45, 0.2, 0.3])
