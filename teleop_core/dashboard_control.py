"""Dashboard-owned command gate for planned robot motion.

The web dashboard is the authority for hardware motion. Selecting a cube only
creates a pending plan and runs the simulator; the real robot is commanded
only when that exact pending plan is approved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .cube_detection import CubeDetectionResult, DetectedCube
from .robot import RobotCommand, RobotDriver, RobotState
from .workspace import Workspace


class GraspPlanner(Protocol):
    def plan(self, cube: DetectedCube):
        """Return an object with ``as_robot_commands()``."""


class PlanSimulator(Protocol):
    async def simulate(
        self,
        commands: Sequence[RobotCommand],
        start_state: RobotState | None = None,
    ) -> "PlanSimulationResult":
        """Run commands on a mock copy of the robot."""


@dataclass(frozen=True)
class PlanSimulationStep:
    name: str
    target_position_m: tuple[float, float, float]
    actual_position_m: tuple[float, float, float]
    position_error_m: float
    reached: bool

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "target_position_m": [float(v) for v in self.target_position_m],
            "actual_position_m": [float(v) for v in self.actual_position_m],
            "position_error_m": float(self.position_error_m),
            "reached": bool(self.reached),
        }


@dataclass(frozen=True)
class PlanSimulationResult:
    ok: bool
    message: str
    steps: tuple[PlanSimulationStep, ...] = ()

    def as_dict(self) -> dict:
        return {
            "ok": bool(self.ok),
            "message": self.message,
            "steps": [step.as_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class PendingDashboardPlan:
    plan_id: str
    cube_id: str
    cube_sequence: int
    created_at: float
    commands: tuple[RobotCommand, ...]
    simulation: PlanSimulationResult

    def as_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "cube_id": self.cube_id,
            "cube_sequence": int(self.cube_sequence),
            "created_at": float(self.created_at),
            "command_count": len(self.commands),
            "commands": [_command_dict(cmd) for cmd in self.commands],
            "simulation": self.simulation.as_dict(),
        }


class DashboardControlError(Exception):
    """User-action error surfaced as a dashboard API response."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = int(status)
        self.reason = str(reason)

    def as_dict(self) -> dict:
        return {"error": "dashboard_control", "reason": self.reason}


class DashboardControlService:
    """Stateful dashboard command coordinator."""

    def __init__(
        self,
        *,
        robot_driver: RobotDriver,
        workspace_getter: Callable[[], Workspace],
        workspace_setter: Callable[[Workspace], None],
        cube_provider: Callable[[], CubeDetectionResult],
        grasp_planner: GraspPlanner | None = None,
        plan_simulator: PlanSimulator | None = None,
    ) -> None:
        self._robot = robot_driver
        self._workspace_getter = workspace_getter
        self._workspace_setter = workspace_setter
        self._cube_provider = cube_provider
        self._grasp_planner = grasp_planner
        self._plan_simulator = plan_simulator
        self._enabled = False
        self._pending_plan: PendingDashboardPlan | None = None
        self._executing = False
        self._last_error: str | None = None
        self._next_plan_number = 1

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def enable(self) -> dict:
        self._enabled = True
        self._last_error = None
        return self.snapshot()

    async def disable(self) -> dict:
        self._enabled = False
        self._pending_plan = None
        self._executing = False
        return self.snapshot()

    async def update_workspace(self, payload: dict) -> dict:
        workspace = Workspace.from_dict(payload)
        self._workspace_setter(workspace)
        self._pending_plan = None
        self._last_error = None
        return {
            "workspace": workspace.as_dict(),
            "control": self.snapshot(),
        }

    async def plan_grasp(self, cube_id: str) -> PendingDashboardPlan:
        self._require_enabled()
        if self._grasp_planner is None:
            raise self._fail(503, "grasp planner is not configured")
        if self._plan_simulator is None:
            raise self._fail(503, "plan simulator is not configured")

        detection = self._cube_provider()
        cube = _find_cube(detection, cube_id)
        if cube is None:
            raise self._fail(404, f"cube {cube_id!r} is no longer detected")

        planned = self._grasp_planner.plan(cube)
        commands = tuple(planned.as_robot_commands())
        if not commands:
            raise self._fail(422, "planner produced no robot commands")
        self._validate_workspace(commands)

        start_state = await self._robot.get_state()
        simulation = await self._plan_simulator.simulate(commands, start_state=start_state)
        if not simulation.ok:
            self._pending_plan = None
            raise self._fail(422, f"simulation rejected plan: {simulation.message}")

        pending = PendingDashboardPlan(
            plan_id=self._new_plan_id(),
            cube_id=str(cube.id),
            cube_sequence=int(detection.sequence),
            created_at=time.monotonic(),
            commands=commands,
            simulation=simulation,
        )
        self._pending_plan = pending
        self._last_error = None
        return pending

    async def approve(self, plan_id: str) -> dict:
        self._require_enabled()
        pending = self._pending_plan
        if pending is None:
            raise self._fail(409, "no simulated plan is waiting for approval")
        if pending.plan_id != plan_id:
            raise self._fail(409, "approval does not match the pending plan")

        self._executing = True
        try:
            for cmd in pending.commands:
                await self._robot.send(cmd)
        finally:
            self._executing = False
            self._pending_plan = None
        self._last_error = None
        return {
            "executed": True,
            "plan_id": pending.plan_id,
            "command_count": len(pending.commands),
        }

    def snapshot(self) -> dict:
        return {
            "enabled": self._enabled,
            "executing": self._executing,
            "pending_plan": (
                None
                if self._pending_plan is None
                else self._pending_plan.as_dict()
            ),
            "last_error": self._last_error,
        }

    def _require_enabled(self) -> None:
        if not self._enabled:
            raise self._fail(409, "dashboard control mode is disabled")

    def _validate_workspace(self, commands: tuple[RobotCommand, ...]) -> None:
        workspace = self._workspace_getter()
        for index, cmd in enumerate(commands):
            pose = cmd.target_wrist_pose
            if pose is None:
                continue
            if not workspace.contains(pose.position):
                raise self._fail(
                    422,
                    f"command {index + 1} target is outside the workspace",
                )

    def _new_plan_id(self) -> str:
        plan_id = f"plan-{self._next_plan_number}"
        self._next_plan_number += 1
        return plan_id

    def _fail(self, status: int, reason: str) -> DashboardControlError:
        self._last_error = reason
        return DashboardControlError(status, reason)


def _find_cube(
    detection: CubeDetectionResult,
    cube_id: str,
) -> DetectedCube | None:
    for cube in detection.cubes:
        if cube.id == cube_id:
            return cube
    return None


def _command_dict(cmd: RobotCommand) -> dict:
    pose = cmd.target_wrist_pose
    return {
        "target_wrist_pose": (
            None
            if pose is None
            else {
                "position": [float(v) for v in pose.position],
                "orientation": [float(v) for v in pose.orientation],
                "frame": pose.frame,
            }
        ),
        "has_finger_curls": cmd.target_finger_curls is not None,
        "has_thumb_abduction": cmd.target_thumb_abduction is not None,
        "has_aero_actuators": cmd.target_aero_actuator_degrees is not None,
        "timestamp": float(cmd.timestamp),
    }
