"""PyBullet mirror used by the dashboard approval gate."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from teleop_core.dashboard_control import (
    PlanSimulationResult,
    PlanSimulationStep,
)
from teleop_core.robot import RobotCommand, RobotState

from .pybullet_driver import PybulletRobotDriver


class PybulletPlanSimulator:
    """Run a planned command sequence on a separate DIRECT pybullet robot."""

    def __init__(
        self,
        *,
        urdf_path: Path,
        home_joint_angles: tuple[float, ...] | None = None,
        settle_seconds: float = 0.7,
        position_tolerance_m: float = 0.035,
        gui: bool = False,
    ) -> None:
        self._urdf_path = Path(urdf_path)
        self._home_joint_angles = home_joint_angles
        self._settle_seconds = float(settle_seconds)
        self._position_tolerance_m = float(position_tolerance_m)
        self._gui = bool(gui)

    async def simulate(
        self,
        commands: Sequence[RobotCommand],
        start_state: RobotState | None = None,
    ) -> PlanSimulationResult:
        driver = PybulletRobotDriver(
            urdf_path=self._urdf_path,
            home_joint_angles=self._home_joint_angles,
            gui=self._gui,
        )
        steps: list[PlanSimulationStep] = []
        ok = True
        try:
            await driver.start()
            for index, command in enumerate(commands):
                await driver.send(command)
                await asyncio.sleep(max(0.0, self._settle_seconds))
                state = await driver.get_state()
                target = command.target_wrist_pose
                if target is None:
                    actual = tuple(float(v) for v in state.wrist_pose.position)
                    step = PlanSimulationStep(
                        name=f"command-{index + 1}",
                        target_position_m=actual,
                        actual_position_m=actual,
                        position_error_m=0.0,
                        reached=True,
                    )
                else:
                    target_pos = np.asarray(target.position, dtype=np.float64)
                    actual_pos = np.asarray(state.wrist_pose.position, dtype=np.float64)
                    err = float(np.linalg.norm(actual_pos - target_pos))
                    reached = bool(math.isfinite(err) and err <= self._position_tolerance_m)
                    ok = ok and reached
                    step = PlanSimulationStep(
                        name=f"command-{index + 1}",
                        target_position_m=tuple(float(v) for v in target_pos),
                        actual_position_m=tuple(float(v) for v in actual_pos),
                        position_error_m=err,
                        reached=reached,
                    )
                steps.append(step)
        except Exception as exc:
            return PlanSimulationResult(
                ok=False,
                message=f"pybullet preview failed: {exc!r}",
                steps=tuple(steps),
            )
        finally:
            await driver.stop()

        return PlanSimulationResult(
            ok=ok,
            message="simulation reached all targets" if ok else "simulation missed target tolerance",
            steps=tuple(steps),
        )
