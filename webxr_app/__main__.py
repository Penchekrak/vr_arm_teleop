"""CLI entry point. Picks backends, builds the TeleopServer, runs it.

This is the *only* file that imports from both ``teleop_core`` and
``teleop_backends``. Everything else stays one-directional.

Usage:
    python -m webxr_app --pc-backend mock --robot-backend pybullet
    python -m webxr_app --pc-backend hardware --robot-backend aero \\
        --cameras config/hardware_cameras.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from teleop_core import (
    Pose, ServerConfig, SafetyConfig, TeleopServer, Workspace,
)
from teleop_backends.pointcloud import (
    HardwarePointCloudSource, MockPointCloudSource, MultiRealSenseSource,
    PybulletPointCloudSource,
)


def _default_full_urdf() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "urdf_rc5_right_hand"
        / "urdf_with_simple_collisions.urdf"
    )


def _default_floating_urdf() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "urdf_rc5_right_hand"
        / "robot_one_joint.urdf"
    )


def _make_pc_source(name: str, args: argparse.Namespace):
    """Resolve --pc-backend into a PointCloudSource instance."""
    if name == "mock":
        return MockPointCloudSource()
    if name == "hardware":
        if args.cameras is None:
            raise SystemExit("--pc-backend hardware requires --cameras <path>")
        return HardwarePointCloudSource.from_config_file(args.cameras)
    if name == "realsense":
        if args.cameras is None:
            raise SystemExit("--pc-backend realsense requires --cameras <path>")
        return MultiRealSenseSource.from_config_file(args.cameras)
    if name == "pybullet":
        raise SystemExit("pc-backend 'pybullet' not implemented (punchlist 4+)")
    raise SystemExit(f"unknown pc-backend: {name!r}")


def _make_robot_driver(name: str, args: argparse.Namespace):
    """Resolve --robot-backend into a RobotDriver instance."""
    if name == "noop":
        from teleop_backends.robot import NoopRobotDriver

        return NoopRobotDriver(home=Pose.identity(frame="world"))
    if name == "pybullet":
        from teleop_backends.robot import PybulletRobotDriver

        urdf = args.urdf
        if urdf is None:
            # Canonical full robot model: arm + prehand D405 + right hand,
            # with simplified collisions for stable PyBullet simulation.
            urdf = _default_full_urdf()
        home_q = _parse_home_joints(args)
        return PybulletRobotDriver(
            urdf_path=urdf, gui=args.pybullet_gui, home_joint_angles=home_q,
        )
    if name == "floating":
        from teleop_backends.robot import FloatingWristDriver

        urdf = args.urdf
        if urdf is None:
            urdf = _default_floating_urdf()
        return FloatingWristDriver(urdf_path=urdf, gui=args.pybullet_gui)
    if name == "aero":
        from teleop_backends.robot import AeroArmDriver
        return AeroArmDriver(
            arm_ip=args.arm_ip,
            aero_port=args.aero_port,
            arm_speed=args.arm_speed,
            arm_accel=args.arm_accel,
        )
    raise SystemExit(f"unknown robot-backend: {name!r}")


def _parse_home_joints(args: argparse.Namespace) -> tuple[float, ...] | None:
    if args.home_joints is None:
        return None
    home_q = tuple(float(s) for s in args.home_joints.split(","))
    if len(home_q) != 6:
        raise SystemExit("--home-joints needs 6 comma-separated radians")
    return home_q


def _make_workspace(args: argparse.Namespace, home: Pose | None) -> Workspace:
    """Read workspace box from CLI / config file, or derive it from the
    robot's home pose so the operator can reach forward from where the
    arm starts but not above it or below the robot base."""
    if args.workspace is not None:
        data = json.loads(args.workspace.read_text())
        return Workspace.from_dict(data)
    if home is not None:
        hx, hy, hz = (float(v) for v in home.position)
        # Forward reach is +X relative to the robot base. Y is symmetric
        # around the home Y so the operator has some lateral slack. Z is
        # bounded above by the home height (no going up) and below by 0
        # (robot origin == approximately the table / mounting surface).
        return Workspace(
            min_corner=np.array([hx, hy - 0.3, 0.0], dtype=np.float32),
            max_corner=np.array([hx + 0.4, hy + 0.3, hz], dtype=np.float32),
        )
    # No home available (e.g. noop driver). Fall back to the old default.
    return Workspace(
        min_corner=np.array([-0.5, 0.6, -1.1], dtype=np.float32),
        max_corner=np.array([0.5, 1.4, -0.1], dtype=np.float32),
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pc-backend",
                    choices=("mock", "hardware", "realsense", "pybullet"),
                    default="mock")
    ap.add_argument("--robot-backend",
                    choices=("noop", "pybullet", "floating", "aero"),
                    default="pybullet")

    # Backend-specific knobs:
    ap.add_argument("--cameras", type=Path, default=None,
                    help="camera config JSON for --pc-backend hardware/realsense")
    ap.add_argument("--arm-ip", default="10.10.10.10",
                    help="RC5 IP address for --robot-backend aero (default 10.10.10.10)")
    ap.add_argument("--aero-port", default=None,
                    help="Serial port for --robot-backend aero (auto-detect if omitted)")
    ap.add_argument("--arm-speed", type=float, default=0.1,
                    help="RC5 waypoint speed m/s (default 0.1)")
    ap.add_argument("--arm-accel", type=float, default=0.1,
                    help="RC5 waypoint acceleration m/s² (default 0.1)")
    ap.add_argument("--urdf", type=Path, default=None,
                    help="Robot URDF for the pybullet driver")
    ap.add_argument("--pybullet-gui", action="store_true",
                    help="Show the pybullet GUI window (DIRECT mode otherwise)")
    ap.add_argument("--home-joints", default=None,
                    help="6 comma-separated joint angles (radians) for the "
                         "pybullet home pose, e.g. '0,-2.0,1.8,-1.4,1.57,0'")

    # Networking + TLS:
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dashboard-port", type=int, default=8001)
    ap.add_argument("--cert", type=Path, default=None)
    ap.add_argument("--key", type=Path, default=None)

    # Workspace:
    ap.add_argument("--workspace", type=Path, default=None,
                    help="workspace.json with min/max corners in world frame")

    return ap.parse_args()


async def main() -> None:
    args = _parse_args()
    urdf_for_dashboard = args.urdf or (
        _default_floating_urdf()
        if args.robot_backend == "floating"
        else _default_full_urdf()
    )
    pc_source = _make_pc_source(args.pc_backend, args)
    robot = _make_robot_driver(args.robot_backend, args)
    # Start the robot first so its home_pose is available for the workspace
    # derivation. The server's run() also calls start(), but the driver is
    # idempotent on a second call.
    await robot.start()
    home = None
    try:
        home = robot.home_pose
    except Exception:
        pass
    workspace = _make_workspace(args, home)
    print(f"[teleop] workspace: min={workspace.min_corner.tolist()} "
          f"max={workspace.max_corner.tolist()}")
    server = TeleopServer(
        point_cloud_source=pc_source,
        robot_driver=robot,
        workspace=workspace,
        config=ServerConfig(
            port=args.port,
            dashboard_port=args.dashboard_port,
            cert=args.cert,
            key=args.key,
            urdf_path=urdf_for_dashboard,
            robot_assets_root=urdf_for_dashboard.parent,
        ),
        safety_config=SafetyConfig(),
        dashboard_grasp_planner=_make_dashboard_grasp_planner(urdf_for_dashboard),
        dashboard_plan_simulator=_make_dashboard_plan_simulator(urdf_for_dashboard, args),
    )
    await server.run()


def _make_dashboard_grasp_planner(urdf_path: Path):
    from teleop_backends.robot.aero_grasp import AeroCubePinchPlanner

    return AeroCubePinchPlanner.from_urdf(urdf_path)


def _make_dashboard_plan_simulator(urdf_path: Path, args: argparse.Namespace):
    from teleop_backends.robot import PybulletPlanSimulator

    return PybulletPlanSimulator(
        urdf_path=urdf_path,
        home_joint_angles=_parse_home_joints(args),
        gui=False,
    )


if __name__ == "__main__":
    asyncio.run(main())
