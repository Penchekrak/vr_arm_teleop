import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

from teleop_core.point_cloud import PointCloudFrame
from teleop_core.cube_detection import CubeDetectionConfig, CubeDetector
from teleop_core.robot import RobotState
from teleop_core.server import _resolve_robot_asset_path
from teleop_core.telemetry import TelemetryHub
from teleop_core.types import Pose
from teleop_core.workspace import Workspace
from scripts.calibrate_two_cameras_charuco import _make_dashboard_app


def test_robot_state_named_joint_angles_pairs_names_with_values():
    state = RobotState(
        wrist_pose=Pose.identity(frame="world"),
        joint_angles=np.array([1.25, -0.5], dtype=np.float32),
        finger_curls=np.zeros(5, dtype=np.float32),
        timestamp=time.monotonic(),
        joint_names=("joint0", "right_index_pip"),
    )

    assert state.named_joint_angles == {
        "joint0": 1.25,
        "right_index_pip": -0.5,
    }


def test_robot_state_named_joint_angles_is_empty_when_lengths_do_not_match():
    state = RobotState(
        wrist_pose=Pose.identity(frame="world"),
        joint_angles=np.array([1.25], dtype=np.float32),
        finger_curls=np.zeros(5, dtype=np.float32),
        timestamp=time.monotonic(),
        joint_names=("joint0", "joint1"),
    )

    assert state.named_joint_angles == {}


class FakeRobot:
    def __init__(self):
        self.state = RobotState(
            wrist_pose=Pose(
                position=np.array([0.1, 0.2, 0.3], dtype=np.float64),
                orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                frame="world",
            ),
            joint_angles=np.array([0.4, -0.2], dtype=np.float32),
            finger_curls=np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            timestamp=123.0,
            joint_names=("joint0", "right_index_pip"),
        )

    async def get_state(self):
        return self.state


class CountingPointCloud:
    def __init__(self):
        self.grab_count = 0
        self.frame = PointCloudFrame(
            points=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
            colors=np.array([[255, 0, 0]], dtype=np.uint8),
            timestamp=234.0,
        )

    async def grab(self):
        self.grab_count += 1
        return self.frame

    def dashboard_camera_feeds(self):
        return [
            {
                "name": "d405",
                "urdf_link": "d405_depth_optical_frame",
                "url": "/api/cameras/d405/color.jpg",
                "width": 640,
                "height": 480,
            }
        ]


def _workspace():
    return Workspace(
        min_corner=np.array([-1.0, -2.0, 0.0], dtype=np.float32),
        max_corner=np.array([1.0, 2.0, 1.0], dtype=np.float32),
    )


def test_dashboard_snapshot_contains_model_workspace_robot_and_unaligned_xr():
    async def run():
        hub = TelemetryHub(
            point_cloud_source=CountingPointCloud(),
            robot_driver=FakeRobot(),
            workspace=_workspace(),
            urdf_url="/robot/robot.urdf",
            urdf_assets_url="/robot/assets/",
            urdf_path="/tmp/source.urdf",
            pointcloud_hz=1000.0,
            robot_hz=1000.0,
            status_hz=1000.0,
        )
        await hub.sample_robot_once()

        snap = hub.snapshot()

        assert snap["type"] == "snapshot"
        assert snap["model"]["urdf_url"] == "/robot/robot.urdf"
        assert snap["model"]["urdf_assets_url"] == "/robot/assets/"
        assert snap["model"]["urdf_path"] == "/tmp/source.urdf"
        assert snap["model"]["camera_feeds"] == [
            {
                "name": "d405",
                "urdf_link": "d405_depth_optical_frame",
                "url": "/api/cameras/d405/color.jpg",
                "width": 640,
                "height": 480,
            }
        ]
        assert snap["workspace"]["min"] == [-1.0, -2.0, 0.0]
        assert snap["workspace"]["max"] == [1.0, 2.0, 1.0]
        assert snap["robot"]["joints"] == {
            "joint0": 0.4000000059604645,
            "right_index_pip": -0.20000000298023224,
        }
        assert snap["xr"]["aligned"] is False
        assert snap["xr"]["head"] is None
        assert snap["xr"]["right_wrist"] is None

    asyncio.run(run())


def test_dashboard_snapshot_keeps_camera_feeds_without_urdf_link():
    class UnlinkedCameraPointCloud(CountingPointCloud):
        def dashboard_camera_feeds(self):
            return [
                {
                    "name": "d435i",
                    "url": "/api/cameras/d435i/color.jpg",
                    "calibration_url": "/api/cameras/d435i/calibration.jpg",
                    "width": 640,
                    "height": 480,
                }
            ]

    hub = TelemetryHub(
        point_cloud_source=UnlinkedCameraPointCloud(),
        robot_driver=FakeRobot(),
        workspace=_workspace(),
        urdf_url="/robot/robot.urdf",
        urdf_assets_url="/robot/assets/",
    )

    assert hub.snapshot()["model"]["camera_feeds"] == [
        {
            "name": "d435i",
            "url": "/api/cameras/d435i/color.jpg",
            "calibration_url": "/api/cameras/d435i/calibration.jpg",
            "width": 640,
            "height": 480,
        }
    ]


def test_dashboard_xr_pose_stays_unaligned_until_anchor_exists():
    hub = TelemetryHub(
        point_cloud_source=CountingPointCloud(),
        robot_driver=FakeRobot(),
        workspace=_workspace(),
        urdf_url="/robot/robot.urdf",
        urdf_assets_url="/robot/assets/",
    )
    hub.update_xr_pose(
        head_position=(0.0, 1.6, 0.0),
        head_orientation=(0.0, 0.0, 0.0, 1.0),
        right_wrist_position=(0.2, 1.1, -0.3),
        right_wrist_orientation=(0.0, 0.0, 0.0, 1.0),
        valid=True,
        timestamp=10.0,
    )
    assert hub.snapshot()["xr"]["aligned"] is False

    hub.update_anchor((0.1, 1.0, -0.2), timestamp=11.0)
    snap = hub.snapshot()

    assert snap["xr"]["aligned"] is True
    assert snap["xr"]["head"]["position"] == [0.0, 1.6, 0.0]
    assert snap["xr"]["right_wrist"]["position"] == [0.2, 1.1, -0.3]


def test_dashboard_xr_head_survives_missing_hand_and_wrist_curls_are_cached():
    hub = TelemetryHub(
        point_cloud_source=CountingPointCloud(),
        robot_driver=FakeRobot(),
        workspace=_workspace(),
        urdf_url="/robot/robot.urdf",
        urdf_assets_url="/robot/assets/",
    )

    hub.update_xr_pose(
        head_position=(0.0, 1.6, 0.0),
        head_orientation=(0.0, 0.0, 0.0, 1.0),
        right_wrist_position=(0.2, 1.1, -0.3),
        right_wrist_orientation=(0.0, 0.0, 0.0, 1.0),
        valid=False,
        head_valid=True,
        right_wrist_curls=(0.1, 0.2, 0.3, 0.4, 0.5),
        timestamp=10.0,
    )
    hub.update_anchor((0.1, 1.0, -0.2), timestamp=11.0)

    snap = hub.snapshot()

    assert snap["xr"]["aligned"] is True
    assert snap["xr"]["head"]["position"] == [0.0, 1.6, 0.0]
    assert snap["xr"]["right_wrist"] is None

    hub.update_xr_pose(
        head_position=(0.0, 1.6, 0.0),
        head_orientation=(0.0, 0.0, 0.0, 1.0),
        right_wrist_position=(0.2, 1.1, -0.3),
        right_wrist_orientation=(0.0, 0.0, 0.0, 1.0),
        valid=True,
        head_valid=True,
        right_wrist_curls=(0.1, 0.2, 0.3, 0.4, 0.5),
        timestamp=12.0,
    )

    snap = hub.snapshot()

    assert snap["xr"]["right_wrist"]["curls"] == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_multiple_dashboard_cloud_waiters_share_one_grab():
    async def run():
        pc = CountingPointCloud()
        hub = TelemetryHub(
            point_cloud_source=pc,
            robot_driver=FakeRobot(),
            workspace=_workspace(),
            urdf_url="/robot/robot.urdf",
            urdf_assets_url="/robot/assets/",
        )

        await hub.sample_pointcloud_once()
        first = await hub.wait_for_pointcloud(after_sequence=0, timeout=0.01)
        second = await hub.wait_for_pointcloud(after_sequence=0, timeout=0.01)

        assert first is not None
        assert second is not None
        assert first.sequence == second.sequence == 1
        assert first.payload == second.payload
        assert pc.grab_count == 1

    asyncio.run(run())


def test_dashboard_snapshot_contains_detected_cubes_from_reference_cloud():
    class CubePointCloud(CountingPointCloud):
        def __init__(self):
            super().__init__()
            edge = 0.038
            top = np.array(
                [
                    [x, y, edge]
                    for x in np.linspace(0.10, 0.138, 8)
                    for y in np.linspace(0.02, 0.058, 8)
                ],
                dtype=np.float32,
            )
            side = np.array(
                [
                    [0.10, y, z]
                    for y in np.linspace(0.02, 0.058, 8)
                    for z in np.linspace(0.006, 0.034, 4)
                ],
                dtype=np.float32,
            )
            points = np.vstack([top, side])
            self.frame = PointCloudFrame(
                points=points,
                colors=np.repeat(np.array([[200, 20, 20]], dtype=np.uint8), len(points), axis=0),
                timestamp=345.0,
            )

        def latest_cube_reference_frame(self):
            return self.frame

    async def run():
        hub = TelemetryHub(
            point_cloud_source=CubePointCloud(),
            robot_driver=FakeRobot(),
            workspace=_workspace(),
            urdf_url="/robot/robot.urdf",
            urdf_assets_url="/robot/assets/",
            cube_detector=CubeDetector(CubeDetectionConfig(window_size=1, min_observations=1)),
        )
        await hub.sample_pointcloud_once()
        snap = hub.snapshot()

        assert snap["cubes"]["count"] == 1
        cube = snap["cubes"]["items"][0]
        assert cube["id"] == "cube-0"
        assert cube["edge_m"] == 0.038
        assert cube["center_m"] == pytest.approx([0.119, 0.039, 0.019], abs=0.008)
        assert cube["roll_rad"] == 0.0
        assert cube["pitch_rad"] == 0.0
        assert "yaw_rad" in cube

    asyncio.run(run())


def test_dashboard_robot_snapshot_omits_joints_when_driver_has_no_names():
    async def run():
        robot = FakeRobot()
        robot.state = RobotState(
            wrist_pose=Pose.identity(frame="world"),
            joint_angles=np.zeros(6, dtype=np.float32),
            finger_curls=np.zeros(5, dtype=np.float32),
            timestamp=42.0,
        )
        hub = TelemetryHub(
            point_cloud_source=CountingPointCloud(),
            robot_driver=robot,
            workspace=_workspace(),
            urdf_url="/robot/robot.urdf",
            urdf_assets_url="/robot/assets/",
        )
        await hub.sample_robot_once()
        assert hub.snapshot()["robot"]["joints"] == {}

    asyncio.run(run())


def test_dashboard_snapshot_can_carry_calibration_status_and_world_cloud_mode():
    class CalibrationPointCloud(CountingPointCloud):
        def dashboard_pointcloud_frame(self):
            return "world"

    def calibration_snapshot():
        return {
            "mode": "continuous_calibration",
            "anchor_camera": "d405",
            "autosave": {"enabled": True, "state": "waiting_for_stability"},
            "targets": {
                "d435i": {
                    "stable": False,
                    "accepted_samples": 2,
                    "reprojection_error_px": 0.4,
                }
            },
        }

    hub = TelemetryHub(
        point_cloud_source=CalibrationPointCloud(),
        robot_driver=FakeRobot(),
        workspace=_workspace(),
        urdf_url="/robot/robot.urdf",
        urdf_assets_url="/robot/assets/",
        calibration_snapshot_provider=calibration_snapshot,
    )

    snap = hub.snapshot()

    assert snap["model"]["pointcloud_frame"] == "world"
    assert snap["calibration"]["mode"] == "continuous_calibration"
    assert snap["calibration"]["anchor_camera"] == "d405"
    assert snap["calibration"]["targets"]["d435i"]["accepted_samples"] == 2


def test_calibration_dashboard_serves_detection_overlay_jpeg(tmp_path):
    class FakeHub:
        def snapshot(self):
            return {"type": "snapshot"}

        async def wait_for_pointcloud(self, *, after_sequence, timeout):
            return None

    class FakeCalibrationSource:
        def latest_color_jpeg(self, camera_name):
            return None

        def latest_calibration_jpeg(self, camera_name):
            assert camera_name == "d405"
            return b"fake-jpeg"

    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot name='test'/>")
    app = _make_dashboard_app(
        hub=FakeHub(),
        source=FakeCalibrationSource(),
        static_dir=Path(__file__).resolve().parents[1] / "webxr_app" / "dashboard_static",
        urdf_path=urdf,
        robot_assets_root=tmp_path,
    )

    routes = {
        route.resource.canonical
        for route in app.router.routes()
        if route.resource is not None
    }

    assert "/api/cameras/{name}/calibration.jpg" in routes


def test_resolve_robot_asset_path_allows_assets_under_root(tmp_path):
    root = tmp_path / "robot"
    mesh_dir = root / "meshes"
    mesh_dir.mkdir(parents=True)
    mesh = mesh_dir / "link.stl"
    mesh.write_text("mesh")

    assert _resolve_robot_asset_path(root, "meshes/link.stl") == mesh.resolve()


def test_resolve_robot_asset_path_rejects_path_escape(tmp_path):
    root = tmp_path / "robot"
    root.mkdir()
    outside = tmp_path / "secret.stl"
    outside.write_text("secret")

    with pytest.raises(ValueError):
        _resolve_robot_asset_path(root, "../secret.stl")
