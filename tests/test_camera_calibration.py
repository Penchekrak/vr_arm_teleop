import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.calibrate_two_cameras_charuco import _detect_board_pose
from teleop_backends.camera_calibration import (
    AnchoredExtrinsicOptimizer,
    CameraDescriptor,
    CalibrationObservation,
    CalibrationDetection,
    CharucoBoardSpec,
    RealSenseColorFeed,
    UnavailableJointStateProvider,
    UrdfKinematicTree,
    ZedColorFeed,
    build_hardware_camera_config,
    estimate_two_camera_extrinsics,
    invert_transform,
    rotation_matrix_from_axis_angle,
    select_anchor_camera,
    transform_from_rt,
    write_calibrated_hardware_config,
)
from teleop_backends.pointcloud.hardware import CalibrationCameraFrame
from teleop_backends.pointcloud.hardware import load_hardware_config


def test_estimate_two_camera_extrinsics_chains_external_camera_to_world():
    world_from_arm_camera = transform_from_rt(
        rotation_matrix_from_axis_angle([0, 0, 1], 0.3),
        [0.4, -0.2, 0.8],
    )
    arm_camera_from_external_camera = transform_from_rt(
        rotation_matrix_from_axis_angle([0, 1, 0], -0.25),
        [0.15, 0.03, -0.08],
    )
    external_camera_from_arm_camera = invert_transform(arm_camera_from_external_camera)

    observations = []
    for i in range(12):
        arm_camera_from_board = transform_from_rt(
            rotation_matrix_from_axis_angle([1, 0.2, 0.1], 0.05 * i),
            [0.02 * i, -0.01 * i, 0.7 + 0.03 * i],
        )
        observations.append(
            CalibrationObservation(
                arm_camera_from_board=arm_camera_from_board,
                external_camera_from_board=(
                    external_camera_from_arm_camera @ arm_camera_from_board
                ),
                arm_reprojection_error=0.15,
                external_reprojection_error=0.2,
                corner_count=30,
            )
        )

    result = estimate_two_camera_extrinsics(
        observations,
        world_from_arm_camera=world_from_arm_camera,
        min_samples=10,
    )

    assert result.accepted_samples == 12
    assert result.rejected_samples == 0
    assert np.allclose(
        result.arm_camera_from_external_camera,
        arm_camera_from_external_camera,
        atol=1e-6,
    )
    assert np.allclose(
        result.world_from_external_camera,
        world_from_arm_camera @ arm_camera_from_external_camera,
        atol=1e-6,
    )


def test_build_hardware_camera_config_writes_runtime_compatible_extrinsics(tmp_path):
    world_from_arm_camera = transform_from_rt(np.eye(3), [0.1, 0.2, 0.3])
    world_from_external_camera = transform_from_rt(np.eye(3), [0.4, 0.5, 0.6])
    result = estimate_two_camera_extrinsics(
        [
            CalibrationObservation(
                arm_camera_from_board=np.eye(4),
                external_camera_from_board=transform_from_rt(np.eye(3), [-0.3, -0.3, -0.3]),
                arm_reprojection_error=0.1,
                external_reprojection_error=0.1,
                corner_count=24,
            )
            for _ in range(10)
        ],
        world_from_arm_camera=world_from_arm_camera,
        min_samples=10,
    )
    assert np.allclose(result.world_from_external_camera, world_from_external_camera)

    config = build_hardware_camera_config(
        result,
        arm_camera=CameraDescriptor(
            name="arm-cam",
            camera_type="zed2i",
            serial="zed-serial",
            width=1280,
            height=720,
            fps=30,
            camera_matrix=[[700, 0, 640], [0, 700, 360], [0, 0, 1]],
            distortion=[0, 0, 0, 0, 0],
        ),
        external_camera=CameraDescriptor(
            name="external-cam",
            camera_type="realsense",
            serial="rs-serial",
            width=640,
            height=480,
            fps=30,
            camera_matrix=[[610, 0, 320], [0, 610, 240], [0, 0, 1]],
            distortion=[0, 0, 0, 0, 0],
        ),
    )
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(config))

    loaded = load_hardware_config(path)

    assert [camera.name for camera in loaded.cameras] == ["arm-cam", "external-cam"]
    assert all(camera.calibrated for camera in loaded.cameras)
    assert np.allclose(loaded.cameras[0].world_from_camera, world_from_arm_camera)
    assert np.allclose(loaded.cameras[1].world_from_camera, world_from_external_camera)
    assert config["cameras"][0]["extrinsic_world_from_cam"] == config["cameras"][0]["world_from_camera"]


def test_charuco_board_spec_validates_dimensions_and_dictionary():
    spec = CharucoBoardSpec(
        squares_x=7,
        squares_y=5,
        square_length=0.035,
        marker_length=0.026,
        dictionary="DICT_5X5_100",
    )

    assert spec.squares_x == 7
    assert spec.squares_y == 5

    with pytest.raises(ValueError, match="marker_length"):
        CharucoBoardSpec(
            squares_x=7,
            squares_y=5,
            square_length=0.035,
            marker_length=0.04,
            dictionary="DICT_5X5_100",
        )


def test_urdf_fk_computes_base_to_camera_link(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """
        <robot name="test">
          <link name="base"/>
          <link name="joint_link"/>
          <link name="camera_optical"/>
          <joint name="pan" type="revolute">
            <parent link="base"/>
            <child link="joint_link"/>
            <origin xyz="1 0 0" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
          </joint>
          <joint name="camera_mount" type="fixed">
            <parent link="joint_link"/>
            <child link="camera_optical"/>
            <origin xyz="0 2 0" rpy="0 0 0"/>
          </joint>
        </robot>
        """
    )

    tree = UrdfKinematicTree.from_file(urdf)
    base_from_camera = tree.transform("base", "camera_optical", {"pan": math.pi / 2})

    assert np.allclose(base_from_camera[:3, 3], [-1, 0, 0], atol=1e-6)
    assert np.allclose(
        base_from_camera[:3, :3],
        rotation_matrix_from_axis_angle([0, 0, 1], math.pi / 2),
        atol=1e-6,
    )


def test_canonical_urdf_resolves_world_to_d405_depth_optical_frame():
    urdf = (
        Path(__file__).resolve().parents[1]
        / "urdf_rc5_right_hand"
        / "urdf_with_simple_collisions.urdf"
    )

    tree = UrdfKinematicTree.from_file(urdf)
    base_from_camera = tree.transform(
        "world",
        "d405_depth_optical_frame",
        {f"joint{i}": 0.0 for i in range(6)},
    )
    base_from_depth = tree.transform(
        "world",
        "d405_depth_frame",
        {f"joint{i}": 0.0 for i in range(6)},
    )

    assert base_from_camera.shape == (4, 4)
    assert np.all(np.isfinite(base_from_camera))
    assert np.allclose(
        base_from_camera[:3, 2],
        base_from_depth[:3, 0],
        atol=1e-6,
    )
    assert np.allclose(
        base_from_camera[:3, 0],
        -base_from_depth[:3, 1],
        atol=1e-6,
    )
    assert np.allclose(
        base_from_camera[:3, 1],
        -base_from_depth[:3, 2],
        atol=1e-6,
    )


def test_missing_camera_sdks_are_reported_only_when_feeds_start(monkeypatch):
    def missing_import(name):
        if name in {"pyrealsense2", "pyzed.sl"}:
            raise ModuleNotFoundError(name)
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", missing_import)

    rs_feed = RealSenseColorFeed(serial="rs-serial", width=640, height=480, fps=30)
    zed_feed = ZedColorFeed(serial="123456", resolution="HD720", fps=30)

    with pytest.raises(RuntimeError, match="pyrealsense2"):
        rs_feed.start()
    with pytest.raises(RuntimeError, match="pyzed.sl"):
        zed_feed.start()


def test_placeholder_joint_state_provider_fails_loudly():
    provider = UnavailableJointStateProvider("install the future arm SDK")

    with pytest.raises(RuntimeError, match="future arm SDK"):
        provider.read_joint_state()


def test_select_anchor_camera_defaults_to_camera_link(tmp_path):
    config_path = tmp_path / "cameras.json"
    config_path.write_text(json.dumps({
        "cameras": [
            {
                "name": "d405",
                "type": "realsense",
                "serial": "anchor",
                "urdf_link": "d405_depth_optical_frame",
                "calibrated": False,
            },
            {
                "name": "d435i",
                "type": "realsense",
                "serial": "target",
                "calibrated": False,
            },
        ]
    }))
    config = load_hardware_config(config_path)

    anchor = select_anchor_camera(
        config,
        anchor_name=None,
        camera_link="d405_depth_optical_frame",
    )

    assert anchor.name == "d405"


def test_select_anchor_camera_can_use_explicit_name(tmp_path):
    config_path = tmp_path / "cameras.json"
    config_path.write_text(json.dumps({
        "cameras": [
            {"name": "d405", "type": "realsense", "serial": "anchor"},
            {"name": "zed-overhead", "type": "zed2i", "serial": "123"},
        ]
    }))
    config = load_hardware_config(config_path)

    anchor = select_anchor_camera(
        config,
        anchor_name="zed-overhead",
        camera_link="d405_depth_optical_frame",
    )

    assert anchor.name == "zed-overhead"


def _detection(camera_from_board, reprojection=0.2, corners=24):
    return CalibrationDetection(
        camera_from_board=camera_from_board,
        reprojection_error=reprojection,
        corner_count=corners,
    )


def test_anchored_optimizer_uses_live_fk_to_keep_external_camera_world_pose_stable():
    world_from_external = transform_from_rt(
        rotation_matrix_from_axis_angle([0, 1, 0], -0.2),
        [0.45, -0.1, 0.7],
    )
    optimizer = AnchoredExtrinsicOptimizer(
        anchor_name="d405",
        target_names=("d435i",),
        min_samples=5,
        rolling_window=8,
        min_corners=12,
        max_reprojection_error_px=1.0,
        stability_seconds=0.0,
    )

    for i in range(6):
        world_from_anchor = transform_from_rt(
            rotation_matrix_from_axis_angle([0, 0, 1], 0.03 * i),
            [0.1 + 0.01 * i, 0.02 * i, 0.4],
        )
        world_from_board = transform_from_rt(
            rotation_matrix_from_axis_angle([1, 0.2, 0.1], 0.04 * i),
            [0.25 + 0.01 * i, -0.05, 0.85 + 0.02 * i],
        )
        optimizer.add_frame(
            world_from_anchor_camera=world_from_anchor,
            detections={
                "d405": _detection(invert_transform(world_from_anchor) @ world_from_board),
                "d435i": _detection(invert_transform(world_from_external) @ world_from_board),
            },
            timestamp=float(i),
        )

    status = optimizer.status()["targets"]["d435i"]

    assert status["stable"] is True
    assert status["accepted_samples"] >= 5
    assert np.allclose(status["world_from_camera"], world_from_external, atol=1e-6)


def test_anchored_optimizer_rejects_bad_detections_and_requires_all_targets_stable():
    world_from_anchor = np.eye(4)
    world_from_target = transform_from_rt(np.eye(3), [0.2, 0.0, 0.0])
    world_from_board = transform_from_rt(np.eye(3), [0.1, 0.0, 0.7])
    optimizer = AnchoredExtrinsicOptimizer(
        anchor_name="d405",
        target_names=("d435i", "zed-overhead"),
        min_samples=3,
        rolling_window=4,
        min_corners=12,
        max_reprojection_error_px=1.0,
        stability_seconds=0.0,
    )

    for i in range(3):
        optimizer.add_frame(
            world_from_anchor_camera=world_from_anchor,
            detections={
                "d405": _detection(world_from_board),
                "d435i": _detection(invert_transform(world_from_target) @ world_from_board),
                "zed-overhead": _detection(
                    invert_transform(world_from_target) @ world_from_board,
                    reprojection=4.0,
                ),
            },
            timestamp=float(i),
        )

    status = optimizer.status()

    assert status["targets"]["d435i"]["stable"] is True
    assert status["targets"]["zed-overhead"]["stable"] is False
    assert status["targets"]["zed-overhead"]["rejected_samples"] == 3
    assert status["all_stable"] is False


def test_write_calibrated_hardware_config_preserves_fields_and_creates_backup(tmp_path):
    config_path = tmp_path / "hardware_cameras.json"
    original = {
        "workspace_crop": {"min": [0, 0, 0], "max": [1, 1, 1]},
        "max_points": 5000,
        "cameras": [
            {
                "name": "d405",
                "type": "realsense",
                "serial": "anchor",
                "urdf_link": "d405_depth_optical_frame",
                "enabled": True,
                "calibrated": False,
            },
            {
                "name": "disabled",
                "type": "realsense",
                "serial": "disabled",
                "enabled": False,
                "calibrated": False,
            },
        ],
    }
    config_path.write_text(json.dumps(original, indent=2))
    world_from_d405 = transform_from_rt(np.eye(3), [0.1, 0.2, 0.3])

    result = write_calibrated_hardware_config(
        config_path,
        {"d405": world_from_d405},
        backup=True,
    )
    written = json.loads(config_path.read_text())
    backup = json.loads(result.backup_path.read_text())

    assert result.updated_camera_names == ("d405",)
    assert backup == original
    assert written["workspace_crop"] == original["workspace_crop"]
    assert written["max_points"] == 5000
    assert written["cameras"][0]["calibrated"] is True
    assert written["cameras"][0]["world_from_camera"] == world_from_d405.tolist()
    assert written["cameras"][0]["extrinsic_world_from_cam"] == world_from_d405.tolist()
    assert written["cameras"][1] == original["cameras"][1]


def _synthetic_charuco_scene(*, depth: float = 0.4):
    cv2 = importlib.import_module("cv2")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard((5, 4), 0.04, 0.03, dictionary)
    gray = board.generateImage((500, 400), marginSize=0)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)
    depth_m = np.full(gray.shape, depth, dtype=np.float32)
    descriptor = CameraDescriptor(
        name="synthetic",
        camera_type="realsense",
        serial=None,
        width=500,
        height=400,
        fps=30,
        camera_matrix=[
            [1000.0, 0.0, 0.0],
            [0.0, 1000.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        distortion=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    frame = CalibrationCameraFrame(
        image_rgb=rgb,
        depth_meters=depth_m,
        descriptor=descriptor,
        timestamp=123.0,
        frame_number=456,
    )
    return cv2, board, frame


def test_detect_board_pose_reports_partial_charuco_when_depth_is_missing():
    cv2, board, frame = _synthetic_charuco_scene()
    frame = CalibrationCameraFrame(
        image_rgb=frame.image_rgb,
        depth_meters=np.zeros_like(frame.depth_meters),
        descriptor=frame.descriptor,
        timestamp=frame.timestamp,
        frame_number=frame.frame_number,
    )

    detection = _detect_board_pose(
        cv2,
        board,
        frame,
        min_corners=8,
        min_depth_corners=6,
        depth_neighborhood=1,
        max_kabsch_rms_m=0.005,
    )

    assert detection.accepted is False
    assert detection.reason == "not_enough_valid_depth"
    assert detection.marker_count > 0
    assert detection.charuco_corner_count >= 8
    assert detection.depth_valid_corners == 0
    assert detection.overlay_rgb.shape == frame.image_rgb.shape


def test_detect_board_pose_uses_class_based_aruco_api_when_legacy_functions_are_missing(monkeypatch):
    cv2, board, frame = _synthetic_charuco_scene(depth=0.4)
    monkeypatch.delattr(cv2.aruco, "detectMarkers")
    monkeypatch.delattr(cv2.aruco, "interpolateCornersCharuco")

    detection = _detect_board_pose(
        cv2,
        board,
        frame,
        min_corners=8,
        min_depth_corners=8,
        depth_neighborhood=1,
        max_kabsch_rms_m=0.005,
    )

    assert detection.accepted is True
    assert detection.camera_from_board is not None
    assert detection.depth_valid_corners >= 8
    assert detection.kabsch_rms_m is not None
    assert detection.kabsch_rms_m < 0.005
    assert np.allclose(
        detection.camera_from_board[:3, :3],
        np.eye(3),
        atol=0.03,
    )
    assert np.allclose(
        detection.camera_from_board[:3, 3],
        [0.0, 0.0, 0.4],
        atol=0.015,
    )


def test_detect_board_pose_uses_depth_kabsch_to_recover_camera_from_board():
    cv2, board, frame = _synthetic_charuco_scene(depth=0.4)

    detection = _detect_board_pose(
        cv2,
        board,
        frame,
        min_corners=8,
        min_depth_corners=8,
        depth_neighborhood=1,
        max_kabsch_rms_m=0.005,
    )

    assert detection.accepted is True
    assert detection.camera_from_board is not None
    assert detection.depth_valid_corners >= 8
    assert detection.kabsch_rms_m is not None
    assert detection.kabsch_rms_m < 0.005
    assert np.allclose(
        detection.camera_from_board[:3, :3],
        np.eye(3),
        atol=0.03,
    )
    assert np.allclose(
        detection.camera_from_board[:3, 3],
        [0.0, 0.0, 0.4],
        atol=0.015,
    )


def test_detect_board_pose_rejects_high_depth_kabsch_rms():
    cv2, board, frame = _synthetic_charuco_scene(depth=0.4)
    noisy_depth = frame.depth_meters.copy()
    noisy_depth[:200, :] = 0.6
    frame = CalibrationCameraFrame(
        image_rgb=frame.image_rgb,
        depth_meters=noisy_depth,
        descriptor=frame.descriptor,
        timestamp=frame.timestamp,
        frame_number=frame.frame_number,
    )

    detection = _detect_board_pose(
        cv2,
        board,
        frame,
        min_corners=8,
        min_depth_corners=8,
        depth_neighborhood=1,
        max_kabsch_rms_m=0.001,
    )

    assert detection.accepted is False
    assert detection.reason == "kabsch_rms_too_high"
    assert detection.kabsch_rms_m is not None
    assert detection.kabsch_rms_m > 0.001
