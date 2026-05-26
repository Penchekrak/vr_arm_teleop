from pathlib import Path
import subprocess
import sys

import pytest

from scripts.calibrate_two_cameras_charuco import parse_args


def test_calibration_cli_parses_config_driven_options():
    args = parse_args(
        [
            "--cameras",
            "config/hardware_cameras.json",
            "--anchor-camera",
            "d405",
            "--urdf",
            "robot.urdf",
            "--base-link",
            "base",
            "--camera-link",
            "camera_optical",
            "--squares-x",
            "7",
            "--squares-y",
            "5",
            "--square-length",
            "0.035",
            "--marker-length",
            "0.026",
            "--dictionary",
            "DICT_5X5_100",
            "--dashboard-port",
            "9001",
            "--rolling-window",
            "24",
            "--stability-seconds",
            "1.5",
            "--min-depth-corners",
            "9",
            "--max-kabsch-rms",
            "0.006",
            "--depth-neighborhood",
            "2",
            "--max-arm-motion-per-sample",
            "0.004",
            "--no-live-fusion-correction",
            "--live-correction-max-translation",
            "0.012",
            "--live-correction-max-rotation",
            "0.8",
            "--live-correction-max-rmse",
            "0.006",
            "--live-correction-min-points",
            "250",
            "--live-correction-smoothing",
            "0.6",
            "--no-autosave",
        ]
    )

    assert args.cameras == Path("config/hardware_cameras.json")
    assert args.anchor_camera == "d405"
    assert args.urdf == Path("robot.urdf")
    assert args.base_link == "base"
    assert args.camera_link == "camera_optical"
    assert args.squares_x == 7
    assert args.square_length == 0.035
    assert args.dashboard_port == 9001
    assert args.rolling_window == 24
    assert args.stability_seconds == 1.5
    assert args.min_depth_corners == 9
    assert args.max_kabsch_rms == 0.006
    assert args.depth_neighborhood == 2
    assert args.max_arm_motion_per_sample == 0.004
    assert args.live_fusion_correction is False
    assert args.live_correction_max_translation == 0.012
    assert args.live_correction_max_rotation == 0.8
    assert args.live_correction_max_rmse == 0.006
    assert args.live_correction_min_points == 250
    assert args.live_correction_smoothing == 0.6
    assert args.autosave is False


def test_calibration_cli_defaults_to_canonical_urdf_camera_chain():
    args = parse_args(
        [
            "--cameras",
            "config/hardware_cameras.json",
            "--squares-x",
            "7",
            "--squares-y",
            "5",
            "--square-length",
            "0.035",
            "--marker-length",
            "0.026",
            "--arm-ip",
            "10.10.10.20",
            "--rc5-api-path",
            "/opt/rc5_python_api",
        ]
    )

    assert args.urdf == (
        Path(__file__).resolve().parents[1]
        / "urdf_rc5_right_hand"
        / "urdf_with_simple_collisions.urdf"
    )
    assert args.base_link == "world"
    assert args.camera_link == "d405_depth_optical_frame"
    assert args.anchor_camera is None
    assert args.dashboard_port == 8001
    assert args.autosave is True
    assert args.live_fusion_correction is True
    assert args.live_correction_max_translation == 0.03
    assert args.live_correction_max_rotation == 3.0
    assert args.live_correction_max_rmse == 0.015
    assert args.live_correction_min_points == 300
    assert args.live_correction_smoothing == 0.35
    assert args.arm_ip == "10.10.10.20"
    assert args.rc5_api_path == Path("/opt/rc5_python_api")


def test_calibration_cli_requires_camera_config():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--squares-x",
                "7",
                "--squares-y",
                "5",
                "--square-length",
                "0.035",
                "--marker-length",
                "0.026",
            ]
        )


def test_calibration_script_help_runs_as_direct_file():
    result = subprocess.run(
        [sys.executable, "scripts/calibrate_two_cameras_charuco.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--cameras" in result.stdout
    assert "--anchor-camera" in result.stdout
