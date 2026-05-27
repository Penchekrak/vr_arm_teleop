import argparse
import json
import sys

import pytest

from teleop_backends.pointcloud.hardware import HardwarePointCloudSource
from teleop_backends.robot import NoopRobotDriver
from teleop_core.server import ServerConfig
from webxr_app.__main__ import _make_pc_source, _make_robot_driver


def _write_config(tmp_path):
    path = tmp_path / "cameras.json"
    path.write_text(
        json.dumps(
            {
                "cameras": [
                    {
                        "name": "rs-front",
                        "type": "realsense",
                        "serial": "123",
                        "calibrated": False,
                    }
                ]
            }
        )
    )
    return path


def test_hardware_pc_backend_uses_configured_fused_source(tmp_path):
    args = argparse.Namespace(cameras=_write_config(tmp_path))

    source = _make_pc_source("hardware", args)

    assert isinstance(source, HardwarePointCloudSource)


def test_hardware_pc_backend_requires_cameras_config():
    args = argparse.Namespace(cameras=None)

    with pytest.raises(SystemExit, match="--pc-backend hardware requires --cameras"):
        _make_pc_source("hardware", args)


def test_realsense_pc_backend_uses_legacy_realsense_wrapper(tmp_path):
    args = argparse.Namespace(cameras=_write_config(tmp_path))

    source = _make_pc_source("realsense", args)

    assert isinstance(source, HardwarePointCloudSource)


def test_server_config_has_dashboard_port_default():
    config = ServerConfig()

    assert config.port == 8000
    assert config.dashboard_port == 8001


def test_make_workspace_reads_default_config_workspace_when_no_flag(monkeypatch, tmp_path):
    import webxr_app.__main__ as main

    workspace_path = tmp_path / "workspace.json"
    workspace_path.write_text(json.dumps({
        "center": [0.35, 0.05, 0.225],
        "half_extents": [0.25, 0.3, 0.225],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "frame": "world",
    }))
    monkeypatch.setattr(main, "DEFAULT_WORKSPACE_PATH", workspace_path)
    args = argparse.Namespace(workspace=None)

    workspace = main._make_workspace(args, home=None)

    assert workspace.center == pytest.approx([0.35, 0.05, 0.225])
    assert workspace.half_extents == pytest.approx([0.25, 0.3, 0.225])


def test_parse_args_accepts_dashboard_port(monkeypatch):
    from webxr_app.__main__ import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["webxr_app", "--dashboard-port", "9001"],
    )

    args = _parse_args()

    assert args.dashboard_port == 9001


def test_noop_robot_backend_does_not_require_pybullet(monkeypatch):
    monkeypatch.delitem(sys.modules, "pybullet", raising=False)
    args = argparse.Namespace()

    robot = _make_robot_driver("noop", args)

    assert isinstance(robot, NoopRobotDriver)
