import asyncio
import importlib
import json
import time

import numpy as np
import pytest

from teleop_core.point_cloud import PointCloudFrame
from teleop_backends.pointcloud.hardware import (
    CameraPointCloudReader,
    HardwarePointCloudSource,
    RealSensePointCloudReader,
    Zed2iPointCloudReader,
    fuse_camera_frames,
    load_hardware_config,
)


IDENTITY = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]


def _write_config(tmp_path, data):
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(data))
    return path


def _base_camera(name, camera_type="realsense", calibrated=True, enabled=True):
    return {
        "name": name,
        "type": camera_type,
        "enabled": enabled,
        "serial": f"{name}-serial",
        "width": 320,
        "height": 240,
        "fps": 15,
        "downsample": 2,
        "z_min": 0.2,
        "z_max": 1.5,
        "calibrated": calibrated,
        "world_from_camera": IDENTITY,
    }


def test_load_hardware_config_parses_mixed_enabled_cameras(tmp_path):
    first = _base_camera("rs-front", "realsense")
    first["urdf_link"] = "d405_depth_optical_frame"
    path = _write_config(
        tmp_path,
        {
            "workspace_crop": {"min": [0, 0, 0], "max": [1, 1, 1]},
            "max_points": 5000,
            "cameras": [
                first,
                _base_camera("zed-overhead", "zed2i"),
                _base_camera("rs-disabled", "realsense", enabled=False),
            ],
        },
    )

    config = load_hardware_config(path)

    assert [camera.name for camera in config.cameras] == ["rs-front", "zed-overhead"]
    assert [camera.camera_type for camera in config.cameras] == ["realsense", "zed2i"]
    assert config.max_points == 5000
    assert config.workspace_crop is not None
    assert np.allclose(config.workspace_crop[0], [0, 0, 0])
    assert np.allclose(config.workspace_crop[1], [1, 1, 1])
    assert config.cameras[0].world_from_camera.shape == (4, 4)
    assert config.cameras[0].downsample == 2
    assert config.cameras[0].z_min == 0.2
    assert config.cameras[0].z_max == 1.5
    assert config.cameras[0].urdf_link == "d405_depth_optical_frame"


def test_load_hardware_config_rejects_invalid_transform(tmp_path):
    bad_camera = _base_camera("rs-front")
    bad_camera["world_from_camera"] = [[1, 0, 0], [0, 1, 0]]
    path = _write_config(tmp_path, {"cameras": [bad_camera]})

    with pytest.raises(ValueError, match="world_from_camera"):
        load_hardware_config(path)


def test_load_hardware_config_accepts_calibration_extrinsic_alias(tmp_path):
    camera = _base_camera("rs-front")
    camera.pop("world_from_camera")
    camera["extrinsic_world_from_cam"] = [
        [1, 0, 0, 0.1],
        [0, 1, 0, 0.2],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1],
    ]
    path = _write_config(tmp_path, {"cameras": [camera]})

    config = load_hardware_config(path)

    assert np.allclose(config.cameras[0].world_from_camera[:3, 3], [0.1, 0.2, 0.3])


class FakeReader(CameraPointCloudReader):
    def __init__(self, frame):
        self.frame = frame
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def grab_camera_frame(self):
        return self.frame


class FakeJpegReader(FakeReader):
    def latest_color_jpeg(self):
        return b"jpeg-bytes"


def test_hardware_source_exposes_dashboard_camera_feed_metadata_and_jpeg(tmp_path):
    camera = _base_camera("d405")
    camera["urdf_link"] = "d405_depth_optical_frame"
    path = _write_config(tmp_path, {"cameras": [camera]})

    source = HardwarePointCloudSource.from_config_file(
        path,
        reader_factory=lambda _camera: FakeJpegReader(None),
    )

    async def run():
        await source.start()
        try:
            assert source.dashboard_camera_feeds() == [
                {
                    "name": "d405",
                    "urdf_link": "d405_depth_optical_frame",
                    "url": "/api/cameras/d405/color.jpg",
                    "width": 320,
                    "height": 240,
                }
            ]
            assert source.dashboard_pointcloud_frame() == "world"
            assert source.latest_color_jpeg("d405") == b"jpeg-bytes"
        finally:
            await source.stop()

    asyncio.run(run())


def test_uncalibrated_enabled_camera_starts_but_gates_display(tmp_path, capsys):
    path = _write_config(
        tmp_path,
        {"cameras": [_base_camera("rs-front", calibrated=False)]},
    )
    frame = PointCloudFrame(
        points=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
        colors=np.array([[255, 0, 0]], dtype=np.uint8),
        timestamp=time.monotonic(),
    )
    readers = []

    def reader_factory(_camera):
        reader = FakeReader(frame)
        readers.append(reader)
        return reader

    source = HardwarePointCloudSource.from_config_file(path, reader_factory=reader_factory)
    asyncio.run(source.start())
    try:
        assert readers[0].started is True
        assert asyncio.run(source.grab()) is None
        assert "uncalibrated" in capsys.readouterr().out.lower()
    finally:
        asyncio.run(source.stop())


def test_fuse_camera_frames_applies_transform_crop_and_max_points(tmp_path):
    world_from_camera = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    config = _base_camera("rs-front")
    config["world_from_camera"] = world_from_camera.tolist()
    camera = load_hardware_config(_write_config(tmp_path, {"cameras": [config]})).cameras[0]
    frame = PointCloudFrame(
        points=np.array(
            [
                [0.0, 0.0, 0.2],
                [1.0, 0.0, 0.2],
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float32,
        ),
        colors=np.array(
            [
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90],
            ],
            dtype=np.uint8,
        ),
        timestamp=123.0,
    )

    fused = fuse_camera_frames(
        [(camera, frame)],
        workspace_crop=(
            np.array([0, 0, 0], dtype=np.float32),
            np.array([3, 3, 4], dtype=np.float32),
        ),
        max_points=1,
    )

    assert fused is not None
    assert fused.timestamp == 123.0
    assert fused.points.shape == (1, 3)
    assert np.allclose(fused.points[0], [1.0, 2.0, 3.2])
    assert np.array_equal(fused.colors[0], [10, 20, 30])


def test_configured_realsense_source_does_not_import_sdk_until_start(tmp_path, monkeypatch):
    path = _write_config(tmp_path, {"cameras": [_base_camera("rs-front", "realsense")]})

    def fail_import(name):
        if name == "pyrealsense2":
            raise AssertionError("pyrealsense2 imported during config load")
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", fail_import)
    HardwarePointCloudSource.from_config_file(path)


def test_realsense_reader_reports_missing_sdk_on_start(monkeypatch, tmp_path):
    camera = load_hardware_config(
        _write_config(tmp_path, {"cameras": [_base_camera("rs-front", "realsense")]})
    ).cameras[0]

    def missing_import(name):
        if name == "pyrealsense2":
            raise ModuleNotFoundError(name)
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", missing_import)
    reader = RealSensePointCloudReader(camera)

    with pytest.raises(RuntimeError, match="pyrealsense2"):
        asyncio.run(reader.start())


def test_zed_reader_reports_missing_sdk_on_start(monkeypatch, tmp_path):
    camera = load_hardware_config(
        _write_config(tmp_path, {"cameras": [_base_camera("zed-overhead", "zed2i")]})
    ).cameras[0]

    def missing_import(name):
        if name == "pyzed.sl":
            raise ModuleNotFoundError(name)
        return importlib.import_module(name)

    monkeypatch.setattr(importlib, "import_module", missing_import)
    reader = Zed2iPointCloudReader(camera)

    with pytest.raises(RuntimeError, match="pyzed"):
        asyncio.run(reader.start())
