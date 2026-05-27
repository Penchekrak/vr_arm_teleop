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


def test_load_hardware_config_defaults_realsense_to_low_bandwidth_resolution(tmp_path):
    raw_camera = _base_camera("rs-front", "realsense")
    raw_camera.pop("width")
    raw_camera.pop("height")
    path = _write_config(tmp_path, {"cameras": [raw_camera]})

    config = load_hardware_config(path)

    assert config.cameras[0].width == 424
    assert config.cameras[0].height == 240


def test_realsense_reader_uses_separate_depth_and_color_stream_profiles(tmp_path):
    raw_camera = _base_camera("rs-front", "realsense")
    raw_camera["width"] = 424
    raw_camera["height"] = 240
    raw_camera["color_width"] = 640
    raw_camera["color_height"] = 480
    raw_camera["color_fps"] = 30
    camera = load_hardware_config(
        _write_config(tmp_path, {"cameras": [raw_camera]})
    ).cameras[0]
    stream_calls = []

    class FakeConfig:
        def enable_device(self, _serial):
            pass

        def enable_stream(self, *args):
            stream_calls.append(args)

    class FakePipeline:
        def start(self, _config):
            return FakeProfile()

    class FakeProfile:
        def get_device(self):
            return self

        def first_depth_sensor(self):
            return self

        def get_depth_scale(self):
            return 0.001

    class FakeRs:
        class stream:
            depth = "depth"
            color = "color"

        class format:
            z16 = "z16"
            rgb8 = "rgb8"

        def pipeline(self):
            return FakePipeline()

        def config(self):
            return FakeConfig()

        def align(self, _stream):
            return None

        def pointcloud(self):
            return object()

    reader = RealSensePointCloudReader(camera)

    reader._start_blocking(FakeRs())

    assert stream_calls == [
        ("depth", 424, 240, "z16", 15),
        ("color", 640, 480, "rgb8", 30),
    ]


def test_realsense_reader_selects_profiles_for_requested_serial(tmp_path):
    raw_camera = _base_camera("d405", "realsense")
    raw_camera["serial"] = "d405-serial"
    raw_camera["width"] = 424
    raw_camera["height"] = 240
    raw_camera["fps"] = 30
    camera = load_hardware_config(
        _write_config(tmp_path, {"cameras": [raw_camera]})
    ).cameras[0]
    queried_serials = []
    stream_calls = []

    class FakeVideoProfile:
        def __init__(self, stream, fmt, width, height, fps):
            self._stream = stream
            self._format = fmt
            self._width = width
            self._height = height
            self._fps = fps

        def stream_type(self):
            return self._stream

        def format(self):
            return self._format

        def fps(self):
            return self._fps

        def as_video_stream_profile(self):
            return self

        def width(self):
            return self._width

        def height(self):
            return self._height

    class FakeSensor:
        def __init__(self, profiles):
            self._profiles = profiles

        def get_stream_profiles(self):
            return self._profiles

    class FakeDevice:
        def __init__(self, serial, profiles):
            self._serial = serial
            self._profiles = profiles

        def get_info(self, _info):
            queried_serials.append(self._serial)
            return self._serial

        def query_sensors(self):
            return [FakeSensor(self._profiles)]

    class FakeContext:
        def query_devices(self):
            return [
                FakeDevice(
                    "d435-serial",
                    [
                        FakeVideoProfile("depth", "z16", 848, 480, 30),
                        FakeVideoProfile("color", "rgb8", 1280, 720, 30),
                    ],
                ),
                FakeDevice(
                    "d405-serial",
                    [
                        FakeVideoProfile("depth", "z16", 640, 480, 30),
                        FakeVideoProfile("depth", "z16", 480, 270, 30),
                        FakeVideoProfile("depth", "z16", 256, 144, 90),
                        FakeVideoProfile("color", "rgb8", 640, 480, 30),
                        FakeVideoProfile("color", "rgb8", 480, 270, 30),
                    ],
                ),
            ]

    class FakeConfig:
        def enable_device(self, _serial):
            pass

        def enable_stream(self, *args):
            stream_calls.append(args)

    class FakePipeline:
        def start(self, _config):
            return FakeProfile()

    class FakeProfile:
        def get_device(self):
            return self

        def first_depth_sensor(self):
            return self

        def get_depth_scale(self):
            return 0.001

    class FakeRs:
        class camera_info:
            serial_number = "serial_number"

        class stream:
            depth = "depth"
            color = "color"

        class format:
            z16 = "z16"
            rgb8 = "rgb8"

        def context(self):
            return FakeContext()

        def pipeline(self):
            return FakePipeline()

        def config(self):
            return FakeConfig()

        def align(self, _stream):
            return None

        def pointcloud(self):
            return object()

    reader = RealSensePointCloudReader(camera)

    reader._start_blocking(FakeRs())

    assert queried_serials == ["d435-serial", "d405-serial"]
    assert stream_calls == [
        ("depth", 480, 270, "z16", 30),
        ("color", 480, 270, "rgb8", 30),
    ]


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


def test_hardware_source_exposes_unlinked_dashboard_camera_feeds(tmp_path):
    d405 = _base_camera("d405")
    d405["urdf_link"] = "d405_depth_optical_frame"
    d435i = _base_camera("d435i")
    path = _write_config(tmp_path, {"cameras": [d405, d435i]})

    source = HardwarePointCloudSource.from_config_file(
        path,
        reader_factory=lambda _camera: FakeJpegReader(None),
    )

    feeds = source.dashboard_camera_feeds()

    assert [feed["name"] for feed in feeds] == ["d405", "d435i"]
    assert feeds[1] == {
        "name": "d435i",
        "url": "/api/cameras/d435i/color.jpg",
        "width": 320,
        "height": 240,
    }
    assert "urdf_link" not in feeds[1]


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


def test_hardware_source_keeps_cube_reference_cloud_external_only(tmp_path):
    wrist_camera = _base_camera("d405")
    wrist_camera["urdf_link"] = "d405_depth_optical_frame"
    external_camera = _base_camera("d435i")
    path = _write_config(tmp_path, {"cameras": [wrist_camera, external_camera]})
    wrist_frame = PointCloudFrame(
        points=np.array([[0.0, 0.0, 0.2]], dtype=np.float32),
        colors=np.array([[255, 0, 0]], dtype=np.uint8),
        timestamp=10.0,
    )
    external_frame = PointCloudFrame(
        points=np.array([[0.2, 0.1, 0.038]], dtype=np.float32),
        colors=np.array([[0, 255, 0]], dtype=np.uint8),
        timestamp=11.0,
    )
    frames = {"d405": wrist_frame, "d435i": external_frame}

    source = HardwarePointCloudSource.from_config_file(
        path,
        reader_factory=lambda camera: FakeReader(frames[camera.name]),
    )

    async def run():
        await source.start()
        try:
            fused = await source.grab()
            reference = source.latest_cube_reference_frame()
        finally:
            await source.stop()
        assert fused is not None
        assert fused.n_points == 2
        assert reference is not None
        assert reference.n_points == 1
        assert np.allclose(reference.points[0], external_frame.points[0])
        assert np.array_equal(reference.colors[0], external_frame.colors[0])

    asyncio.run(run())


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


def test_realsense_reader_restarts_pipeline_after_frame_timeout(monkeypatch, tmp_path):
    config = _base_camera("rs-front", "realsense")
    config["frame_timeout_ms"] = 25
    config["restart_after_timeouts"] = 1
    camera = load_hardware_config(
        _write_config(tmp_path, {"cameras": [config]})
    ).cameras[0]

    class FakePipeline:
        def __init__(self):
            self.started = False
            self.stopped = False

        def start(self, _config):
            self.started = True
            return FakeProfile()

        def stop(self):
            self.stopped = True

        def wait_for_frames(self, _timeout_ms=None):
            raise RuntimeError("Frame didn't arrive within 5000")

    class FakeConfig:
        def enable_device(self, _serial):
            pass

        def enable_stream(self, *_args):
            pass

    class FakeProfile:
        def get_device(self):
            return self

        def first_depth_sensor(self):
            return self

        def get_depth_scale(self):
            return 0.001

    class FakeRs:
        class stream:
            depth = object()
            color = object()

        class format:
            z16 = object()
            rgb8 = object()

        def __init__(self):
            self.pipelines = []

        def pipeline(self):
            pipeline = FakePipeline()
            self.pipelines.append(pipeline)
            return pipeline

        def config(self):
            return FakeConfig()

        def align(self, _stream):
            return None

        def pointcloud(self):
            return object()

    rs = FakeRs()

    original_import_module = importlib.import_module

    def fake_import(name):
        if name == "pyrealsense2":
            return rs
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    reader = RealSensePointCloudReader(camera)

    asyncio.run(reader.start())
    frame = asyncio.run(reader.grab_camera_frame())

    assert frame is None
    assert len(rs.pipelines) == 2
    assert rs.pipelines[0].stopped is True
    assert rs.pipelines[1].started is True


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
