"""Configured hardware point-cloud capture and fusion.

This module keeps camera SDK details behind backend-local readers. The
core server still sees only ``PointCloudSource`` and receives fused
``PointCloudFrame`` instances in robot-world coordinates.
"""

from __future__ import annotations

import abc
import asyncio
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

import numpy as np

from teleop_core.point_cloud import PointCloudFrame, PointCloudSource
from teleop_backends.camera_calibration import CameraDescriptor


SUPPORTED_CAMERA_TYPES = {"realsense", "zed2i"}


@dataclass(frozen=True)
class HardwareCameraConfig:
    """One enabled camera from the hardware point-cloud config."""

    name: str
    camera_type: str
    serial: str | None
    world_from_camera: np.ndarray
    calibrated: bool
    urdf_link: str | None = None
    width: int = 424
    height: int = 240
    fps: int = 30
    color_width: int | None = None
    color_height: int | None = None
    color_fps: int | None = None
    downsample: int = 4
    z_min: float = 0.15
    z_max: float = 2.5
    resolution: str | None = None
    depth_mode: str | None = None
    frame_timeout_ms: int = 1000
    restart_after_timeouts: int = 2


@dataclass(frozen=True)
class HardwarePointCloudConfig:
    """Top-level fused hardware point-cloud config."""

    cameras: tuple[HardwareCameraConfig, ...]
    workspace_crop: Optional[tuple[np.ndarray, np.ndarray]] = None
    max_points: Optional[int] = None

    @property
    def display_calibrated(self) -> bool:
        return all(camera.calibrated for camera in self.cameras)

    @property
    def uncalibrated_camera_names(self) -> tuple[str, ...]:
        return tuple(camera.name for camera in self.cameras if not camera.calibrated)


@dataclass(frozen=True)
class CalibrationCameraFrame:
    """RGB-D frame used by camera calibration diagnostics and depth fitting."""

    image_rgb: np.ndarray
    depth_meters: np.ndarray
    descriptor: CameraDescriptor
    timestamp: float | None = None
    frame_number: int | None = None


@dataclass(frozen=True)
class _RealSenseVideoProfile:
    stream: object
    format: object
    width: int
    height: int
    fps: int


class CameraPointCloudReader(abc.ABC):
    """Backend-local interface for one hardware camera."""

    @abc.abstractmethod
    async def start(self) -> None:
        """Open the camera and start capture."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Release camera resources."""

    @abc.abstractmethod
    async def grab_camera_frame(self) -> Optional[PointCloudFrame]:
        """Return a point cloud in that camera's local frame."""

    def latest_color_jpeg(self) -> bytes | None:
        """Return the latest RGB color frame encoded as JPEG, if available."""
        return None

    def latest_color_rgb(self) -> np.ndarray | None:
        """Return the latest RGB color frame, if available."""
        return None

    def latest_calibration_frame(self) -> CalibrationCameraFrame | None:
        """Return the latest depth-aligned RGB-D frame for calibration."""
        return None

    def descriptor(self) -> CameraDescriptor | None:
        """Return camera intrinsics when the backend SDK exposes them."""
        return None


ReaderFactory = Callable[[HardwareCameraConfig], CameraPointCloudReader]


def load_hardware_config(path: Path) -> HardwarePointCloudConfig:
    """Load a mixed RealSense/ZED hardware point-cloud config."""

    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("hardware camera config must be a JSON object")
    raw_cameras = data.get("cameras")
    if not isinstance(raw_cameras, list):
        raise ValueError("hardware camera config requires a 'cameras' list")

    cameras = []
    for index, raw in enumerate(raw_cameras):
        if not isinstance(raw, dict):
            raise ValueError(f"camera #{index} must be an object")
        if raw.get("enabled", True) is False:
            continue
        cameras.append(_parse_camera(raw, index))

    if not cameras:
        raise ValueError("hardware camera config has no enabled cameras")

    workspace_crop = _parse_workspace_crop(data.get("workspace_crop"))
    max_points = data.get("max_points")
    if max_points is not None:
        max_points = int(max_points)
        if max_points <= 0:
            raise ValueError("max_points must be positive when provided")

    return HardwarePointCloudConfig(
        cameras=tuple(cameras),
        workspace_crop=workspace_crop,
        max_points=max_points,
    )


def _parse_camera(raw: dict, index: int) -> HardwareCameraConfig:
    camera_type = str(raw.get("type", "")).lower()
    if camera_type not in SUPPORTED_CAMERA_TYPES:
        raise ValueError(
            f"camera #{index} has unsupported type {camera_type!r}; "
            f"expected one of {sorted(SUPPORTED_CAMERA_TYPES)}"
        )

    serial = raw.get("serial")
    if serial is not None:
        serial = str(serial)
    name = str(raw.get("name") or serial or f"{camera_type}-{index}")
    calibrated = bool(raw.get("calibrated", False))

    transform_value = raw.get("world_from_camera", raw.get("extrinsic_world_from_cam"))
    if transform_value is None:
        if calibrated:
            raise ValueError(
                f"camera {name!r} is calibrated but lacks world_from_camera"
            )
        world_from_camera = np.eye(4, dtype=np.float32)
    else:
        world_from_camera = _parse_transform(transform_value, name)

    z_min = float(raw.get("z_min", 0.15))
    z_max = float(raw.get("z_max", 2.5))
    if z_min >= z_max:
        raise ValueError(f"camera {name!r} requires z_min < z_max")

    downsample = int(raw.get("downsample", 4))
    if downsample <= 0:
        raise ValueError(f"camera {name!r} downsample must be positive")
    color_width = raw.get("color_width")
    color_height = raw.get("color_height")
    color_fps = raw.get("color_fps")

    return HardwareCameraConfig(
        name=name,
        camera_type=camera_type,
        serial=serial,
        world_from_camera=world_from_camera,
        calibrated=calibrated,
        urdf_link=(
            None if raw.get("urdf_link") is None else str(raw.get("urdf_link"))
        ),
        width=int(raw.get("width", 424)),
        height=int(raw.get("height", 240)),
        fps=int(raw.get("fps", 30)),
        color_width=None if color_width is None else int(color_width),
        color_height=None if color_height is None else int(color_height),
        color_fps=None if color_fps is None else int(color_fps),
        downsample=downsample,
        z_min=z_min,
        z_max=z_max,
        resolution=raw.get("resolution"),
        depth_mode=raw.get("depth_mode"),
        frame_timeout_ms=max(1, int(raw.get("frame_timeout_ms", 1000))),
        restart_after_timeouts=max(1, int(raw.get("restart_after_timeouts", 2))),
    )


def _parse_transform(value, camera_name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError(f"camera {camera_name!r} world_from_camera must be a 4x4 matrix")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"camera {camera_name!r} world_from_camera contains non-finite values")
    return transform


def _parse_workspace_crop(value) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if value is None:
        return None
    if not isinstance(value, dict) or "min" not in value or "max" not in value:
        raise ValueError("workspace_crop must be an object with 'min' and 'max'")
    min_corner = np.asarray(value["min"], dtype=np.float32)
    max_corner = np.asarray(value["max"], dtype=np.float32)
    if min_corner.shape != (3,) or max_corner.shape != (3,):
        raise ValueError("workspace_crop min/max must each have 3 values")
    if np.any(min_corner > max_corner):
        raise ValueError("workspace_crop min must be <= max on every axis")
    return min_corner, max_corner


def fuse_camera_frames(
    camera_frames: list[tuple[HardwareCameraConfig, PointCloudFrame]],
    *,
    workspace_crop: Optional[tuple[np.ndarray, np.ndarray]] = None,
    max_points: Optional[int] = None,
) -> Optional[PointCloudFrame]:
    """Transform camera-local frames into one world-frame cloud."""

    world_points_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    timestamps: list[float] = []

    for camera, frame in camera_frames:
        if frame.n_points == 0:
            continue
        points = _transform_points(frame.points, camera.world_from_camera)
        colors = np.asarray(frame.colors, dtype=np.uint8)
        if workspace_crop is not None:
            min_corner, max_corner = workspace_crop
            mask = np.all((points >= min_corner) & (points <= max_corner), axis=1)
            points = points[mask]
            colors = colors[mask]
        if points.shape[0] == 0:
            continue
        world_points_parts.append(points.astype(np.float32, copy=False))
        color_parts.append(colors.astype(np.uint8, copy=False))
        timestamps.append(float(frame.timestamp))

    if not world_points_parts:
        return None

    points = np.concatenate(world_points_parts, axis=0)
    colors = np.concatenate(color_parts, axis=0)
    if max_points is not None and points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[indices]
        colors = colors[indices]

    return PointCloudFrame(
        points=points,
        colors=colors,
        timestamp=max(timestamps) if timestamps else time.monotonic(),
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homogeneous = np.concatenate([points, ones], axis=1)
    return (homogeneous @ transform.T)[:, :3]


def _encode_rgb_jpeg(image: np.ndarray) -> bytes | None:
    try:
        cv2 = importlib.import_module("cv2")
    except ModuleNotFoundError:
        return None
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    bgr = np.ascontiguousarray(rgb[:, :, :3][:, :, ::-1])
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 80],
    )
    if not ok:
        return None
    return encoded.tobytes()


def _is_realsense_frame_timeout(message: str) -> bool:
    normalized = message.lower()
    return (
        "frame" in normalized
        and "arrive" in normalized
        and (
            "5000" in normalized
            or "5 seconds" in normalized
            or "timeout" in normalized
        )
    )


class HardwarePointCloudSource(PointCloudSource):
    """Fuses configured RealSense and ZED 2i readers into one cloud."""

    def __init__(
        self,
        config: HardwarePointCloudConfig,
        reader_factory: ReaderFactory | None = None,
    ) -> None:
        self._config = config
        self._reader_factory = reader_factory or default_reader_factory
        self._readers: list[tuple[HardwareCameraConfig, CameraPointCloudReader]] = []
        self._started = False
        self._warned_uncalibrated = False
        self._latest_cube_reference_frame: PointCloudFrame | None = None

    @classmethod
    def from_config_file(
        cls,
        path: Path,
        reader_factory: ReaderFactory | None = None,
    ) -> "HardwarePointCloudSource":
        return cls(load_hardware_config(path), reader_factory=reader_factory)

    @property
    def display_enabled(self) -> bool:
        return self._config.display_calibrated

    def dashboard_camera_feeds(self) -> list[dict[str, object]]:
        """Describe color feeds that the dashboard can display."""
        feeds: list[dict[str, object]] = []
        for camera in self._config.cameras:
            feed: dict[str, object] = {
                "name": camera.name,
                "url": f"/api/cameras/{quote(camera.name, safe='')}/color.jpg",
                "width": int(camera.color_width or camera.width),
                "height": int(camera.color_height or camera.height),
            }
            if camera.urdf_link:
                feed["urdf_link"] = camera.urdf_link
            feeds.append(feed)
        return feeds

    def dashboard_pointcloud_frame(self) -> str:
        return "world"

    def latest_color_jpeg(self, camera_name: str) -> bytes | None:
        for camera, reader in self._readers:
            if camera.name == camera_name:
                return reader.latest_color_jpeg()
        return None

    def latest_cube_reference_frame(self) -> PointCloudFrame | None:
        """Return the latest external-camera world cloud for cube detection.

        Wrist-mounted cameras have a URDF link in the current hardware config;
        those are useful for operator view but intentionally excluded from
        tabletop cube fitting because their viewpoint moves with the robot.
        """
        return self._latest_cube_reference_frame

    async def start(self) -> None:
        if self._started:
            return
        self._readers = [
            (camera, self._reader_factory(camera))
            for camera in self._config.cameras
        ]
        for _, reader in self._readers:
            await reader.start()
        self._started = True
        self._warn_if_uncalibrated()

    async def stop(self) -> None:
        for _, reader in reversed(self._readers):
            try:
                await reader.stop()
            except Exception as exc:
                print(f"[pointcloud] reader stop failed: {exc!r}")
        self._readers = []
        self._started = False

    async def grab(self) -> Optional[PointCloudFrame]:
        if not self._started:
            return None
        if not self.display_enabled:
            self._warn_if_uncalibrated()
            return None

        frames: list[tuple[HardwareCameraConfig, PointCloudFrame]] = []
        for camera, reader in self._readers:
            frame = await reader.grab_camera_frame()
            if frame is not None:
                frames.append((camera, frame))

        external_frames = [
            (camera, frame)
            for camera, frame in frames
            if camera.urdf_link is None
        ]
        self._latest_cube_reference_frame = fuse_camera_frames(
            external_frames,
            workspace_crop=self._config.workspace_crop,
            max_points=self._config.max_points,
        )

        return fuse_camera_frames(
            frames,
            workspace_crop=self._config.workspace_crop,
            max_points=self._config.max_points,
        )

    def _warn_if_uncalibrated(self) -> None:
        if self.display_enabled or self._warned_uncalibrated:
            return
        names = ", ".join(self._config.uncalibrated_camera_names)
        print(
            "[pointcloud] hardware capture started, but AR point-cloud display "
            f"is gated because these cameras are uncalibrated: {names}"
        )
        self._warned_uncalibrated = True


def default_reader_factory(camera: HardwareCameraConfig) -> CameraPointCloudReader:
    if camera.camera_type == "realsense":
        return RealSensePointCloudReader(camera)
    if camera.camera_type == "zed2i":
        return Zed2iPointCloudReader(camera)
    raise ValueError(f"unsupported camera type: {camera.camera_type!r}")


def _resolve_realsense_stream_profiles(
    rs,
    camera: HardwareCameraConfig,
) -> tuple[_RealSenseVideoProfile, _RealSenseVideoProfile]:
    requested_depth = _RealSenseVideoProfile(
        stream=rs.stream.depth,
        format=rs.format.z16,
        width=int(camera.width),
        height=int(camera.height),
        fps=int(camera.fps),
    )
    requested_color = _RealSenseVideoProfile(
        stream=rs.stream.color,
        format=rs.format.rgb8,
        width=int(camera.color_width or 0),
        height=int(camera.color_height or 0),
        fps=int(camera.color_fps or 0),
    )
    profiles = _realsense_profiles_for_camera(rs, camera)
    if profiles is None:
        color_width = camera.color_width or camera.width
        color_height = camera.color_height or camera.height
        color_fps = camera.color_fps or camera.fps
        return requested_depth, _RealSenseVideoProfile(
            stream=rs.stream.color,
            format=rs.format.rgb8,
            width=int(color_width),
            height=int(color_height),
            fps=int(color_fps),
        )
    depth_profile = _choose_realsense_depth_profile(
        profiles,
        requested_depth,
        camera=camera,
    )
    color_profile = _choose_realsense_color_profile(
        profiles,
        requested_color,
        fallback_fps=depth_profile.fps,
        target_area=depth_profile.width * depth_profile.height,
        camera=camera,
    )
    return depth_profile, color_profile


def _realsense_profiles_for_camera(
    rs,
    camera: HardwareCameraConfig,
) -> list[_RealSenseVideoProfile] | None:
    if not camera.serial:
        return None
    context_factory = getattr(rs, "context", None)
    if not callable(context_factory):
        return None
    try:
        devices = list(context_factory().query_devices())
    except Exception:
        return None

    connected_serials: list[str] = []
    serial_info = getattr(getattr(rs, "camera_info", object()), "serial_number", None)
    for device in devices:
        try:
            serial = str(device.get_info(serial_info))
        except Exception:
            continue
        connected_serials.append(serial)
        if serial == camera.serial:
            return _realsense_video_profiles(device)

    connected = ", ".join(connected_serials) if connected_serials else "none"
    raise RuntimeError(
        f"RealSense camera {camera.name!r} serial {camera.serial!r} is not connected; "
        f"connected serials: {connected}"
    )


def _realsense_video_profiles(device) -> list[_RealSenseVideoProfile]:
    profiles: list[_RealSenseVideoProfile] = []
    seen: set[tuple[str, str, int, int, int]] = set()
    for sensor in device.query_sensors():
        for raw_profile in sensor.get_stream_profiles():
            try:
                video_profile = raw_profile.as_video_stream_profile()
                profile = _RealSenseVideoProfile(
                    stream=raw_profile.stream_type(),
                    format=raw_profile.format(),
                    width=int(video_profile.width()),
                    height=int(video_profile.height()),
                    fps=int(raw_profile.fps()),
                )
            except Exception:
                continue
            key = (
                repr(profile.stream),
                repr(profile.format),
                profile.width,
                profile.height,
                profile.fps,
            )
            if key in seen:
                continue
            seen.add(key)
            profiles.append(profile)
    return profiles


def _choose_realsense_depth_profile(
    profiles: list[_RealSenseVideoProfile],
    requested: _RealSenseVideoProfile,
    *,
    camera: HardwareCameraConfig,
) -> _RealSenseVideoProfile:
    for profile in profiles:
        if _realsense_profile_matches(profile, requested):
            return profile
    candidates = [
        profile
        for profile in profiles
        if profile.stream == requested.stream and profile.format == requested.format
    ]
    if not candidates:
        raise RuntimeError(
            f"RealSense camera {camera.name!r} serial {camera.serial!r} exposes no Z16 "
            "depth video profiles"
        )
    chosen = _sort_realsense_profiles(
        candidates,
        preferred_fps=requested.fps,
        target_area=requested.width * requested.height,
        prefer_at_least_target=True,
    )[0]
    print(
        f"[pointcloud] RealSense {camera.name} requested depth "
        f"{requested.width}x{requested.height}@{requested.fps} is unsupported; "
        f"using {chosen.width}x{chosen.height}@{chosen.fps}"
    )
    return chosen


def _choose_realsense_color_profile(
    profiles: list[_RealSenseVideoProfile],
    requested: _RealSenseVideoProfile,
    *,
    fallback_fps: int,
    target_area: int,
    camera: HardwareCameraConfig,
) -> _RealSenseVideoProfile:
    candidates = [
        profile
        for profile in profiles
        if profile.stream == requested.stream and profile.format == requested.format
    ]
    explicit = requested.width > 0 or requested.height > 0 or requested.fps > 0
    if explicit:
        matches = [
            profile
            for profile in candidates
            if (requested.width <= 0 or profile.width == requested.width)
            and (requested.height <= 0 or profile.height == requested.height)
            and (requested.fps <= 0 or profile.fps == requested.fps)
        ]
        if matches:
            return _sort_realsense_profiles(
                matches,
                preferred_fps=fallback_fps,
                target_area=target_area,
                prefer_at_least_target=True,
            )[0]
        supported = _format_realsense_profiles(candidates)
        requested_text = (
            f"{requested.width or '*'}x{requested.height or '*'}@"
            f"{requested.fps or '*'}"
        )
        raise RuntimeError(
            f"RealSense camera {camera.name!r} serial {camera.serial!r} does not support "
            f"requested color profile {requested_text}; "
            f"supported color profiles: {supported}"
        )
    if not candidates:
        raise RuntimeError(
            f"RealSense camera {camera.name!r} serial {camera.serial!r} exposes no RGB8 "
            "color video profiles"
        )
    return _sort_realsense_profiles(
        candidates,
        preferred_fps=fallback_fps,
        target_area=target_area,
        prefer_at_least_target=True,
    )[0]


def _realsense_profile_matches(
    profile: _RealSenseVideoProfile,
    requested: _RealSenseVideoProfile,
) -> bool:
    return (
        profile.stream == requested.stream
        and profile.format == requested.format
        and profile.width == requested.width
        and profile.height == requested.height
        and profile.fps == requested.fps
    )


def _sort_realsense_profiles(
    profiles: list[_RealSenseVideoProfile],
    *,
    preferred_fps: int,
    target_area: int = 640 * 480,
    prefer_at_least_target: bool = False,
) -> list[_RealSenseVideoProfile]:
    return sorted(
        profiles,
        key=lambda profile: (
            profile.fps != preferred_fps,
            (
                prefer_at_least_target
                and profile.width * profile.height < target_area
            ),
            abs((profile.width * profile.height) - target_area),
            profile.width * profile.height,
            profile.width,
            profile.height,
            profile.fps,
        ),
    )


def _format_realsense_profiles(profiles) -> str:
    items = _sort_realsense_profiles(list(profiles), preferred_fps=30)
    if not items:
        return "none"
    return ", ".join(
        f"{profile.width}x{profile.height}@{profile.fps}"
        for profile in items[:12]
    )


class RealSensePointCloudReader(CameraPointCloudReader):
    """Reader for one Intel RealSense depth camera."""

    def __init__(self, camera: HardwareCameraConfig) -> None:
        self._camera = camera
        self._rs = None
        self._pipeline = None
        self._profile = None
        self._align = None
        self._pointcloud = None
        self._latest_color_jpeg: bytes | None = None
        self._latest_color_rgb: np.ndarray | None = None
        self._latest_depth_meters: np.ndarray | None = None
        self._latest_frame_timestamp: float | None = None
        self._latest_frame_number: int | None = None
        self._depth_scale = 0.001
        self._consecutive_timeouts = 0
        self._active_color_width = camera.color_width or camera.width
        self._active_color_height = camera.color_height or camera.height
        self._active_color_fps = camera.color_fps or camera.fps

    async def start(self) -> None:
        try:
            rs = importlib.import_module("pyrealsense2")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "RealSense camera configured but pyrealsense2 is not installed. "
                "Install Intel RealSense SDK Python bindings on the Linux capture host."
            ) from exc
        await asyncio.to_thread(self._start_blocking, rs)

    def _start_blocking(self, rs) -> None:
        pipeline = rs.pipeline()
        config = rs.config()
        if self._camera.serial:
            config.enable_device(self._camera.serial)
        depth_profile, color_profile = _resolve_realsense_stream_profiles(
            rs,
            self._camera,
        )
        self._active_color_width = color_profile.width
        self._active_color_height = color_profile.height
        self._active_color_fps = color_profile.fps
        config.enable_stream(
            rs.stream.depth,
            depth_profile.width,
            depth_profile.height,
            rs.format.z16,
            depth_profile.fps,
        )
        config.enable_stream(
            rs.stream.color,
            color_profile.width,
            color_profile.height,
            rs.format.rgb8,
            color_profile.fps,
        )
        try:
            self._profile = pipeline.start(config)
        except RuntimeError as exc:
            raise RuntimeError(
                f"failed to start RealSense {self._camera.name!r} with "
                f"depth {depth_profile.width}x{depth_profile.height}@{depth_profile.fps} "
                f"and color {color_profile.width}x{color_profile.height}@{color_profile.fps}: "
                f"{exc}"
            ) from exc
        self._rs = rs
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)
        self._pointcloud = rs.pointcloud()
        self._consecutive_timeouts = 0
        try:
            self._depth_scale = float(
                self._profile.get_device().first_depth_sensor().get_depth_scale()
            )
        except Exception:
            self._depth_scale = 0.001

    async def stop(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        self._profile = None
        if pipeline is not None:
            await asyncio.to_thread(pipeline.stop)

    async def grab_camera_frame(self) -> Optional[PointCloudFrame]:
        if self._pipeline is None or self._rs is None:
            return None
        return await asyncio.to_thread(self._grab_blocking)

    def _grab_blocking(self) -> Optional[PointCloudFrame]:
        frames = self._wait_for_frames()
        if frames is None:
            return self._handle_frame_timeout("no frames returned before timeout")
        if self._align is not None:
            frames = self._align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            return None

        self._consecutive_timeouts = 0
        self._pointcloud.map_to(color)
        points_obj = self._pointcloud.calculate(depth)
        vertices = np.asanyarray(points_obj.get_vertices()).view(np.float32).reshape(-1, 3)
        color_image = np.asanyarray(color.get_data())
        depth_image = np.asanyarray(depth.get_data()).astype(np.float32) * float(self._depth_scale)
        self._latest_color_rgb = color_image.copy()
        self._latest_color_jpeg = _encode_rgb_jpeg(color_image)
        self._latest_depth_meters = depth_image.copy()
        self._latest_frame_timestamp = _frame_timestamp(depth)
        self._latest_frame_number = _frame_number(depth)
        colors = color_image.reshape(-1, 3)

        points, colors = _filter_camera_points(
            vertices,
            colors,
            z_min=self._camera.z_min,
            z_max=self._camera.z_max,
            downsample=self._camera.downsample,
        )
        return PointCloudFrame(
            points=points,
            colors=colors,
            timestamp=time.monotonic(),
        )

    def _wait_for_frames(self):
        timeout_ms = int(self._camera.frame_timeout_ms)
        try_wait = getattr(self._pipeline, "try_wait_for_frames", None)
        try:
            if callable(try_wait):
                result = try_wait(timeout_ms)
                if isinstance(result, tuple):
                    success, frames = result
                    return frames if success else None
                return result or None
            return self._pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as exc:
            message = str(exc)
            if _is_realsense_frame_timeout(message):
                return None
            raise

    def _handle_frame_timeout(self, reason: str) -> Optional[PointCloudFrame]:
        self._consecutive_timeouts += 1
        print(
            f"[pointcloud] RealSense {self._camera.name} frame timeout "
            f"({self._consecutive_timeouts}/"
            f"{self._camera.restart_after_timeouts}): {reason}"
        )
        if self._consecutive_timeouts >= self._camera.restart_after_timeouts:
            self._restart_blocking()
        return None

    def _restart_blocking(self) -> None:
        rs = self._rs
        if rs is None:
            return
        pipeline = self._pipeline
        self._pipeline = None
        self._profile = None
        self._align = None
        self._pointcloud = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                print(
                    f"[pointcloud] RealSense {self._camera.name} stop before "
                    f"restart failed: {exc!r}"
                )
        print(f"[pointcloud] restarting RealSense {self._camera.name} pipeline")
        time.sleep(0.2)
        self._start_blocking(rs)

    def latest_color_jpeg(self) -> bytes | None:
        return self._latest_color_jpeg

    def latest_color_rgb(self) -> np.ndarray | None:
        return None if self._latest_color_rgb is None else self._latest_color_rgb.copy()

    def latest_calibration_frame(self) -> CalibrationCameraFrame | None:
        if self._latest_color_rgb is None or self._latest_depth_meters is None:
            return None
        descriptor = self.descriptor()
        if descriptor is None:
            return None
        return CalibrationCameraFrame(
            image_rgb=self._latest_color_rgb.copy(),
            depth_meters=self._latest_depth_meters.copy(),
            descriptor=descriptor,
            timestamp=self._latest_frame_timestamp,
            frame_number=self._latest_frame_number,
        )

    def descriptor(self) -> CameraDescriptor | None:
        if self._profile is None or self._rs is None:
            return None
        stream = self._profile.get_stream(self._rs.stream.color)
        intr = stream.as_video_stream_profile().get_intrinsics()
        return CameraDescriptor(
            name=self._camera.name,
            camera_type="realsense",
            serial=self._camera.serial,
            width=self._active_color_width,
            height=self._active_color_height,
            fps=self._active_color_fps,
            camera_matrix=[
                [float(intr.fx), 0.0, float(intr.ppx)],
                [0.0, float(intr.fy), float(intr.ppy)],
                [0.0, 0.0, 1.0],
            ],
            distortion=[float(v) for v in getattr(intr, "coeffs", [])],
        )


class Zed2iPointCloudReader(CameraPointCloudReader):
    """Reader for one Stereolabs ZED 2i camera."""

    def __init__(self, camera: HardwareCameraConfig) -> None:
        self._camera = camera
        self._sl = None
        self._zed = None
        self._mat = None
        self._image_mat = None
        self._latest_color_rgb: np.ndarray | None = None
        self._latest_color_jpeg: bytes | None = None

    async def start(self) -> None:
        try:
            sl = importlib.import_module("pyzed.sl")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ZED 2i camera configured but pyzed.sl is not installed. "
                "Install the Stereolabs ZED SDK Python API on the Linux capture host."
            ) from exc
        await asyncio.to_thread(self._start_blocking, sl)

    def _start_blocking(self, sl) -> None:
        init = sl.InitParameters()
        if self._camera.serial:
            init.set_from_serial_number(int(self._camera.serial))
        init.camera_resolution = _zed_enum(sl.RESOLUTION, self._camera.resolution, "HD720")
        init.camera_fps = self._camera.fps
        init.depth_mode = _zed_enum(sl.DEPTH_MODE, self._camera.depth_mode, "PERFORMANCE")
        init.coordinate_units = sl.UNIT.METER

        zed = sl.Camera()
        status = zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"failed to open ZED 2i {self._camera.name!r}: {status}")
        self._sl = sl
        self._zed = zed
        self._mat = sl.Mat()
        self._image_mat = sl.Mat()

    async def stop(self) -> None:
        zed = self._zed
        self._zed = None
        if zed is not None:
            await asyncio.to_thread(zed.close)

    async def grab_camera_frame(self) -> Optional[PointCloudFrame]:
        if self._zed is None or self._sl is None or self._mat is None:
            return None
        return await asyncio.to_thread(self._grab_blocking)

    def _grab_blocking(self) -> Optional[PointCloudFrame]:
        runtime = self._sl.RuntimeParameters()
        status = self._zed.grab(runtime)
        if status != self._sl.ERROR_CODE.SUCCESS:
            return None
        if self._image_mat is not None:
            self._zed.retrieve_image(self._image_mat, self._sl.VIEW.LEFT)
            image = np.asarray(self._image_mat.get_data())
            if image.ndim == 3 and image.shape[2] >= 3:
                self._latest_color_rgb = image[:, :, 2::-1].copy()
                self._latest_color_jpeg = _encode_rgb_jpeg(self._latest_color_rgb)
        self._zed.retrieve_measure(self._mat, self._sl.MEASURE.XYZRGBA)
        data = np.asarray(self._mat.get_data())
        flat = data.reshape(-1, data.shape[-1])
        points = flat[:, :3].astype(np.float32, copy=False)
        colors = _decode_zed_rgba(flat[:, 3])
        points, colors = _filter_camera_points(
            points,
            colors,
            z_min=self._camera.z_min,
            z_max=self._camera.z_max,
            downsample=self._camera.downsample,
        )
        return PointCloudFrame(
            points=points,
            colors=colors,
            timestamp=time.monotonic(),
        )

    def latest_color_jpeg(self) -> bytes | None:
        return self._latest_color_jpeg

    def latest_color_rgb(self) -> np.ndarray | None:
        return None if self._latest_color_rgb is None else self._latest_color_rgb.copy()

    def descriptor(self) -> CameraDescriptor | None:
        if self._zed is None:
            return None
        info = self._zed.get_camera_information()
        calib = info.camera_configuration.calibration_parameters.left_cam
        width = int(getattr(calib, "image_size", (0, 0))[0] or self._camera.width)
        height = int(getattr(calib, "image_size", (0, 0))[1] or self._camera.height)
        return CameraDescriptor(
            name=self._camera.name,
            camera_type="zed2i",
            serial=self._camera.serial,
            width=width,
            height=height,
            fps=self._camera.fps,
            camera_matrix=[
                [float(calib.fx), 0.0, float(calib.cx)],
                [0.0, float(calib.fy), float(calib.cy)],
                [0.0, 0.0, 1.0],
            ],
            distortion=[float(v) for v in getattr(calib, "disto", [])],
            resolution=self._camera.resolution,
            depth_mode=self._camera.depth_mode,
        )


def _filter_camera_points(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    z_min: float,
    z_max: float,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    mask = (
        np.isfinite(points).all(axis=1)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )
    points = points[mask]
    colors = colors[mask]
    if downsample > 1 and points.shape[0] > 0:
        points = points[::downsample]
        colors = colors[::downsample]
    return points.astype(np.float32, copy=False), colors.astype(np.uint8, copy=False)


def _frame_timestamp(frame) -> float | None:
    getter = getattr(frame, "get_timestamp", None)
    if not callable(getter):
        return None
    try:
        return float(getter())
    except Exception:
        return None


def _frame_number(frame) -> int | None:
    getter = getattr(frame, "get_frame_number", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except Exception:
        return None


def _zed_enum(enum_cls, value: str | None, default: str):
    name = (value or default).upper()
    try:
        return getattr(enum_cls, name)
    except AttributeError as exc:
        raise ValueError(f"unknown ZED enum value {name!r}") from exc


def _decode_zed_rgba(values: np.ndarray) -> np.ndarray:
    rgba = np.asarray(values)
    if rgba.dtype.kind == "f":
        packed = rgba.astype(np.float32, copy=False).view(np.uint32)
    else:
        packed = rgba.astype(np.uint32, copy=False)
    r = (packed & 0xFF).astype(np.uint8)
    g = ((packed >> 8) & 0xFF).astype(np.uint8)
    b = ((packed >> 16) & 0xFF).astype(np.uint8)
    return np.stack([r, g, b], axis=1)
