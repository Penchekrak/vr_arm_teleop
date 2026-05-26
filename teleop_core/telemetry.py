"""Shared read-only telemetry cache for teleop and dashboard clients."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Any, Callable

from .point_cloud import PointCloudSource, encode_frame
from .robot import RobotDriver, RobotState
from .workspace import Workspace


@dataclass(frozen=True)
class EncodedPointCloud:
    """Encoded point-cloud payload plus fanout metadata."""

    sequence: int
    payload: bytes
    timestamp: float
    n_points: int


def _vec(values) -> list[float]:
    return [float(v) for v in values]


def _pose_dict(position, orientation, timestamp: float) -> dict[str, Any]:
    return {
        "position": _vec(position),
        "orientation": _vec(orientation),
        "timestamp": float(timestamp),
    }


def _dashboard_camera_feeds(source: PointCloudSource) -> list[dict[str, Any]]:
    feeds = getattr(source, "dashboard_camera_feeds", None)
    if not callable(feeds):
        return []
    raw_feeds = feeds()
    clean: list[dict[str, Any]] = []
    for raw in raw_feeds:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        urdf_link = raw.get("urdf_link")
        url = raw.get("url")
        if not name or not urdf_link or not url:
            continue
        feed = {
            "name": str(name),
            "urdf_link": str(urdf_link),
            "url": str(url),
            "width": int(raw.get("width", 0)),
            "height": int(raw.get("height", 0)),
        }
        if raw.get("calibration_url") is not None:
            feed["calibration_url"] = str(raw["calibration_url"])
        clean.append(feed)
    return clean


def _dashboard_pointcloud_frame(source: PointCloudSource) -> str | None:
    frame = getattr(source, "dashboard_pointcloud_frame", None)
    if not callable(frame):
        return None
    value = frame()
    if value is None:
        return None
    return str(value)


class TelemetryHub:
    """Caches live setup telemetry and fans it out to read-only clients."""

    def __init__(
        self,
        *,
        point_cloud_source: PointCloudSource,
        robot_driver: RobotDriver,
        workspace: Workspace,
        urdf_url: str,
        urdf_assets_url: str,
        urdf_path: str | None = None,
        pointcloud_hz: float = 15.0,
        robot_hz: float = 30.0,
        status_hz: float = 1.0,
        calibration_snapshot_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._pc = point_cloud_source
        self._robot = robot_driver
        self._workspace = workspace
        self._urdf_url = urdf_url
        self._urdf_assets_url = urdf_assets_url
        self._urdf_path = urdf_path
        self._pointcloud_hz = float(pointcloud_hz)
        self._robot_hz = float(robot_hz)
        self._status_hz = float(status_hz)
        self._calibration_snapshot_provider = calibration_snapshot_provider
        self._robot_state: RobotState | None = None
        self._robot_error: str | None = None
        self._pointcloud: EncodedPointCloud | None = None
        self._pointcloud_error: str | None = None
        self._pointcloud_sequence = 0
        self._cloud_condition = asyncio.Condition()
        self._xr_pose: dict[str, Any] | None = None
        self._anchor: dict[str, Any] | None = None
        self._tasks: list[asyncio.Task] = []
        self._stopping = False

    async def start(self) -> None:
        if self._tasks:
            return
        self._stopping = False
        self._tasks = [
            asyncio.create_task(self._robot_loop(), name="telemetry_robot_loop"),
            asyncio.create_task(
                self._pointcloud_loop(),
                name="telemetry_pointcloud_loop",
            ),
        ]

    async def stop(self) -> None:
        self._stopping = True
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []

    async def sample_robot_once(self) -> None:
        try:
            self._robot_state = await self._robot.get_state()
            self._robot_error = None
        except Exception as exc:
            self._robot_error = repr(exc)

    async def sample_pointcloud_once(self) -> None:
        try:
            frame = await self._pc.grab()
            self._pointcloud_error = None
        except Exception as exc:
            frame = None
            self._pointcloud_error = repr(exc)
        if frame is None:
            return
        payload = encode_frame(frame)
        async with self._cloud_condition:
            self._pointcloud_sequence += 1
            self._pointcloud = EncodedPointCloud(
                sequence=self._pointcloud_sequence,
                payload=payload,
                timestamp=float(frame.timestamp),
                n_points=int(frame.n_points),
            )
            self._cloud_condition.notify_all()

    async def wait_for_pointcloud(
        self,
        *,
        after_sequence: int,
        timeout: float,
    ) -> EncodedPointCloud | None:
        async with self._cloud_condition:
            if (
                self._pointcloud is not None
                and self._pointcloud.sequence > after_sequence
            ):
                return self._pointcloud
            try:
                await asyncio.wait_for(
                    self._cloud_condition.wait_for(
                        lambda: (
                            self._pointcloud is not None
                            and self._pointcloud.sequence > after_sequence
                        )
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return None
            return self._pointcloud

    def update_xr_pose(
        self,
        *,
        head_position,
        head_orientation,
        right_wrist_position,
        right_wrist_orientation,
        valid: bool,
        timestamp: float,
        right_wrist_curls=None,
        head_valid: bool | None = None,
    ) -> None:
        if head_valid is None:
            head_valid = valid
        if not head_valid and not valid:
            self._xr_pose = None
            return
        right_wrist = None
        if valid:
            right_wrist = _pose_dict(
                right_wrist_position,
                right_wrist_orientation,
                timestamp,
            )
            if right_wrist_curls is not None:
                right_wrist["curls"] = _vec(right_wrist_curls)
        self._xr_pose = {
            "head": (
                _pose_dict(head_position, head_orientation, timestamp)
                if head_valid
                else None
            ),
            "right_wrist": right_wrist,
        }

    def update_anchor(self, vr_position_of_robot_origin, *, timestamp: float) -> None:
        self._anchor = {
            "vr_position_of_robot_origin": _vec(vr_position_of_robot_origin),
            "timestamp": float(timestamp),
        }

    def snapshot(self) -> dict[str, Any]:
        robot = self._robot_state
        cloud = self._pointcloud
        model = {
            "urdf_url": self._urdf_url,
            "urdf_assets_url": self._urdf_assets_url,
            "camera_feeds": _dashboard_camera_feeds(self._pc),
        }
        if self._urdf_path is not None:
            model["urdf_path"] = self._urdf_path
        pointcloud_frame = _dashboard_pointcloud_frame(self._pc)
        if pointcloud_frame is not None:
            model["pointcloud_frame"] = pointcloud_frame

        snapshot = {
            "type": "snapshot",
            "model": model,
            "workspace": {
                "min": _vec(self._workspace.min_corner),
                "max": _vec(self._workspace.max_corner),
                "frame": self._workspace.frame,
            },
            "robot": {
                "wrist": None if robot is None else _pose_dict(
                    robot.wrist_pose.position,
                    robot.wrist_pose.orientation,
                    robot.timestamp,
                ),
                "joints": {} if robot is None else robot.named_joint_angles,
                "finger_curls": [] if robot is None else _vec(robot.finger_curls),
                "timestamp": None if robot is None else float(robot.timestamp),
                "error": self._robot_error,
            },
            "pointcloud": {
                "sequence": 0 if cloud is None else int(cloud.sequence),
                "timestamp": None if cloud is None else float(cloud.timestamp),
                "n_points": 0 if cloud is None else int(cloud.n_points),
                "error": self._pointcloud_error,
            },
            "xr": {
                "aligned": self._xr_pose is not None and self._anchor is not None,
                "anchor": self._anchor,
                "head": None if self._xr_pose is None else self._xr_pose["head"],
                "right_wrist": (
                    None if self._xr_pose is None else self._xr_pose["right_wrist"]
                ),
            },
            "status": {
                "server_time": time.monotonic(),
                "robot_hz": self._robot_hz,
                "pointcloud_hz": self._pointcloud_hz,
                "status_hz": self._status_hz,
            },
        }
        if self._calibration_snapshot_provider is not None:
            snapshot["calibration"] = self._calibration_snapshot_provider()
        return snapshot

    async def _robot_loop(self) -> None:
        period = 1.0 / max(self._robot_hz, 1e-3)
        while not self._stopping:
            start = time.monotonic()
            await self.sample_robot_once()
            await asyncio.sleep(max(0.0, period - (time.monotonic() - start)))

    async def _pointcloud_loop(self) -> None:
        period = 1.0 / max(self._pointcloud_hz, 1e-3)
        while not self._stopping:
            start = time.monotonic()
            await self.sample_pointcloud_once()
            await asyncio.sleep(max(0.0, period - (time.monotonic() - start)))
