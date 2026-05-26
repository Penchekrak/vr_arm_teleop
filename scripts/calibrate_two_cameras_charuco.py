"""Continuously calibrate configured cameras with a ChArUco board.

The live solver trusts the URDF/FK pose of the arm-mounted anchor camera
in the robot base frame. Every other enabled camera is optimized from
frames where both it and the anchor see the same ChArUco board. Output
matrices use OpenCV optical camera frames and are written as
``world_from_camera`` matrices compatible with the hardware point-cloud
backend.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from aiohttp import WSMsgType, web
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teleop_backends.camera_calibration import (
    AnchoredExtrinsicOptimizer,
    CameraDescriptor,
    CalibrationDetection,
    CharucoBoardSpec,
    JointStateProvider,
    UrdfKinematicTree,
    select_anchor_camera,
    transform_from_rt,
    write_calibrated_hardware_config,
)
from teleop_backends.pointcloud.hardware import (
    CalibrationCameraFrame,
    HardwareCameraConfig,
    HardwarePointCloudConfig,
    CameraPointCloudReader,
    default_reader_factory,
    fuse_camera_frames,
    load_hardware_config,
)
from teleop_backends.robot.rc5_state import (
    RC5_ARM_JOINT_NAMES,
    RC5JointStateReader,
)
from teleop_core.point_cloud import PointCloudFrame
from teleop_core.robot import RobotState
from teleop_core.server import _resolve_robot_asset_path
from teleop_core.telemetry import TelemetryHub
from teleop_core.types import Pose
from teleop_core.workspace import Workspace


@dataclass(frozen=True)
class _Detection:
    accepted: bool
    reason: str | None
    camera_from_board: np.ndarray | None
    reprojection_error: float | None
    marker_count: int
    charuco_corner_count: int
    depth_valid_corners: int
    kabsch_rms_m: float | None
    depth_range_m: tuple[float, float] | None
    frame_timestamp: float | None
    frame_number: int | None
    overlay_rgb: np.ndarray

    @property
    def corner_count(self) -> int:
        return self.charuco_corner_count


class JsonJointStateProvider(JointStateProvider):
    """Development provider for a fixed or externally refreshed joint JSON."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def read_joint_state(self) -> dict[str, float]:
        data = json.loads(self._path.read_text())
        if not isinstance(data, dict):
            raise ValueError("joint state JSON must be an object keyed by URDF joint name")
        return {str(name): float(value) for name, value in data.items()}


class CalibrationPointCloudSource:
    """Captures all cameras, optimizes extrinsics, and emits world clouds."""

    def __init__(
        self,
        *,
        config_path: Path,
        config: HardwarePointCloudConfig,
        anchor: HardwareCameraConfig,
        optimizer: AnchoredExtrinsicOptimizer,
        joint_provider: JointStateProvider,
        kinematic_tree: UrdfKinematicTree,
        base_link: str,
        camera_link: str,
        board,
        cv2,
        min_corners: int,
        min_depth_corners: int,
        depth_neighborhood: int,
        max_kabsch_rms_m: float,
        max_arm_motion_per_sample_m: float,
        autosave: bool,
        reader_factory=default_reader_factory,
    ) -> None:
        self._config_path = Path(config_path)
        self._config = config
        self._anchor = anchor
        self._optimizer = optimizer
        self._joint_provider = joint_provider
        self._kinematic_tree = kinematic_tree
        self._base_link = base_link
        self._camera_link = camera_link
        self._board = board
        self._cv2 = cv2
        self._min_corners = int(min_corners)
        self._min_depth_corners = int(min_depth_corners)
        self._depth_neighborhood = int(depth_neighborhood)
        self._max_kabsch_rms_m = float(max_kabsch_rms_m)
        self._max_arm_motion_per_sample_m = float(max_arm_motion_per_sample_m)
        self._autosave = bool(autosave)
        self._reader_factory = reader_factory
        self._readers: list[tuple[HardwareCameraConfig, CameraPointCloudReader]] = []
        self._descriptors: dict[str, CameraDescriptor] = {}
        self.latest_joint_state: dict[str, float] = {}
        self._latest_detections: dict[str, dict] = {}
        self._latest_diagnostics: dict[str, dict] = {}
        self._latest_calibration_jpegs: dict[str, bytes] = {}
        self._latest_transforms: dict[str, list[list[float]]] = {}
        self._latest_board_poses: dict[str, list[list[float]]] = {}
        self._latest_error: str | None = None
        self._previous_world_from_anchor: np.ndarray | None = None
        self._latest_arm_motion_m: float | None = None
        self._latest_sample_rejected_reason: str | None = None
        self._autosave_state = "disabled" if not autosave else "waiting_for_stability"
        self._last_saved_timestamp: float | None = None
        self._saved_current_stable_set = False

    async def start(self) -> None:
        if self._readers:
            return
        self._readers = [
            (camera, self._reader_factory(camera))
            for camera in self._config.cameras
        ]
        for _, reader in self._readers:
            await reader.start()
        self._descriptors = {}
        for camera, reader in self._readers:
            descriptor = reader.descriptor()
            if descriptor is not None:
                self._descriptors[camera.name] = descriptor

    async def stop(self) -> None:
        for _, reader in reversed(self._readers):
            with contextlib.suppress(Exception):
                await reader.stop()
        self._readers = []

    async def grab(self) -> Optional[PointCloudFrame]:
        if not self._readers:
            return None

        raw_frames: list[tuple[HardwareCameraConfig, PointCloudFrame]] = []
        detections: dict[str, CalibrationDetection] = {}
        detection_status: dict[str, dict] = {}
        diagnostics: dict[str, dict] = {}

        try:
            joint_state = await asyncio.to_thread(self._joint_provider.read_joint_state)
            self.latest_joint_state = joint_state
            world_from_anchor = self._kinematic_tree.transform(
                self._base_link,
                self._camera_link,
                joint_state,
            )
            self._latest_arm_motion_m = _translation_delta(
                self._previous_world_from_anchor,
                world_from_anchor,
            )
            self._previous_world_from_anchor = world_from_anchor.copy()
            self._latest_error = None
        except Exception as exc:
            self._latest_error = repr(exc)
            return None

        for camera, reader in self._readers:
            frame = await reader.grab_camera_frame()
            if frame is not None:
                raw_frames.append((camera, frame))
            calibration_frame = reader.latest_calibration_frame()
            if calibration_frame is None:
                status = {
                    "detected": False,
                    "reason": "calibration_frame_unavailable",
                }
                detection_status[camera.name] = status
                diagnostics[camera.name] = status
                continue
            detection = _detect_board_pose(
                self._cv2,
                self._board,
                calibration_frame,
                min_corners=self._min_corners,
                min_depth_corners=self._min_depth_corners,
                depth_neighborhood=self._depth_neighborhood,
                max_kabsch_rms_m=self._max_kabsch_rms_m,
            )
            jpeg = _encode_rgb_jpeg(self._cv2, detection.overlay_rgb)
            if jpeg is not None:
                self._latest_calibration_jpegs[camera.name] = jpeg
            status = _detection_status(detection)
            detection_status[camera.name] = status
            diagnostics[camera.name] = status
            if not detection.accepted or detection.camera_from_board is None:
                continue
            detections[camera.name] = CalibrationDetection(
                camera_from_board=detection.camera_from_board,
                reprojection_error=float(detection.reprojection_error or 0.0),
                corner_count=detection.corner_count,
            )

        self._latest_detections = detection_status
        self._latest_diagnostics = diagnostics
        self._latest_sample_rejected_reason = _arm_motion_rejection_reason(
            self._latest_arm_motion_m,
            self._max_arm_motion_per_sample_m,
        )
        if self._latest_sample_rejected_reason is None:
            self._optimizer.add_frame(
                world_from_anchor_camera=world_from_anchor,
                detections=detections,
                timestamp=time.monotonic(),
            )
        self._maybe_autosave()

        transforms = self._optimizer.current_transforms(include_anchor=True)
        self._latest_transforms = {
            name: transform.tolist()
            for name, transform in transforms.items()
        }
        self._latest_board_poses = {
            name: (transforms[name] @ detection.camera_from_board).tolist()
            for name, detection in detections.items()
            if name in transforms
        }
        transformed_frames = [
            (
                replace(
                    camera,
                    world_from_camera=np.asarray(
                        transforms.get(camera.name, camera.world_from_camera),
                        dtype=np.float32,
                    ),
                    calibrated=True,
                ),
                frame,
            )
            for camera, frame in raw_frames
        ]
        return fuse_camera_frames(
            transformed_frames,
            workspace_crop=self._config.workspace_crop,
            max_points=self._config.max_points,
        )

    def dashboard_pointcloud_frame(self) -> str:
        return "world"

    def dashboard_camera_feeds(self) -> list[dict[str, object]]:
        feeds = []
        for camera, _ in self._readers:
            if not camera.urdf_link:
                continue
            feeds.append({
                "name": camera.name,
                "urdf_link": camera.urdf_link,
                "url": f"/api/cameras/{quote(camera.name, safe='')}/color.jpg",
                "calibration_url": (
                    f"/api/cameras/{quote(camera.name, safe='')}/calibration.jpg"
                ),
                "width": int(camera.width),
                "height": int(camera.height),
            })
        return feeds

    def latest_color_jpeg(self, camera_name: str) -> bytes | None:
        for camera, reader in self._readers:
            if camera.name == camera_name:
                return reader.latest_color_jpeg()
        return None

    def latest_calibration_jpeg(self, camera_name: str) -> bytes | None:
        return self._latest_calibration_jpegs.get(camera_name)

    def calibration_snapshot(self) -> dict:
        status = self._optimizer.status()
        cameras = [
            {
                "name": camera.name,
                "type": camera.camera_type,
                "serial": camera.serial,
                "urdf_link": camera.urdf_link,
                "anchor": camera.name == self._anchor.name,
                "detection": self._latest_detections.get(camera.name, {
                    "detected": False,
                    "reason": "not_sampled",
                }),
            }
            for camera in self._config.cameras
        ]
        return {
            **status,
            "cameras": cameras,
            "world_from_camera": self._latest_transforms,
            "world_from_board": self._latest_board_poses,
            "diagnostics": self._latest_diagnostics,
            "arm_motion_m": self._latest_arm_motion_m,
            "sample_rejected_reason": self._latest_sample_rejected_reason,
            "autosave": {
                "enabled": self._autosave,
                "state": self._autosave_state,
                "last_saved_timestamp": self._last_saved_timestamp,
            },
            "error": self._latest_error,
        }

    def _maybe_autosave(self) -> None:
        if not self._autosave:
            return
        status = self._optimizer.status()
        if not status["all_stable"]:
            self._autosave_state = "waiting_for_stability"
            self._saved_current_stable_set = False
            return
        if self._saved_current_stable_set:
            self._autosave_state = "saved"
            return
        write_calibrated_hardware_config(
            self._config_path,
            self._optimizer.stable_transforms(include_anchor=True),
            backup=True,
        )
        self._last_saved_timestamp = time.monotonic()
        self._autosave_state = "saved"
        self._saved_current_stable_set = True


class CalibrationRobotAdapter:
    """Tiny robot-state adapter so the shared dashboard can pose the URDF."""

    def __init__(self, source: CalibrationPointCloudSource) -> None:
        self._source = source

    async def get_state(self) -> RobotState:
        joint_state = self._source.latest_joint_state
        names = [
            name for name in RC5_ARM_JOINT_NAMES
            if name in joint_state
        ]
        names.extend(name for name in sorted(joint_state) if name not in names)
        return RobotState(
            wrist_pose=Pose.identity(frame="world"),
            joint_angles=np.asarray([joint_state[name] for name in names], dtype=np.float32),
            finger_curls=np.zeros(5, dtype=np.float32),
            timestamp=time.monotonic(),
            joint_names=tuple(names),
        )


def _default_urdf() -> Path:
    return REPO_ROOT / "urdf_rc5_right_hand" / "urdf_with_simple_collisions.urdf"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cameras", type=Path, required=True,
                    help="hardware camera config JSON to open and autosave")
    ap.add_argument("--anchor-camera", default=None,
                    help="FK-trusted camera name; defaults to --camera-link match")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--dashboard-port", type=int, default=8001)

    ap.add_argument("--urdf", type=Path, default=_default_urdf())
    ap.add_argument("--base-link", default="world")
    ap.add_argument("--camera-link", default="d405_depth_optical_frame")
    ap.add_argument("--arm-ip", default="10.10.10.10")
    ap.add_argument("--rc5-api-path", type=Path, default=None)
    ap.add_argument(
        "--joint-state-json",
        type=Path,
        default=None,
        help="offline/live joint JSON; omit to read live RC5 joints",
    )

    ap.add_argument("--squares-x", type=int, required=True)
    ap.add_argument("--squares-y", type=int, required=True)
    ap.add_argument("--square-length", type=float, required=True)
    ap.add_argument("--marker-length", type=float, required=True)
    ap.add_argument("--dictionary", default="DICT_5X5_100")

    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--min-corners", type=int, default=12)
    ap.add_argument("--min-depth-corners", type=int, default=12)
    ap.add_argument("--max-reprojection-error", type=float, default=2.0)
    ap.add_argument("--max-kabsch-rms", type=float, default=0.01)
    ap.add_argument("--depth-neighborhood", type=int, default=2)
    ap.add_argument("--max-arm-motion-per-sample", type=float, default=0.02)
    ap.add_argument("--rolling-window", type=int, default=40)
    ap.add_argument("--stability-seconds", type=float, default=2.0)
    ap.add_argument("--max-pair-translation-deviation", type=float, default=0.05)
    ap.add_argument("--max-pair-rotation-deviation", type=float, default=5.0)
    ap.add_argument("--translation-stability", type=float, default=0.01)
    ap.add_argument("--rotation-stability", type=float, default=1.0)
    ap.add_argument(
        "--autosave",
        dest="autosave",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="atomically autosave stable all-camera solutions",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(run_continuous_calibration(args))
    except KeyboardInterrupt:
        pass


async def run_continuous_calibration(args: argparse.Namespace) -> None:
    cv2 = _require_cv2_aruco()
    board_spec = CharucoBoardSpec(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dictionary=args.dictionary,
    )
    board = board_spec.create_cv2_board()

    hardware_config = load_hardware_config(args.cameras)
    anchor = select_anchor_camera(
        hardware_config,
        anchor_name=args.anchor_camera,
        camera_link=args.camera_link,
    )
    target_names = tuple(
        camera.name for camera in hardware_config.cameras
        if camera.name != anchor.name
    )
    if not target_names:
        raise SystemExit("continuous calibration requires at least one non-anchor camera")

    optimizer = AnchoredExtrinsicOptimizer(
        anchor_name=anchor.name,
        target_names=target_names,
        min_samples=args.min_samples,
        rolling_window=args.rolling_window,
        min_corners=args.min_corners,
        max_reprojection_error_px=args.max_reprojection_error,
        max_pair_translation_deviation_m=args.max_pair_translation_deviation,
        max_pair_rotation_deviation_deg=args.max_pair_rotation_deviation,
        translation_stability_m=args.translation_stability,
        rotation_stability_deg=args.rotation_stability,
        stability_seconds=args.stability_seconds,
    )
    source = CalibrationPointCloudSource(
        config_path=args.cameras,
        config=hardware_config,
        anchor=anchor,
        optimizer=optimizer,
        joint_provider=_make_joint_state_provider(args),
        kinematic_tree=UrdfKinematicTree.from_file(args.urdf),
        base_link=args.base_link,
        camera_link=args.camera_link,
        board=board,
        cv2=cv2,
        min_corners=args.min_corners,
        min_depth_corners=args.min_depth_corners,
        depth_neighborhood=args.depth_neighborhood,
        max_kabsch_rms_m=args.max_kabsch_rms,
        max_arm_motion_per_sample_m=args.max_arm_motion_per_sample,
        autosave=args.autosave,
    )
    workspace = _workspace_from_config(hardware_config)
    robot = CalibrationRobotAdapter(source)
    hub = TelemetryHub(
        point_cloud_source=source,
        robot_driver=robot,
        workspace=workspace,
        urdf_url="/robot/robot.urdf",
        urdf_assets_url="/robot/assets/",
        calibration_snapshot_provider=source.calibration_snapshot,
    )

    await source.start()
    await hub.start()
    app = _make_dashboard_app(
        hub=hub,
        source=source,
        static_dir=REPO_ROOT / "webxr_app" / "dashboard_static",
        urdf_path=args.urdf,
        robot_assets_root=args.urdf.parent,
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.dashboard_port)
    try:
        await site.start()
        print(f"[calib] anchor camera: {anchor.name}")
        print(f"[calib] target cameras: {', '.join(target_names)}")
        print(f"[calib] dashboard on http://{args.host}:{args.dashboard_port}")
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        await hub.stop()
        await source.stop()


def _make_joint_state_provider(args: argparse.Namespace) -> JointStateProvider:
    if args.joint_state_json is not None:
        return JsonJointStateProvider(args.joint_state_json)
    return RC5JointStateReader(
        arm_ip=args.arm_ip,
        rc5_api_path=args.rc5_api_path,
    )


def _workspace_from_config(config: HardwarePointCloudConfig) -> Workspace:
    if config.workspace_crop is not None:
        min_corner, max_corner = config.workspace_crop
        return Workspace(
            min_corner=np.asarray(min_corner, dtype=np.float32),
            max_corner=np.asarray(max_corner, dtype=np.float32),
        )
    return Workspace(
        min_corner=np.array([-0.5, -0.5, 0.0], dtype=np.float32),
        max_corner=np.array([0.8, 0.8, 1.2], dtype=np.float32),
    )


def _make_dashboard_app(
    *,
    hub: TelemetryHub,
    source: CalibrationPointCloudSource,
    static_dir: Path,
    urdf_path: Path,
    robot_assets_root: Path,
) -> web.Application:
    app = web.Application()

    async def snapshot(_request) -> web.Response:
        return web.json_response(hub.snapshot())

    async def camera_color(request) -> web.Response:
        image = source.latest_color_jpeg(request.match_info["name"])
        if image is None:
            return web.Response(status=404, text="camera frame not ready")
        return web.Response(
            body=image,
            content_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    async def camera_calibration(request) -> web.Response:
        image = source.latest_calibration_jpeg(request.match_info["name"])
        if image is None:
            return web.Response(status=404, text="calibration frame not ready")
        return web.Response(
            body=image,
            content_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    async def robot_asset(request) -> web.StreamResponse:
        try:
            path = _resolve_robot_asset_path(robot_assets_root, request.match_info["tail"])
        except ValueError:
            return web.Response(status=404)
        if not path.exists() or not path.is_file():
            return web.Response(status=404)
        return web.FileResponse(path)

    async def dashboard_ws(request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps(hub.snapshot()))

        async def json_loop() -> None:
            while not ws.closed:
                try:
                    await ws.send_str(json.dumps(hub.snapshot()))
                except ConnectionResetError:
                    break
                await asyncio.sleep(1.0 / 20.0)

        async def cloud_loop() -> None:
            last_sequence = 0
            while not ws.closed:
                cloud = await hub.wait_for_pointcloud(
                    after_sequence=last_sequence,
                    timeout=1.0,
                )
                if cloud is None:
                    continue
                last_sequence = cloud.sequence
                try:
                    await ws.send_bytes(cloud.payload)
                except ConnectionResetError:
                    break

        tasks = [
            asyncio.create_task(json_loop(), name="calibration_dashboard_json_loop"),
            asyncio.create_task(cloud_loop(), name="calibration_dashboard_cloud_loop"),
        ]
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
                if msg.type == WSMsgType.TEXT:
                    await ws.send_str(json.dumps({
                        "type": "error",
                        "message": "calibration dashboard is read-only",
                    }))
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        return ws

    app.router.add_get("/ws", dashboard_ws)
    app.router.add_get("/api/snapshot", snapshot)
    app.router.add_get("/api/cameras/{name}/color.jpg", camera_color)
    app.router.add_get("/api/cameras/{name}/calibration.jpg", camera_calibration)
    app.router.add_get("/robot/robot.urdf", lambda _request: web.FileResponse(urdf_path))
    app.router.add_get("/robot/assets/{tail:.*}", robot_asset)
    app.router.add_get("/", lambda _request: web.FileResponse(static_dir / "index.html"))
    app.router.add_static("/", path=str(static_dir), show_index=False)
    return app


def _detect_board_pose(
    cv2,
    board,
    frame: CalibrationCameraFrame,
    *,
    min_corners: int,
    min_depth_corners: int,
    depth_neighborhood: int,
    max_kabsch_rms_m: float,
) -> _Detection:
    image_rgb = np.asarray(frame.image_rgb, dtype=np.uint8)
    depth_meters = np.asarray(frame.depth_meters, dtype=np.float32)
    descriptor = frame.descriptor
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    camera_matrix = np.asarray(descriptor.camera_matrix, dtype=np.float64)
    distortion = np.asarray(descriptor.distortion, dtype=np.float64)
    marker_corners, marker_ids, charuco_corners, charuco_ids = _detect_charuco_corners(
        cv2,
        board,
        gray,
        camera_matrix,
        distortion,
    )
    overlay = image_rgb.copy()
    marker_count = 0 if marker_ids is None else int(len(marker_ids))
    if marker_ids is None or len(marker_ids) == 0:
        return _detection_result(
            accepted=False,
            reason="markers_not_found",
            overlay_rgb=overlay,
            marker_count=marker_count,
            frame=frame,
        )

    cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
    charuco_count = 0 if charuco_ids is None else int(len(charuco_ids))
    if charuco_ids is not None and charuco_corners is not None and charuco_count > 0:
        cv2.aruco.drawDetectedCornersCharuco(overlay, charuco_corners, charuco_ids)
    if charuco_ids is None or charuco_corners is None or charuco_count < min_corners:
        return _detection_result(
            accepted=False,
            reason="not_enough_charuco_corners",
            overlay_rgb=overlay,
            marker_count=marker_count,
            charuco_corner_count=charuco_count,
            frame=frame,
        )

    if depth_meters.shape[:2] != gray.shape[:2]:
        return _detection_result(
            accepted=False,
            reason="depth_shape_mismatch",
            overlay_rgb=overlay,
            marker_count=marker_count,
            charuco_corner_count=charuco_count,
            frame=frame,
        )

    object_points = _charuco_object_points(board, charuco_ids)
    observed_pixels = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    valid_object_points = []
    valid_camera_points = []
    valid_observed_pixels = []
    valid_depths = []
    for object_point, pixel in zip(object_points, observed_pixels):
        depth = _sample_depth(depth_meters, pixel, depth_neighborhood)
        if depth is None:
            continue
        valid_object_points.append(object_point)
        valid_camera_points.append(_deproject_pixel(pixel, depth, camera_matrix))
        valid_observed_pixels.append(pixel)
        valid_depths.append(depth)

    depth_valid_corners = len(valid_depths)
    depth_range = (
        (float(min(valid_depths)), float(max(valid_depths)))
        if valid_depths
        else None
    )
    if depth_valid_corners < min_depth_corners:
        return _detection_result(
            accepted=False,
            reason="not_enough_valid_depth",
            overlay_rgb=overlay,
            marker_count=marker_count,
            charuco_corner_count=charuco_count,
            depth_valid_corners=depth_valid_corners,
            depth_range_m=depth_range,
            frame=frame,
        )

    camera_from_board, kabsch_rms = _kabsch_transform(
        np.asarray(valid_object_points, dtype=np.float64),
        np.asarray(valid_camera_points, dtype=np.float64),
    )
    rotation = camera_from_board[:3, :3]
    translation = camera_from_board[:3, 3]
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = translation.reshape(3, 1)
    reprojection_error = _object_reprojection_error(
        cv2,
        np.asarray(valid_object_points, dtype=np.float64),
        np.asarray(valid_observed_pixels, dtype=np.float64),
        rvec,
        tvec,
        camera_matrix,
        distortion,
    )
    cv2.drawFrameAxes(overlay, camera_matrix, distortion, rvec, tvec, 0.05)
    if kabsch_rms > max_kabsch_rms_m:
        return _detection_result(
            accepted=False,
            reason="kabsch_rms_too_high",
            camera_from_board=camera_from_board,
            reprojection_error=reprojection_error,
            overlay_rgb=overlay,
            marker_count=marker_count,
            charuco_corner_count=charuco_count,
            depth_valid_corners=depth_valid_corners,
            kabsch_rms_m=kabsch_rms,
            depth_range_m=depth_range,
            frame=frame,
        )
    return _detection_result(
        accepted=True,
        reason=None,
        camera_from_board=camera_from_board,
        reprojection_error=reprojection_error,
        overlay_rgb=overlay,
        marker_count=marker_count,
        charuco_corner_count=charuco_count,
        depth_valid_corners=depth_valid_corners,
        kabsch_rms_m=kabsch_rms,
        depth_range_m=depth_range,
        frame=frame,
    )


def _detect_charuco_corners(cv2, board, gray, camera_matrix, distortion):
    if hasattr(cv2.aruco, "detectMarkers") and hasattr(
        cv2.aruco,
        "interpolateCornersCharuco",
    ):
        dictionary = (
            board.getDictionary()
            if hasattr(board, "getDictionary")
            else board.dictionary
        )
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        if marker_ids is None or len(marker_ids) == 0:
            return marker_corners, marker_ids, None, None
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion,
        )
        return marker_corners, marker_ids, charuco_corners, charuco_ids

    if hasattr(cv2.aruco, "CharucoDetector"):
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
            gray,
        )
        return marker_corners, marker_ids, charuco_corners, charuco_ids

    raise RuntimeError(
        "OpenCV ArUco module lacks both legacy ChArUco functions and "
        "cv2.aruco.CharucoDetector"
    )


def _detection_result(
    *,
    accepted: bool,
    reason: str | None,
    overlay_rgb: np.ndarray,
    marker_count: int,
    frame: CalibrationCameraFrame,
    camera_from_board: np.ndarray | None = None,
    reprojection_error: float | None = None,
    charuco_corner_count: int = 0,
    depth_valid_corners: int = 0,
    kabsch_rms_m: float | None = None,
    depth_range_m: tuple[float, float] | None = None,
) -> _Detection:
    return _Detection(
        accepted=bool(accepted),
        reason=reason,
        camera_from_board=camera_from_board,
        reprojection_error=None if reprojection_error is None else float(reprojection_error),
        marker_count=int(marker_count),
        charuco_corner_count=int(charuco_corner_count),
        depth_valid_corners=int(depth_valid_corners),
        kabsch_rms_m=None if kabsch_rms_m is None else float(kabsch_rms_m),
        depth_range_m=depth_range_m,
        frame_timestamp=frame.timestamp,
        frame_number=frame.frame_number,
        overlay_rgb=overlay_rgb,
    )


def _charuco_object_points(board, charuco_ids) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        chessboard_corners = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    else:
        chessboard_corners = np.asarray(board.chessboardCorners, dtype=np.float64)
    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    return chessboard_corners[ids]


def _sample_depth(
    depth_meters: np.ndarray,
    pixel: np.ndarray,
    neighborhood: int,
) -> float | None:
    x = int(round(float(pixel[0])))
    y = int(round(float(pixel[1])))
    radius = max(0, int(neighborhood))
    height, width = depth_meters.shape[:2]
    if x < 0 or y < 0 or x >= width or y >= height:
        return None
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    values = np.asarray(depth_meters[y0:y1, x0:x1], dtype=np.float32).reshape(-1)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None
    return float(np.median(values))


def _deproject_pixel(pixel: np.ndarray, depth_m: float, camera_matrix: np.ndarray) -> np.ndarray:
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("camera intrinsics require non-zero focal lengths")
    x = (float(pixel[0]) - cx) * float(depth_m) / fx
    y = (float(pixel[1]) - cy) * float(depth_m) / fy
    return np.array([x, y, float(depth_m)], dtype=np.float64)


def _kabsch_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, float]:
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError("Kabsch requires at least three paired 3D points")
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    transformed = (rotation @ source.T).T + translation
    rms = float(np.sqrt(np.mean(np.sum((transformed - target) ** 2, axis=1))))
    return transform_from_rt(rotation, translation), rms


def _object_reprojection_error(
    cv2,
    object_points: np.ndarray,
    observed_pixels: np.ndarray,
    rvec,
    tvec,
    camera_matrix,
    distortion,
) -> float:
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float32),
        rvec,
        tvec,
        camera_matrix,
        distortion,
    )
    projected = projected.reshape(-1, 2)
    observed = np.asarray(observed_pixels, dtype=np.float32).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=1))))


def _detection_status(detection: _Detection) -> dict:
    return {
        "detected": bool(detection.accepted),
        "reason": detection.reason,
        "marker_count": detection.marker_count,
        "corner_count": detection.corner_count,
        "charuco_corner_count": detection.charuco_corner_count,
        "depth_valid_corners": detection.depth_valid_corners,
        "kabsch_rms_m": detection.kabsch_rms_m,
        "depth_range_m": (
            None
            if detection.depth_range_m is None
            else [float(detection.depth_range_m[0]), float(detection.depth_range_m[1])]
        ),
        "reprojection_error_px": detection.reprojection_error,
        "frame_timestamp": detection.frame_timestamp,
        "frame_number": detection.frame_number,
    }


def _translation_delta(
    previous: np.ndarray | None,
    current: np.ndarray,
) -> float | None:
    if previous is None:
        return None
    return float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))


def _arm_motion_rejection_reason(
    arm_motion_m: float | None,
    max_arm_motion_m: float,
) -> str | None:
    if arm_motion_m is None or max_arm_motion_m <= 0.0:
        return None
    if arm_motion_m > max_arm_motion_m:
        return "arm_motion_too_high"
    return None


def _encode_rgb_jpeg(cv2, image: np.ndarray) -> bytes | None:
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    bgr = np.ascontiguousarray(rgb[:, :, :3][:, :, ::-1])
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 82],
    )
    if not ok:
        return None
    return encoded.tobytes()


def _charuco_reprojection_error(
    cv2,
    board,
    charuco_corners,
    charuco_ids,
    rvec,
    tvec,
    camera_matrix,
    distortion,
) -> float:
    if hasattr(board, "getChessboardCorners"):
        chessboard_corners = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    else:
        chessboard_corners = np.asarray(board.chessboardCorners, dtype=np.float32)
    object_points = chessboard_corners[np.asarray(charuco_ids, dtype=np.int32).reshape(-1)]
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    projected = projected.reshape(-1, 2)
    observed = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=1))))


def _require_cv2_aruco():
    try:
        cv2 = importlib.import_module("cv2")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ChArUco calibration requires opencv-contrib-python with cv2.aruco"
        ) from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("installed cv2 lacks aruco; install opencv-contrib-python")
    return cv2


if __name__ == "__main__":
    main()
