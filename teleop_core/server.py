"""Top-level orchestrator -- wires backends into a running aiohttp app.

This class knows nothing about specific cameras or robots; everything
hardware-specific arrives through the injected :class:`PointCloudSource`
and :class:`RobotDriver`.

Phases:
    idle         after-connect, before finger calibration
    finger_cal   walking through the FingerCalibrationFSM
    ready        calibration done, waiting for left-trigger to engage
    tracking     left-trigger held, CartesianTracker active
    fault        a safety event paused us; user must acknowledge

Loops (all asyncio tasks):
    _control_loop      handles inbound WS messages (hand state, buttons)
    _command_loop      sends RobotCommands at ~50 Hz
    _pointcloud_loop   grabs frames + broadcasts on the binary WS
    _safety_loop       runs SafetyMonitor + emits SafetyMsg events
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import ssl
from dataclasses import dataclass
from pathlib import Path

from aiohttp import WSMsgType, web

from .calibration import FingerCalibrationFSM
from .dashboard_control import DashboardControlError, DashboardControlService
from .messages import (
    AnchorMsg, ButtonMsg, HandStateMsg, PhaseMsg, PromptMsg, RobotEchoMsg,
    TriggerMsg, WorkspaceMsg, decode, encode,
)
from .point_cloud import PointCloudSource
from .robot import RobotCommand, RobotDriver
from .safety import SafetyConfig, SafetyMonitor
from .telemetry import TelemetryHub
from .tracking import CartesianTracker
from .types import Pose
from .workspace import Workspace

import time

import numpy as np


@dataclass
class ServerConfig:
    """Static configuration that doesn't change per-connection."""
    host: str = "0.0.0.0"
    port: int = 8000
    dashboard_port: int = 8001
    static_dir: Path = Path(__file__).parent.parent / "webxr_app" / "static"
    dashboard_static_dir: Path = (
        Path(__file__).parent.parent / "webxr_app" / "dashboard_static"
    )
    cert: Path | None = None
    key: Path | None = None
    urdf_path: Path | None = None
    robot_assets_root: Path | None = None
    command_hz: float = 50.0
    pointcloud_hz: float = 15.0
    safety_hz: float = 30.0
    dashboard_robot_hz: float = 30.0
    dashboard_status_hz: float = 1.0


def _resolve_robot_asset_path(root: Path, tail: str) -> Path:
    root_resolved = Path(root).resolve()
    target = (root_resolved / tail).resolve()
    target.relative_to(root_resolved)
    return target


class TeleopServer:
    """Owns one source / one driver / one connection (single-operator design)."""

    def __init__(
        self,
        point_cloud_source: PointCloudSource,
        robot_driver: RobotDriver,
        workspace: Workspace,
        config: ServerConfig,
        safety_config: SafetyConfig | None = None,
        dashboard_grasp_planner=None,
        dashboard_plan_simulator=None,
    ) -> None:
        self._pc = point_cloud_source
        self._robot = robot_driver
        self._workspace = workspace
        self._config = config
        self._safety = SafetyMonitor(safety_config or SafetyConfig())
        self._tracker = CartesianTracker(workspace)
        self._calib = FingerCalibrationFSM()
        self._phase = "idle"
        self._latest_hand = HandStateMsg()
        self._shutdown = asyncio.Event()
        self._last_debug_print = 0.0
        self._telemetry = TelemetryHub(
            point_cloud_source=self._pc,
            robot_driver=self._robot,
            workspace=self._workspace,
            urdf_url="/robot/robot.urdf",
            urdf_assets_url="/robot/assets/",
            urdf_path=(
                None
                if self._config.urdf_path is None
                else str(Path(self._config.urdf_path).resolve())
            ),
            pointcloud_hz=self._config.pointcloud_hz,
            robot_hz=self._config.dashboard_robot_hz,
            status_hz=self._config.dashboard_status_hz,
        )
        self._dashboard_control = DashboardControlService(
            robot_driver=self._robot,
            workspace_getter=lambda: self._workspace,
            workspace_setter=self._set_workspace,
            cube_provider=lambda: self._telemetry.latest_cube_detection,
            grasp_planner=dashboard_grasp_planner,
            plan_simulator=dashboard_plan_simulator,
        )
        self._telemetry.set_control_snapshot_provider(self._dashboard_control.snapshot)

    async def run(self) -> None:
        """Start backends, start aiohttp, block until shutdown."""
        await self._pc.start()
        await self._robot.start()
        await self._telemetry.start()

        teleop_app = self._make_teleop_app()
        dashboard_app = self._make_dashboard_app()
        teleop_runner = web.AppRunner(teleop_app)
        dashboard_runner = web.AppRunner(dashboard_app)
        await teleop_runner.setup()
        await dashboard_runner.setup()

        ssl_context = None
        if self._config.cert and self._config.key:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(self._config.cert, self._config.key)

        teleop_site = web.TCPSite(
            teleop_runner,
            self._config.host,
            self._config.port,
            ssl_context=ssl_context,
        )
        dashboard_site = web.TCPSite(
            dashboard_runner,
            self._config.host,
            self._config.dashboard_port,
            ssl_context=ssl_context,
        )
        try:
            await teleop_site.start()
            await dashboard_site.start()
            scheme = "https" if ssl_context else "http"
            print(f"[teleop] serving on {scheme}://{self._config.host}:{self._config.port}")
            print(
                f"[teleop] dashboard on "
                f"{scheme}://{self._config.host}:{self._config.dashboard_port}"
            )
            await self._shutdown.wait()
        finally:
            await dashboard_runner.cleanup()
            await teleop_runner.cleanup()
            await self._telemetry.stop()
            await self._pc.stop()
            await self._robot.stop()

    def _make_teleop_app(self) -> web.Application:
        static_dir = Path(self._config.static_dir)
        app = web.Application()
        app.router.add_get("/ws", self._handle_ws)
        # Serve index.html at "/" explicitly, then fall through to static
        # files for everything else under the static dir.
        app.router.add_get(
            "/",
            lambda _req: web.FileResponse(static_dir / "index.html"),
        )
        app.router.add_static("/", path=str(static_dir), show_index=False)
        return app

    def _make_dashboard_app(self) -> web.Application:
        static_dir = Path(self._config.dashboard_static_dir)
        app = web.Application()
        app.router.add_get("/ws", self._handle_dashboard_ws)
        app.router.add_get("/api/snapshot", self._handle_dashboard_snapshot)
        app.router.add_post("/api/control/enable", self._handle_dashboard_control_enable)
        app.router.add_post("/api/control/disable", self._handle_dashboard_control_disable)
        app.router.add_post("/api/workspace", self._handle_dashboard_workspace)
        app.router.add_post("/api/grasp/plan", self._handle_dashboard_grasp_plan)
        app.router.add_post("/api/grasp/approve", self._handle_dashboard_grasp_approve)
        app.router.add_get(
            "/api/cameras/{name}/color.jpg",
            self._handle_dashboard_camera_color,
        )
        app.router.add_get("/robot/robot.urdf", self._handle_robot_urdf)
        app.router.add_get("/robot/assets/{tail:.*}", self._handle_robot_asset)
        app.router.add_get(
            "/",
            lambda _req: web.FileResponse(static_dir / "index.html"),
        )
        app.router.add_static("/", path=str(static_dir), show_index=False)
        return app

    async def _handle_dashboard_snapshot(self, _request) -> web.Response:
        return web.json_response(self._telemetry.snapshot())

    async def _handle_dashboard_control_enable(self, _request) -> web.Response:
        self._tracker.disengage()
        return await self._dashboard_control_response(
            lambda: self._dashboard_control.enable(),
        )

    async def _handle_dashboard_control_disable(self, _request) -> web.Response:
        return await self._dashboard_control_response(
            lambda: self._dashboard_control.disable(),
        )

    async def _handle_dashboard_workspace(self, request) -> web.Response:
        try:
            payload = await _read_json_body(request)
        except ValueError as exc:
            return web.json_response({"error": "bad_request", "reason": str(exc)}, status=400)
        return await self._dashboard_control_response(
            lambda: self._dashboard_control.update_workspace(payload),
        )

    async def _handle_dashboard_grasp_plan(self, request) -> web.Response:
        try:
            payload = await _read_json_body(request)
            cube_id = str(payload["cube_id"])
        except (KeyError, ValueError) as exc:
            return web.json_response({"error": "bad_request", "reason": str(exc)}, status=400)
        return await self._dashboard_control_response(
            lambda: self._dashboard_control.plan_grasp(cube_id),
            envelope="pending_plan",
        )

    async def _handle_dashboard_grasp_approve(self, request) -> web.Response:
        try:
            payload = await _read_json_body(request)
            plan_id = str(payload["plan_id"])
        except (KeyError, ValueError) as exc:
            return web.json_response({"error": "bad_request", "reason": str(exc)}, status=400)
        return await self._dashboard_control_response(
            lambda: self._dashboard_control.approve(plan_id),
        )

    async def _dashboard_control_response(self, action, *, envelope: str | None = None) -> web.Response:
        try:
            result = await action()
        except DashboardControlError as exc:
            return web.json_response(exc.as_dict(), status=exc.status)
        if hasattr(result, "as_dict"):
            result = result.as_dict()
        body = result if envelope is None else {envelope: result}
        if isinstance(body, dict) and "control" not in body:
            body = {**body, "control": self._dashboard_control.snapshot()}
        return web.json_response(body)

    async def _handle_dashboard_camera_color(self, request) -> web.Response:
        latest_color_jpeg = getattr(self._pc, "latest_color_jpeg", None)
        if not callable(latest_color_jpeg):
            return web.Response(status=404, text="camera feed unavailable")
        image = latest_color_jpeg(request.match_info["name"])
        if image is None:
            return web.Response(status=404, text="camera frame not ready")
        return web.Response(
            body=image,
            content_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    async def _handle_robot_urdf(self, _request) -> web.StreamResponse:
        if self._config.urdf_path is None:
            return web.Response(status=404, text="URDF not configured")
        return web.FileResponse(
            Path(self._config.urdf_path),
            headers={"Cache-Control": "no-store"},
        )

    async def _handle_robot_asset(self, request) -> web.StreamResponse:
        root = self._config.robot_assets_root
        if root is None:
            return web.Response(status=404, text="robot asset root not configured")
        try:
            path = _resolve_robot_asset_path(Path(root), request.match_info["tail"])
        except ValueError:
            return web.Response(status=404)
        if not path.exists() or not path.is_file():
            return web.Response(status=404)
        return web.FileResponse(path)

    async def _handle_dashboard_ws(self, request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps(self._telemetry.snapshot()))

        async def json_loop() -> None:
            # Snapshot JSON is small and carries robot/XR state, so send it at
            # the dashboard robot rate.
            period = 1.0 / max(self._config.dashboard_robot_hz, 1e-3)
            while not ws.closed:
                try:
                    await ws.send_str(json.dumps(self._telemetry.snapshot()))
                except ConnectionResetError:
                    break
                await asyncio.sleep(period)

        async def cloud_loop() -> None:
            last_sequence = 0
            while not ws.closed:
                cloud = await self._telemetry.wait_for_pointcloud(
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
            asyncio.create_task(json_loop(), name="dashboard_json_loop"),
            asyncio.create_task(cloud_loop(), name="dashboard_cloud_loop"),
        ]
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
                if msg.type == WSMsgType.TEXT:
                    await ws.send_str(json.dumps({
                        "type": "error",
                        "message": "dashboard is read-only",
                    }))
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        return ws

    # ----- inbound channels -----

    async def _handle_ws(self, request) -> web.WebSocketResponse:
        """Main WebSocket handler. One connection -> spawn loops."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Tell the client where we are in the state machine right now.
        await ws.send_str(encode(PhaseMsg(phase=self._phase)))
        # Announce the workspace box so the client can draw the wireframe.
        await ws.send_str(encode(WorkspaceMsg(
            min=tuple(float(v) for v in self._workspace.min_corner),
            max=tuple(float(v) for v in self._workspace.max_corner),
        )))
        # Initial prompt: ask the operator to begin calibration.
        await ws.send_str(encode(PromptMsg(text=self._calib.current_prompt)))

        tasks = [
            asyncio.create_task(self._control_loop(ws), name="control_loop"),
            asyncio.create_task(self._pointcloud_loop(ws), name="pointcloud_loop"),
            asyncio.create_task(self._command_loop(), name="command_loop"),
        ]
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            for t in pending:
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            # Surface any exception from the loop that finished first.
            for t in done:
                exc = t.exception()
                if exc:
                    print(f"[teleop] {t.get_name()} crashed: {exc!r}")
        finally:
            if not ws.closed:
                await ws.close()
        return ws

    async def _control_loop(self, ws: web.WebSocketResponse) -> None:
        """Receive HandStateMsg/ButtonMsg/TriggerMsg; update internal state."""
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    decoded = decode(msg.data)
                except Exception as exc:
                    print(f"[teleop] control decode error: {exc!r}; raw={msg.data!r}")
                    continue
                if isinstance(decoded, HandStateMsg):
                    self._latest_hand = decoded
                    self._telemetry.update_xr_pose(
                        head_position=decoded.head_position,
                        head_orientation=decoded.head_orientation,
                        right_wrist_position=decoded.wrist_position,
                        right_wrist_orientation=decoded.wrist_orientation,
                        valid=decoded.valid,
                        timestamp=time.monotonic(),
                        right_wrist_curls=decoded.curls,
                        head_valid=decoded.head_valid,
                    )
                elif isinstance(decoded, ButtonMsg):
                    await self._on_button(ws, decoded)
                elif isinstance(decoded, TriggerMsg):
                    await self._on_trigger(ws, decoded)
            elif msg.type == WSMsgType.ERROR:
                print(f"[teleop] ws error: {ws.exception()!r}")
                break

    async def _on_button(self, ws: web.WebSocketResponse, btn: ButtonMsg) -> None:
        """Handle a single rising-edge button event."""
        if btn.hand != "left" or not btn.pressed:
            return
        if btn.name == "x_click":
            await self._advance_calibration(ws)
        # 'y_click' (quit) and others -- TBD.

    async def _advance_calibration(self, ws: web.WebSocketResponse) -> None:
        """Drive the finger-calibration FSM from an X click."""
        if self._phase == "idle":
            self._calib.on_start()
            self._phase = "finger_cal"
            await ws.send_str(encode(PhaseMsg(phase=self._phase)))
            await ws.send_str(encode(PromptMsg(text=self._calib.current_prompt)))
            return

        if self._phase == "finger_cal":
            if not self._latest_hand.valid:
                await ws.send_str(encode(PromptMsg(
                    text=self._calib.current_prompt
                         + "\n(no hand tracked -- bring your right hand into view)",
                    severity="warn",
                )))
                return
            self._calib.on_confirm(self._latest_hand.curls, self._latest_hand.abduction)
            if self._calib.is_complete:
                self._phase = "ready"
                rec = self._calib.record
                print(
                    f"[teleop] calibration complete: "
                    f"curl_min={rec.min_curl.tolist()} curl_max={rec.max_curl.tolist()} "
                    f"abd_min={rec.min_abd:.3f} abd_max={rec.max_abd:.3f}"
                )
                await ws.send_str(encode(PhaseMsg(phase=self._phase)))
                await ws.send_str(encode(PromptMsg(
                    text="Ready. Hold LEFT trigger to engage tracking.",
                )))
                # The client wants to draw a starting-ghost hand whose
                # orientation and finger curls mirror the robot's current
                # state, so the operator can align to the real arm before
                # the first engage.
                await self._send_robot_state(ws)
            else:
                await ws.send_str(encode(PromptMsg(text=self._calib.current_prompt)))

    # ----- outbound loops -----

    async def _on_trigger(self, ws: web.WebSocketResponse, trg: TriggerMsg) -> None:
        """Engage / disengage Cartesian tracking on the left trigger."""
        if trg.hand != "left" or trg.name != "trigger":
            return
        if self._dashboard_control.enabled:
            await ws.send_str(encode(PromptMsg(
                text="Dashboard control mode is enabled; VR teleop is read-only.",
                severity="warn",
            )))
            return
        # Threshold edges: the client sends analog values, the server makes
        # the engage decision so the policy (e.g. hysteresis) lives in one
        # place if we ever need to harden it.
        ENGAGE_THRESHOLD = 0.6
        DISENGAGE_THRESHOLD = 0.3
        engaged = self._tracker.is_engaged
        if not engaged and trg.value >= ENGAGE_THRESHOLD:
            if self._phase != "ready" and self._phase != "tracking":
                # Don't allow engage until finger calibration is complete --
                # otherwise the curls we send to the robot are uncalibrated.
                return
            if not self._latest_hand.valid:
                return
            user_wrist = self._latest_user_wrist_pose()
            if user_wrist is None:
                return
            robot_state = await self._robot.get_state()
            self._tracker.engage(
                user_wrist=user_wrist,
                robot_wrist=robot_state.wrist_pose,
                t=time.monotonic(),
            )
            # Tell the client where the robot-world origin lives in the
            # VR play_space. Helmet axes (x=right, y=up, z=back) differ
            # from robot axes (x=right, y=forward, z=up); the tracker
            # already accounts for this on the wrist. To position the
            # robot-frame origin in helmet coords:
            #   helmet_origin = user_anchor_helmet - R^-1 @ robot_anchor_robot
            # where R^-1 takes (x,y,z)_robot -> (x, z, -y)_helmet.
            user_pos = np.asarray(user_wrist.position, dtype=np.float64)        # helmet frame
            rp = np.asarray(robot_state.wrist_pose.position, dtype=np.float64)  # robot frame
            robot_anchor_in_helmet = np.array([rp[0], rp[2], -rp[1]], dtype=np.float64)
            vr_origin = (user_pos - robot_anchor_in_helmet).tolist()
            self._telemetry.update_anchor(
                (float(vr_origin[0]), float(vr_origin[1]), float(vr_origin[2])),
                timestamp=time.monotonic(),
            )
            await ws.send_str(encode(AnchorMsg(
                vr_position_of_robot_origin=(
                    float(vr_origin[0]), float(vr_origin[1]), float(vr_origin[2]),
                ),
            )))
            print(
                f"[engage] anchor user_pos={np.asarray(user_wrist.position).round(3).tolist()}  "
                f"user_ori={np.asarray(user_wrist.orientation).round(3).tolist()}  "
                f"robot_pos={np.asarray(robot_state.wrist_pose.position).round(3).tolist()}  "
                f"robot_ori={np.asarray(robot_state.wrist_pose.orientation).round(3).tolist()}",
                flush=True,
            )
            self._phase = "tracking"
            await ws.send_str(encode(PhaseMsg(phase=self._phase)))
            await ws.send_str(encode(PromptMsg(
                text="Tracking. Release trigger to freeze.",
            )))
        elif engaged and trg.value <= DISENGAGE_THRESHOLD:
            self._tracker.disengage()
            self._phase = "ready"
            await ws.send_str(encode(PhaseMsg(phase=self._phase)))
            await ws.send_str(encode(PromptMsg(
                text="Held. Pull LEFT trigger to engage tracking again.",
            )))

    async def _send_robot_state(self, ws: web.WebSocketResponse) -> None:
        """Push a one-shot RobotEchoMsg with the current robot wrist + curls.

        The client uses this to render the starting-ghost hand. Failures
        are logged but non-fatal: the client falls back to identity.
        """
        try:
            state = await self._robot.get_state()
        except Exception as exc:
            print(f"[teleop] get_state error: {exc!r}")
            return
        await ws.send_str(encode(RobotEchoMsg(
            wrist_position=tuple(float(v) for v in state.wrist_pose.position),
            wrist_orientation=tuple(float(v) for v in state.wrist_pose.orientation),
            finger_curls=tuple(float(v) for v in state.finger_curls),
            timestamp=float(state.timestamp),
        )))

    def _latest_user_wrist_pose(self) -> Pose | None:
        """Build a Pose from the most recent HandStateMsg, or None if stale."""
        h = self._latest_hand
        if not h.valid:
            return None
        return Pose(
            position=np.asarray(h.wrist_position, dtype=np.float64),
            orientation=np.asarray(h.wrist_orientation, dtype=np.float64),
            frame="play_space",
        )

    async def _command_loop(self) -> None:
        """At command_hz: compute target via tracker, send to RobotDriver."""
        period = 1.0 / max(self._config.command_hz, 1e-3)
        loop = asyncio.get_running_loop()
        while not self._shutdown.is_set():
            t0 = loop.time()
            try:
                await self._tick_command()
            except Exception as exc:
                # Don't let one bad frame kill the loop -- log and keep going,
                # the operator can disengage and re-engage to recover.
                print(f"[teleop] command tick error: {exc!r}")
            elapsed = loop.time() - t0
            await asyncio.sleep(max(0.0, period - elapsed))

    async def _tick_command(self) -> None:
        """One iteration of the command loop. Extracted for readability."""
        if self._dashboard_control.enabled:
            return
        if not self._tracker.is_engaged:
            return
        user_wrist = self._latest_user_wrist_pose()
        if user_wrist is None:
            return
        result = self._tracker.update(user_wrist, time.monotonic())
        # DEBUG: print the wrist delta relative to the engage anchor so the
        # operator can sanity-check that head rotation doesn't leak into the
        # commanded pose. Rate-limited to ~5 Hz so the terminal stays readable.
        now = time.monotonic()
        if now - self._last_debug_print >= 0.2:
            self._last_debug_print = now
            anchor = self._tracker._anchor  # internal but fine for debug
            if anchor is not None:
                dpos = np.asarray(user_wrist.position) \
                    - np.asarray(anchor.user_wrist.position)
                print(
                    f"[wrist] dpos={dpos.round(3).tolist()}  "
                    f"abs_pos={np.asarray(user_wrist.position).round(3).tolist()}  "
                    f"abs_ori={np.asarray(user_wrist.orientation).round(3).tolist()}",
                    flush=True,
                )
        # Calibrated curls live alongside the wrist in the same HandStateMsg.
        # If calibration didn't run (shouldn't happen in 'tracking', but be
        # defensive) the record is identity-ish and apply_curl returns the
        # raw values clamped to [0,1].
        raw_curls = np.asarray(self._latest_hand.curls, dtype=np.float32)
        curls = self._calib.record.apply_curl(raw_curls)
        abd = self._calib.record.apply_abduction(float(self._latest_hand.abduction))
        cmd = RobotCommand(
            target_wrist_pose=result.target,
            target_finger_curls=curls,
            target_thumb_abduction=abd,
            timestamp=time.monotonic(),
        )
        await self._robot.send(cmd)

    async def _pointcloud_loop(self, ws: web.WebSocketResponse) -> None:
        """Send cached point-cloud frames from the shared telemetry hub."""
        last_sequence = 0
        while not ws.closed:
            cloud = await self._telemetry.wait_for_pointcloud(
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

    async def _safety_loop(self, ws) -> None:
        """At safety_hz: run SafetyMonitor; broadcast events to client."""
        raise NotImplementedError

    # ----- phase transitions (small enough to keep in one place) -----

    def _enter_finger_cal(self) -> None: raise NotImplementedError
    def _finish_finger_cal(self) -> None: raise NotImplementedError
    def _engage_tracking(self) -> None: raise NotImplementedError
    def _disengage_tracking(self) -> None: raise NotImplementedError
    def _fault(self, reason: str) -> None: raise NotImplementedError

    def _set_workspace(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._tracker.disengage()
        self._tracker.set_workspace(workspace)
        self._telemetry.set_workspace(workspace)


async def _read_json_body(request) -> dict:
    try:
        payload = await request.json()
    except Exception as exc:
        raise ValueError("request body must be JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload
