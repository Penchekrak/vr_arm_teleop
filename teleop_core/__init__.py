"""Pure, backend-agnostic teleop core.

This package defines:
- data types (Pose, PointCloudFrame, RobotState, RobotCommand, ...),
- abstract interfaces (PointCloudSource, RobotDriver),
- pure logic (CartesianTracker, FingerCalibrationFSM, SafetyMonitor,
  Workspace),
- the orchestrator (TeleopServer).

It never imports from a specific camera or robot SDK. Concrete
implementations live in :mod:`teleop_backends`.
"""

from .calibration import (
    CalibBound, CalibStep, CalibStepKind, CalibrationRecord, FingerCalibrationFSM,
)
from .point_cloud import PointCloudFrame, PointCloudSource, encode_frame
from .cube_detection import (
    CubeDetectionConfig, CubeDetectionResult, CubeDetector, DetectedCube,
)
from .robot import RobotCommand, RobotDriver, RobotState
from .safety import SafetyConfig, SafetyEvent, SafetyKind, SafetyMonitor, Severity
from .server import ServerConfig, TeleopServer
from .telemetry import EncodedPointCloud, TelemetryHub
from .tracking import CartesianTracker, TrackingResult, WristAnchor
from .types import Pose, Vec3
from .workspace import Workspace

__all__ = [
    "CalibBound", "CalibStep", "CalibStepKind", "CalibrationRecord",
    "FingerCalibrationFSM",
    "PointCloudFrame", "PointCloudSource", "encode_frame",
    "CubeDetectionConfig", "CubeDetectionResult", "CubeDetector", "DetectedCube",
    "RobotCommand", "RobotDriver", "RobotState",
    "SafetyConfig", "SafetyEvent", "SafetyKind", "SafetyMonitor", "Severity",
    "ServerConfig", "TeleopServer",
    "EncodedPointCloud", "TelemetryHub",
    "CartesianTracker", "TrackingResult", "WristAnchor",
    "Pose", "Vec3",
    "Workspace",
]
