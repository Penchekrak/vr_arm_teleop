"""Backward-compatible RealSense-only point-cloud source.

The current hardware pipeline supports mixed RealSense and ZED 2i
configs. This wrapper preserves the old ``--pc-backend realsense`` name
and rejects non-RealSense cameras so legacy runs stay explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hardware import (
    HardwarePointCloudConfig,
    HardwarePointCloudSource,
    ReaderFactory,
    load_hardware_config,
)


@dataclass(frozen=True)
class CameraConfig:
    """Legacy RealSense camera identity + placement in the world."""

    serial: str
    extrinsic_world_from_cam: np.ndarray
    width: int = 424
    height: int = 240
    fps: int = 30


class MultiRealSenseSource(HardwarePointCloudSource):
    """RealSense-only wrapper around the configured hardware source."""

    def __init__(
        self,
        config: HardwarePointCloudConfig,
        reader_factory: ReaderFactory | None = None,
    ) -> None:
        _validate_realsense_only(config)
        super().__init__(config, reader_factory=reader_factory)

    @classmethod
    def from_config_file(
        cls,
        path: Path,
        reader_factory: ReaderFactory | None = None,
    ) -> "MultiRealSenseSource":
        return cls(load_hardware_config(path), reader_factory=reader_factory)


def _validate_realsense_only(config: HardwarePointCloudConfig) -> None:
    non_realsense = [
        camera.name for camera in config.cameras
        if camera.camera_type != "realsense"
    ]
    if non_realsense:
        joined = ", ".join(non_realsense)
        raise ValueError(
            "--pc-backend realsense accepts only camera type 'realsense'; "
            f"non-RealSense cameras: {joined}"
        )
