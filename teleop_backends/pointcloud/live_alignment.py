"""Live point-cloud alignment helpers for calibration-time fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class LiveCorrectionConfig:
    """Bounds for a small live ICP correction in world coordinates."""

    enabled: bool = True
    min_points: int = 300
    max_points: int = 4000
    max_correspondence_m: float = 0.03
    max_translation_m: float = 0.03
    max_rotation_deg: float = 3.0
    max_rmse_m: float = 0.015
    min_inlier_ratio: float = 0.10
    iterations: int = 5
    smoothing_alpha: float = 0.35


@dataclass(frozen=True)
class LiveCorrectionResult:
    """A correction that maps currently transformed target points to anchor points."""

    accepted: bool
    reason: str | None
    transform: np.ndarray
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    rmse_m: float | None = None
    translation_m: float | None = None
    rotation_deg: float | None = None

    def diagnostics(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "inlier_count": int(self.inlier_count),
            "inlier_ratio": float(self.inlier_ratio),
            "rmse_m": self.rmse_m,
            "translation_m": self.translation_m,
            "rotation_deg": self.rotation_deg,
        }


class LiveCorrectionTracker:
    """Tracks per-camera live corrections while preserving last good values."""

    def __init__(self, config: LiveCorrectionConfig) -> None:
        self._config = config
        self._corrections: dict[str, np.ndarray] = {}
        self._diagnostics: dict[str, dict[str, object]] = {}

    def update(
        self,
        name: str,
        anchor_world_points: np.ndarray,
        target_world_points: np.ndarray,
    ) -> LiveCorrectionResult:
        result = estimate_icp_correction(
            anchor_world_points,
            target_world_points,
            self._config,
        )
        if result.accepted:
            previous = self._corrections.get(name)
            correction = (
                result.transform
                if previous is None
                else _blend_transforms(
                    previous,
                    result.transform,
                    self._config.smoothing_alpha,
                )
            )
            self._corrections[name] = correction
            result = replace(result, transform=correction)
        diagnostic = result.diagnostics()
        diagnostic["active"] = name in self._corrections
        self._diagnostics[name] = diagnostic
        return result

    def correction_for(self, name: str) -> np.ndarray:
        correction = self._corrections.get(name)
        if correction is None:
            return np.eye(4, dtype=np.float64)
        return correction.copy()

    def corrected_transform(self, name: str, world_from_camera: np.ndarray) -> np.ndarray:
        return self.correction_for(name) @ np.asarray(
            world_from_camera,
            dtype=np.float64,
        ).reshape(4, 4)

    def diagnostics(self) -> dict[str, dict[str, object]]:
        return {
            name: dict(diagnostic)
            for name, diagnostic in self._diagnostics.items()
        }


def estimate_icp_correction(
    anchor_world_points: np.ndarray,
    target_world_points: np.ndarray,
    config: LiveCorrectionConfig,
) -> LiveCorrectionResult:
    """Estimate a bounded rigid correction from target world points to anchor.

    This is intentionally small-scope ICP: ChArUco supplies the coarse
    transform, and this only trims residual cloud mismatch. Any correction
    outside the configured envelope is rejected.
    """

    identity = np.eye(4, dtype=np.float64)
    if not config.enabled:
        return _rejected("disabled", identity)

    anchor = _clean_points(anchor_world_points)
    target = _clean_points(target_world_points)
    if anchor.shape[0] < config.min_points or target.shape[0] < config.min_points:
        return _rejected("not_enough_points", identity)

    anchor = _sample_points(anchor, config.max_points)
    target = _sample_points(target, config.max_points)
    moving = target.copy()
    total = identity.copy()
    inlier_count = 0
    inlier_ratio = 0.0
    rmse: float | None = None

    for _ in range(max(1, int(config.iterations))):
        tree = cKDTree(anchor)
        distances, indices = tree.query(
            moving,
            k=1,
            distance_upper_bound=float(config.max_correspondence_m),
        )
        valid = np.isfinite(distances) & (indices < anchor.shape[0])
        inlier_count = int(np.count_nonzero(valid))
        inlier_ratio = float(inlier_count / max(1, moving.shape[0]))
        if (
            inlier_count < config.min_points
            or inlier_ratio < float(config.min_inlier_ratio)
        ):
            return LiveCorrectionResult(
                accepted=False,
                reason="not_enough_overlap",
                transform=identity,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
            )
        source = moving[valid]
        destination = anchor[indices[valid]]
        delta, rmse = _kabsch_transform(source, destination)
        moving = _transform_points(moving, delta)
        total = delta @ total

    translation_m = float(np.linalg.norm(total[:3, 3]))
    rotation_deg = math.degrees(_rotation_angle(total[:3, :3]))
    if translation_m > float(config.max_translation_m):
        return _rejected(
            "translation_too_large",
            total,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            rmse_m=rmse,
            translation_m=translation_m,
            rotation_deg=rotation_deg,
        )
    if rotation_deg > float(config.max_rotation_deg):
        return _rejected(
            "rotation_too_large",
            total,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            rmse_m=rmse,
            translation_m=translation_m,
            rotation_deg=rotation_deg,
        )
    if rmse is None or rmse > float(config.max_rmse_m):
        return _rejected(
            "rmse_too_high",
            total,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            rmse_m=rmse,
            translation_m=translation_m,
            rotation_deg=rotation_deg,
        )
    return LiveCorrectionResult(
        accepted=True,
        reason=None,
        transform=total.astype(np.float64, copy=False),
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        rmse_m=float(rmse),
        translation_m=translation_m,
        rotation_deg=rotation_deg,
    )


def _rejected(
    reason: str,
    transform: np.ndarray,
    *,
    inlier_count: int = 0,
    inlier_ratio: float = 0.0,
    rmse_m: float | None = None,
    translation_m: float | None = None,
    rotation_deg: float | None = None,
) -> LiveCorrectionResult:
    return LiveCorrectionResult(
        accepted=False,
        reason=reason,
        transform=np.asarray(transform, dtype=np.float64).reshape(4, 4),
        inlier_count=int(inlier_count),
        inlier_ratio=float(inlier_ratio),
        rmse_m=None if rmse_m is None else float(rmse_m),
        translation_m=None if translation_m is None else float(translation_m),
        rotation_deg=None if rotation_deg is None else float(rotation_deg),
    )


def _clean_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    max_points = int(max_points)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[indices]


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([points, ones], axis=1)
    return (homogeneous @ np.asarray(transform, dtype=np.float64).reshape(4, 4).T)[:, :3]


def _kabsch_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, float]:
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
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
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transformed = (rotation @ source.T).T + translation
    rmse = float(np.sqrt(np.mean(np.sum((transformed - target) ** 2, axis=1))))
    return transform, rmse


def _rotation_angle(rotation: np.ndarray) -> float:
    cos_angle = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.acos(max(-1.0, min(1.0, cos_angle)))


def _blend_transforms(
    previous: np.ndarray,
    current: np.ndarray,
    alpha: float,
) -> np.ndarray:
    alpha = max(0.0, min(1.0, float(alpha)))
    previous = np.asarray(previous, dtype=np.float64).reshape(4, 4)
    current = np.asarray(current, dtype=np.float64).reshape(4, 4)
    if alpha >= 1.0:
        return current.copy()
    if alpha <= 0.0:
        return previous.copy()
    blended = np.eye(4, dtype=np.float64)
    blended[:3, 3] = (1.0 - alpha) * previous[:3, 3] + alpha * current[:3, 3]
    rotations = Rotation.from_matrix([previous[:3, :3], current[:3, :3]])
    slerp = Slerp([0.0, 1.0], rotations)
    blended[:3, :3] = slerp([alpha]).as_matrix()[0]
    return blended
