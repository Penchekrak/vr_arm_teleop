"""Tabletop cube detection for world-frame point clouds.

The targeted setup has one useful simplifying constraint: the support
surface is always ``z=0`` in robot base coordinates, and cubes lie flat
on that surface. The fitted pose is therefore only ``x, y, yaw`` with a
fixed ``z=edge/2`` and zero roll/pitch.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .point_cloud import PointCloudFrame


@dataclass(frozen=True)
class DetectedCube:
    id: str
    center_m: tuple[float, float, float]
    yaw_rad: float
    edge_m: float
    points: int
    mean_abs_sdf_m: float
    confidence: float
    roll_rad: float = 0.0
    pitch_rad: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        half = self.edge_m / 2.0
        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)
        return {
            "id": self.id,
            "center_m": [float(v) for v in self.center_m],
            "edge_m": float(self.edge_m),
            "yaw_rad": float(self.yaw_rad),
            "roll_rad": float(self.roll_rad),
            "pitch_rad": float(self.pitch_rad),
            "quaternion_xyzw": [
                0.0,
                0.0,
                float(math.sin(self.yaw_rad / 2.0)),
                float(math.cos(self.yaw_rad / 2.0)),
            ],
            "rotation_matrix": [
                [float(c), float(-s), 0.0],
                [float(s), float(c), 0.0],
                [0.0, 0.0, 1.0],
            ],
            "bounds_m": {
                "min": [
                    float(self.center_m[0] - half),
                    float(self.center_m[1] - half),
                    0.0,
                ],
                "max": [
                    float(self.center_m[0] + half),
                    float(self.center_m[1] + half),
                    float(self.edge_m),
                ],
            },
            "points": int(self.points),
            "mean_abs_sdf_m": float(self.mean_abs_sdf_m),
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class CubeDetectionResult:
    sequence: int
    timestamp: float | None
    cubes: tuple[DetectedCube, ...] = ()
    error: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.cubes)

    def as_snapshot(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "timestamp": self.timestamp,
            "count": self.count,
            "items": [cube.as_dict() for cube in self.cubes],
            "error": self.error,
            "stats": self.stats,
        }


@dataclass(frozen=True)
class CubeDetectionConfig:
    edge_m: float = 0.038
    surface_z_m: float = 0.0
    min_height_m: float = 0.003
    max_height_margin_m: float = 0.012
    voxel_m: float = 0.003
    window_size: int = 6
    min_observations: int = 2
    cluster_eps_m: float = 0.055
    min_cluster_points: int = 35
    max_fit_error_m: float = 0.006
    face_tolerance_m: float = 0.006
    min_inside_fraction: float = 0.72
    min_surface_fraction: float = 0.34
    nms_distance_m: float = 0.024
    max_cubes: int = 24


class CubeDetector:
    """Stateful detector with short-window voxel averaging."""

    def __init__(self, config: CubeDetectionConfig | None = None) -> None:
        self.config = config or CubeDetectionConfig()
        self._frames: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=max(1, int(self.config.window_size))
        )
        self._sequence = 0
        self._last_result = CubeDetectionResult(sequence=0, timestamp=None)

    @property
    def last_result(self) -> CubeDetectionResult:
        return self._last_result

    def reset(self) -> None:
        self._frames.clear()
        self._last_result = CubeDetectionResult(sequence=self._sequence, timestamp=None)

    def process(self, frame: PointCloudFrame | None) -> CubeDetectionResult:
        self._sequence += 1
        if frame is None:
            self._last_result = CubeDetectionResult(
                sequence=self._sequence,
                timestamp=None,
                error="no_reference_frame",
            )
            return self._last_result

        points = self._preselect_points(frame.points)
        keys, averaged = _voxel_average(points, self.config.voxel_m)
        self._frames.append((keys, averaged))
        stable = self._stable_voxels()
        clusters = _cluster_xy(stable, self.config.cluster_eps_m)

        candidates = []
        rejected = 0
        for cluster in clusters:
            if len(cluster) < self.config.min_cluster_points:
                rejected += 1
                continue
            fit = self._fit_cluster(cluster)
            if fit is None:
                rejected += 1
                continue
            candidates.append(fit)

        candidates.sort(key=lambda cube: (-cube.confidence, cube.mean_abs_sdf_m))
        accepted: list[DetectedCube] = []
        for candidate in candidates:
            center = np.asarray(candidate.center_m, dtype=np.float64)
            duplicate = any(
                np.linalg.norm(center[:2] - np.asarray(cube.center_m)[:2])
                < self.config.nms_distance_m
                for cube in accepted
            )
            if duplicate:
                continue
            accepted.append(
                DetectedCube(
                    id=f"cube-{len(accepted)}",
                    center_m=candidate.center_m,
                    yaw_rad=candidate.yaw_rad,
                    edge_m=candidate.edge_m,
                    points=candidate.points,
                    mean_abs_sdf_m=candidate.mean_abs_sdf_m,
                    confidence=candidate.confidence,
                )
            )
            if len(accepted) >= self.config.max_cubes:
                break

        self._last_result = CubeDetectionResult(
            sequence=self._sequence,
            timestamp=float(frame.timestamp),
            cubes=tuple(accepted),
            stats={
                "input_points": int(np.asarray(frame.points).shape[0]),
                "height_filtered_points": int(points.shape[0]),
                "stable_points": int(stable.shape[0]),
                "clusters": int(len(clusters)),
                "rejected_clusters": int(rejected),
                "window_frames": int(len(self._frames)),
                "min_observations": int(min(self.config.min_observations, len(self._frames))),
            },
        )
        return self._last_result

    def _preselect_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        z_min = self.config.surface_z_m + self.config.min_height_m
        z_max = self.config.surface_z_m + self.config.edge_m + self.config.max_height_margin_m
        mask = (
            np.isfinite(pts).all(axis=1)
            & (pts[:, 2] >= z_min)
            & (pts[:, 2] <= z_max)
        )
        return pts[mask]

    def _stable_voxels(self) -> np.ndarray:
        if not self._frames:
            return np.empty((0, 3), dtype=np.float32)
        keys = np.concatenate([item[0] for item in self._frames], axis=0)
        points = np.concatenate([item[1] for item in self._frames], axis=0)
        if len(keys) == 0:
            return np.empty((0, 3), dtype=np.float32)

        unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        sums = np.zeros((len(unique), 3), dtype=np.float64)
        np.add.at(sums, inverse, points)
        min_obs = min(max(1, int(self.config.min_observations)), len(self._frames))
        keep = counts >= min_obs
        if not np.any(keep):
            return np.empty((0, 3), dtype=np.float32)
        averaged = sums[keep] / counts[keep, None]
        return averaged.astype(np.float32)

    def _fit_cluster(self, points: np.ndarray) -> DetectedCube | None:
        cfg = self.config
        xy = points[:, :2].astype(np.float64)
        center_xy = np.median(xy, axis=0)
        yaw0 = _principal_yaw(xy)
        seeds = [yaw0, yaw0 + math.pi / 4.0, yaw0 + math.pi / 2.0, 0.0]
        best = None

        for seed in seeds:
            x0 = np.array([center_xy[0], center_xy[1], _wrap_half_pi(seed)], dtype=np.float64)

            def residuals(x):
                local = _local_cube_points(points, x[0], x[1], x[2], cfg)
                return _box_sdf(local, cfg.edge_m / 2.0)

            sol = least_squares(
                residuals,
                x0,
                loss="huber",
                f_scale=0.003,
                max_nfev=80,
            )
            err = float(np.mean(np.abs(residuals(sol.x)))) if len(points) else math.inf
            if best is None or err < best[0]:
                best = (err, sol.x)

        if best is None:
            return None
        err, params = best
        yaw = _wrap_half_pi(float(params[2]))
        local = _local_cube_points(points, float(params[0]), float(params[1]), yaw, cfg)
        quality = _fit_quality(local, cfg)
        if err > cfg.max_fit_error_m:
            return None
        if quality["inside_fraction"] < cfg.min_inside_fraction:
            return None
        if quality["surface_fraction"] < cfg.min_surface_fraction:
            return None

        confidence = float(np.clip(
            0.35 * quality["inside_fraction"]
            + 0.35 * quality["surface_fraction"]
            + 0.30 * (1.0 - err / max(cfg.max_fit_error_m, 1e-6)),
            0.0,
            1.0,
        ))
        return DetectedCube(
            id="candidate",
            center_m=(float(params[0]), float(params[1]), float(cfg.surface_z_m + cfg.edge_m / 2.0)),
            yaw_rad=yaw,
            edge_m=float(cfg.edge_m),
            points=int(len(points)),
            mean_abs_sdf_m=err,
            confidence=confidence,
        )


def _voxel_average(points: np.ndarray, voxel_m: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return (
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.float32),
        )
    keys = np.floor(points / voxel_m).astype(np.int64)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(unique), 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    averaged = sums / counts[:, None]
    return unique, averaged.astype(np.float32)


def _cluster_xy(points: np.ndarray, eps_m: float) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    tree = cKDTree(points[:, :2])
    visited = np.zeros(len(points), dtype=bool)
    clusters: list[np.ndarray] = []
    for start in range(len(points)):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        members = []
        while queue:
            idx = queue.pop()
            members.append(idx)
            for neighbor in tree.query_ball_point(points[idx, :2], eps_m):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        clusters.append(points[np.asarray(members, dtype=np.int64)])
    return clusters


def _principal_yaw(xy: np.ndarray) -> float:
    centered = xy - np.mean(xy, axis=0)
    if len(centered) < 3:
        return 0.0
    cov = centered.T @ centered
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    return math.atan2(float(axis[1]), float(axis[0]))


def _local_cube_points(points: np.ndarray, cx: float, cy: float, yaw: float, cfg: CubeDetectionConfig) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    dx = points[:, 0].astype(np.float64) - cx
    dy = points[:, 1].astype(np.float64) - cy
    z = points[:, 2].astype(np.float64) - (cfg.surface_z_m + cfg.edge_m / 2.0)
    return np.column_stack([c * dx + s * dy, -s * dx + c * dy, z])


def _box_sdf(local_points: np.ndarray, half: float) -> np.ndarray:
    q = np.abs(local_points) - half
    outside = np.maximum(q, 0.0)
    outside_distance = np.linalg.norm(outside, axis=1)
    inside_distance = np.minimum(np.max(q, axis=1), 0.0)
    return outside_distance + inside_distance


def _fit_quality(local_points: np.ndarray, cfg: CubeDetectionConfig) -> dict[str, float]:
    half = cfg.edge_m / 2.0
    abs_local = np.abs(local_points)
    expanded = half + cfg.face_tolerance_m
    inside = np.all(abs_local <= expanded, axis=1)
    nearest_face = np.min(np.abs(abs_local - half), axis=1)
    surface = inside & (nearest_face <= cfg.face_tolerance_m)
    return {
        "inside_fraction": float(np.mean(inside)) if len(local_points) else 0.0,
        "surface_fraction": float(np.mean(surface)) if len(local_points) else 0.0,
    }


def _wrap_half_pi(yaw: float) -> float:
    period = math.pi / 2.0
    return ((float(yaw) + period / 2.0) % period) - period / 2.0
