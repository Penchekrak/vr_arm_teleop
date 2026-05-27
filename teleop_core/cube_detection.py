"""Tabletop cube detection for world-frame point clouds.

The detector assumes cubes lie flat on a horizontal support surface, but
the surface height and cube edge length are estimated from the cloud so
small setup changes do not require retuning.
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
                    float(self.center_m[2] - half),
                ],
                "max": [
                    float(self.center_m[0] + half),
                    float(self.center_m[1] + half),
                    float(self.center_m[2] + half),
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
    edge_min_m: float = 0.020
    edge_max_m: float = 0.080
    surface_z_m: float | None = None
    surface_z_bin_m: float = 0.004
    surface_z_percentile: float = 35.0
    min_height_m: float = 0.003
    max_height_margin_m: float = 0.012
    voxel_m: float = 0.003
    window_size: int = 6
    min_observations: int = 2
    cluster_eps_m: float = 0.014
    min_cluster_points: int = 20
    max_fit_error_m: float = 0.006
    face_tolerance_m: float = 0.006
    min_inside_fraction: float = 0.72
    min_surface_fraction: float = 0.34
    min_supported_faces: int = 2
    min_face_points: int = 8
    min_face_fraction: float = 0.06
    min_local_extent_ratio: float = 0.45
    min_large_extent_axes: int = 2
    nms_distance_m: float = 0.024
    nms_distance_ratio: float = 0.7
    max_cubes: int = 24
    max_stable_points: int = 6000
    max_fit_clusters: int = 64
    max_fit_points_per_cluster: int = 1200


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

        surface_z = self._estimate_surface_z(frame.points)
        edge_min, edge_max = self._edge_bounds()
        points = self._preselect_points(frame.points, surface_z, edge_max)
        keys, averaged = _voxel_average(points, self.config.voxel_m)
        self._frames.append((keys, averaged))
        stable = self._stable_voxels()
        stable_points_before_limit = int(stable.shape[0])
        stable = _limit_points(stable, self.config.max_stable_points)
        clusters = _cluster_xy(stable, self.config.cluster_eps_m)
        cluster_count = len(clusters)
        clusters.sort(key=len, reverse=True)
        max_fit_clusters = max(0, int(self.config.max_fit_clusters))
        if max_fit_clusters:
            fit_clusters = clusters[:max_fit_clusters]
        else:
            fit_clusters = []
        truncated_clusters = max(0, cluster_count - len(fit_clusters))

        candidates = []
        rejected = truncated_clusters
        for cluster in fit_clusters:
            if len(cluster) < self.config.min_cluster_points:
                rejected += 1
                continue
            fit_points = _limit_points(cluster, self.config.max_fit_points_per_cluster)
            fit = self._fit_cluster(fit_points, surface_z, edge_min, edge_max)
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
                < max(
                    self.config.nms_distance_m,
                    self.config.nms_distance_ratio * min(candidate.edge_m, cube.edge_m),
                )
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
                "surface_z_m": float(surface_z),
                "edge_min_m": float(edge_min),
                "edge_max_m": float(edge_max),
                "height_filtered_points": int(points.shape[0]),
                "stable_points_before_limit": stable_points_before_limit,
                "stable_points": int(stable.shape[0]),
                "clusters": int(cluster_count),
                "fit_clusters": int(len(fit_clusters)),
                "truncated_clusters": int(truncated_clusters),
                "rejected_clusters": int(rejected),
                "window_frames": int(len(self._frames)),
                "min_observations": int(min(self.config.min_observations, len(self._frames))),
            },
        )
        return self._last_result

    def _edge_bounds(self) -> tuple[float, float]:
        cfg = self.config
        edge = max(1e-6, float(cfg.edge_m))
        edge_min = max(1e-6, min(float(cfg.edge_min_m), edge, float(cfg.edge_max_m)))
        edge_max = max(edge_min, max(float(cfg.edge_min_m), edge, float(cfg.edge_max_m)))
        return edge_min, edge_max

    def _estimate_surface_z(self, points: np.ndarray) -> float:
        cfg = self.config
        if cfg.surface_z_m is not None:
            return float(cfg.surface_z_m)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        z = pts[np.isfinite(pts).all(axis=1), 2]
        if len(z) == 0:
            return 0.0

        percentile = float(np.clip(cfg.surface_z_percentile, 1.0, 95.0))
        upper = float(np.percentile(z, percentile))
        low = z[z <= upper]
        if len(low) == 0:
            low = z

        bin_m = max(1e-4, float(cfg.surface_z_bin_m))
        bins = np.floor(low / bin_m).astype(np.int64)
        unique, counts = np.unique(bins, return_counts=True)
        min_support = max(
            4,
            int(math.ceil(0.08 * int(counts.max()))),
            int(math.ceil(0.02 * len(low))),
        )
        supported = unique[counts >= min_support]
        chosen = int(supported.min() if len(supported) else unique[counts.argmax()])
        in_bin = low[bins == chosen]
        if len(in_bin) == 0:
            return float(np.min(z))
        return float(np.median(in_bin))

    def _preselect_points(
        self,
        points: np.ndarray,
        surface_z: float,
        edge_max_m: float,
    ) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        z_min = surface_z + self.config.min_height_m
        z_max = surface_z + edge_max_m + self.config.max_height_margin_m
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

    def _fit_cluster(
        self,
        points: np.ndarray,
        surface_z: float,
        edge_min_m: float,
        edge_max_m: float,
    ) -> DetectedCube | None:
        cfg = self.config
        xy = points[:, :2].astype(np.float64)
        center_xy = np.median(xy, axis=0)
        yaw0 = _principal_yaw(xy)
        seeds = [yaw0, yaw0 + math.pi / 4.0, yaw0 + math.pi / 2.0, 0.0]
        edge0 = _estimate_cluster_edge(points, surface_z, cfg.edge_m)
        edge_seeds = _unique_floats([
            edge0,
            cfg.edge_m,
            edge_min_m,
            edge_max_m,
        ])
        best = None

        for seed in seeds:
            for edge_seed in edge_seeds:
                x0 = np.array([
                    center_xy[0],
                    center_xy[1],
                    _wrap_half_pi(seed),
                    float(np.clip(edge_seed, edge_min_m, edge_max_m)),
                ], dtype=np.float64)

                def residuals(x):
                    edge = float(x[3])
                    local = _local_cube_points(points, x[0], x[1], x[2], surface_z, edge)
                    return _box_sdf(local, edge / 2.0)

                sol = least_squares(
                    residuals,
                    x0,
                    bounds=(
                        [-math.inf, -math.inf, -math.pi / 4.0, edge_min_m],
                        [math.inf, math.inf, math.pi / 4.0, edge_max_m],
                    ),
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
        edge = float(params[3])
        local = _local_cube_points(points, float(params[0]), float(params[1]), yaw, surface_z, edge)
        quality = _fit_quality(local, cfg, edge)
        if err > cfg.max_fit_error_m:
            return None
        if quality["inside_fraction"] < cfg.min_inside_fraction:
            return None
        if quality["surface_fraction"] < cfg.min_surface_fraction:
            return None
        if quality["supported_faces"] < cfg.min_supported_faces:
            return None
        if quality["large_extent_axes"] < cfg.min_large_extent_axes:
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
            center_m=(float(params[0]), float(params[1]), float(surface_z + edge / 2.0)),
            yaw_rad=yaw,
            edge_m=edge,
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


def _limit_points(points: np.ndarray, limit: int) -> np.ndarray:
    limit = int(limit)
    if limit <= 0 or len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    return points[indices]


def _principal_yaw(xy: np.ndarray) -> float:
    centered = xy - np.mean(xy, axis=0)
    if len(centered) < 3:
        return 0.0
    cov = centered.T @ centered
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    return math.atan2(float(axis[1]), float(axis[0]))


def _estimate_cluster_edge(
    points: np.ndarray,
    surface_z: float,
    fallback_edge_m: float,
) -> float:
    if len(points) == 0:
        return float(fallback_edge_m)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    height = float(np.percentile(pts[:, 2] - surface_z, 95.0))
    xy_extent = np.ptp(pts[:, :2], axis=0)
    xy_guess = float(np.median(xy_extent))
    guesses = [value for value in (height, xy_guess, float(fallback_edge_m)) if value > 0]
    return float(np.median(guesses)) if guesses else float(fallback_edge_m)


def _unique_floats(values: list[float], *, min_separation: float = 1e-4) -> list[float]:
    result: list[float] = []
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            continue
        if any(abs(value - existing) < min_separation for existing in result):
            continue
        result.append(value)
    return result


def _local_cube_points(
    points: np.ndarray,
    cx: float,
    cy: float,
    yaw: float,
    surface_z: float,
    edge_m: float,
) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    dx = points[:, 0].astype(np.float64) - cx
    dy = points[:, 1].astype(np.float64) - cy
    z = points[:, 2].astype(np.float64) - (surface_z + edge_m / 2.0)
    return np.column_stack([c * dx + s * dy, -s * dx + c * dy, z])


def _box_sdf(local_points: np.ndarray, half: float) -> np.ndarray:
    q = np.abs(local_points) - half
    outside = np.maximum(q, 0.0)
    outside_distance = np.linalg.norm(outside, axis=1)
    inside_distance = np.minimum(np.max(q, axis=1), 0.0)
    return outside_distance + inside_distance


def _fit_quality(local_points: np.ndarray, cfg: CubeDetectionConfig, edge_m: float) -> dict[str, float]:
    half = edge_m / 2.0
    abs_local = np.abs(local_points)
    expanded = half + cfg.face_tolerance_m
    inside = np.all(abs_local <= expanded, axis=1)
    nearest_face = np.min(np.abs(abs_local - half), axis=1)
    surface = inside & (nearest_face <= cfg.face_tolerance_m)

    face_counts = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            on_face = inside & (
                np.abs(local_points[:, axis] - sign * half) <= cfg.face_tolerance_m
            )
            face_counts.append(int(np.sum(on_face)))
    min_face_points = max(
        int(cfg.min_face_points),
        int(math.ceil(float(cfg.min_face_fraction) * len(local_points))),
    )
    supported_faces = sum(count >= min_face_points for count in face_counts)
    local_extent = np.ptp(local_points, axis=0) if len(local_points) else np.zeros(3)
    min_extent = max(0.0, float(cfg.min_local_extent_ratio)) * edge_m
    large_extent_axes = int(np.sum(local_extent >= min_extent))

    return {
        "inside_fraction": float(np.mean(inside)) if len(local_points) else 0.0,
        "surface_fraction": float(np.mean(surface)) if len(local_points) else 0.0,
        "supported_faces": int(supported_faces),
        "large_extent_axes": int(large_extent_axes),
    }


def _wrap_half_pi(yaw: float) -> float:
    period = math.pi / 2.0
    return ((float(yaw) + period / 2.0) % period) - period / 2.0
