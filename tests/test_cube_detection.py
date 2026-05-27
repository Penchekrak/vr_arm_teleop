import math
import time

import numpy as np
import pytest

from teleop_core.cube_detection import CubeDetectionConfig, CubeDetector
from teleop_core.point_cloud import PointCloudFrame


def _cube_points(center_xy, yaw, edge=0.038, samples=12, surface_z=0.0):
    center = np.array([center_xy[0], center_xy[1], surface_z + edge / 2.0], dtype=np.float32)
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    vals = np.linspace(-edge / 2.0, edge / 2.0, samples, dtype=np.float32)
    pts = []

    for x in vals:
        for y in vals:
            top_xy = np.array([x, y], dtype=np.float32) @ rot.T
            pts.append([center[0] + top_xy[0], center[1] + top_xy[1], surface_z + edge])

    for fixed_axis in range(2):
        for fixed_value in (-edge / 2.0, edge / 2.0):
            for free_value in vals:
                for z in np.linspace(0.004, edge - 0.004, 4, dtype=np.float32):
                    local = np.array([0.0, 0.0], dtype=np.float32)
                    local[fixed_axis] = fixed_value
                    local[1 - fixed_axis] = free_value
                    side_xy = local @ rot.T
                    pts.append([center[0] + side_xy[0], center[1] + side_xy[1], surface_z + z])

    return np.asarray(pts, dtype=np.float32)


def _table_points(surface_z=0.0, x_range=(0.05, 0.35), y_range=(-0.08, 0.12), nx=16, ny=12):
    return np.array(
        [
            [x, y, surface_z]
            for x in np.linspace(x_range[0], x_range[1], nx, dtype=np.float32)
            for y in np.linspace(y_range[0], y_range[1], ny, dtype=np.float32)
        ],
        dtype=np.float32,
    )


def _frame(points):
    colors = np.repeat(np.array([[220, 40, 40]], dtype=np.uint8), len(points), axis=0)
    return PointCloudFrame(points=points, colors=colors, timestamp=time.monotonic())


def test_cube_detector_fits_multiple_tabletop_cubes_with_yaw_only_pose():
    floor = _table_points(nx=10, ny=8)
    points = np.vstack([
        floor,
        _cube_points((0.12, -0.02), yaw=0.0),
        _cube_points((0.27, 0.07), yaw=math.radians(30.0)),
    ])
    detector = CubeDetector(CubeDetectionConfig(window_size=1, min_observations=1))

    result = detector.process(_frame(points))

    assert result.count == 2
    centers = sorted((cube.center_m for cube in result.cubes), key=lambda p: p[0])
    assert np.allclose(centers[0], [0.12, -0.02, 0.019], atol=0.006)
    assert np.allclose(centers[1], [0.27, 0.07, 0.019], atol=0.006)
    assert all(abs(cube.center_m[2] - cube.edge_m / 2.0) < 1e-6 for cube in result.cubes)
    assert all(abs(cube.roll_rad) < 1e-6 and abs(cube.pitch_rad) < 1e-6 for cube in result.cubes)


def test_cube_detector_estimates_table_height_and_cube_edge():
    surface_z = 0.18
    edge = 0.052
    points = np.vstack([
        _table_points(surface_z=surface_z, x_range=(0.05, 0.28), y_range=(-0.08, 0.14)),
        _cube_points((0.16, 0.03), yaw=math.radians(20.0), edge=edge, surface_z=surface_z),
    ])
    detector = CubeDetector(CubeDetectionConfig(window_size=1, min_observations=1))

    result = detector.process(_frame(points))

    assert result.count == 1
    cube = result.cubes[0]
    assert cube.center_m == pytest.approx((0.16, 0.03, surface_z + edge / 2.0), abs=0.008)
    assert cube.edge_m == pytest.approx(edge, abs=0.006)
    assert result.stats["surface_z_m"] == pytest.approx(surface_z, abs=0.004)


def test_cube_detector_splits_nearby_cube_positions():
    points = np.vstack([
        _table_points(x_range=(0.05, 0.27), y_range=(-0.08, 0.10), nx=12, ny=10),
        _cube_points((0.12, -0.01), yaw=0.0),
        _cube_points((0.185, -0.01), yaw=math.radians(10.0)),
    ])
    detector = CubeDetector(CubeDetectionConfig(window_size=1, min_observations=1))

    result = detector.process(_frame(points))

    assert result.count == 2
    centers = sorted((cube.center_m for cube in result.cubes), key=lambda p: p[0])
    assert np.allclose(centers[0], [0.12, -0.01, 0.019], atol=0.007)
    assert np.allclose(centers[1], [0.185, -0.01, 0.019], atol=0.007)


def test_cube_detector_temporal_voxel_averaging_rejects_single_frame_clutter():
    cube = _cube_points((0.16, 0.04), yaw=0.0)
    clutter = np.array([[0.32, 0.0, 0.02], [0.325, 0.0, 0.028], [0.33, 0.0, 0.036]], dtype=np.float32)
    detector = CubeDetector(CubeDetectionConfig(window_size=3, min_observations=2))

    detector.process(_frame(np.vstack([cube, clutter])))
    result = detector.process(_frame(cube + np.array([0.0004, -0.0003, 0.0], dtype=np.float32)))

    assert result.count == 1
    assert np.allclose(result.cubes[0].center_m, [0.16, 0.04, 0.019], atol=0.006)


def test_cube_detector_caps_work_on_pathological_point_clouds():
    points = np.array(
        [
            [x, y, 0.02]
            for x in np.linspace(-0.2, 0.2, 20, dtype=np.float32)
            for y in np.linspace(-0.2, 0.2, 20, dtype=np.float32)
        ],
        dtype=np.float32,
    )
    detector = CubeDetector(CubeDetectionConfig(
        window_size=1,
        min_observations=1,
        voxel_m=0.001,
        cluster_eps_m=0.0001,
        min_cluster_points=999,
        max_stable_points=40,
        max_fit_clusters=3,
        surface_z_m=0.0,
    ))

    result = detector.process(_frame(points))

    assert result.count == 0
    assert result.stats["stable_points_before_limit"] > 40
    assert result.stats["stable_points"] == 40
    assert result.stats["clusters"] == 40
    assert result.stats["fit_clusters"] == 3
    assert result.stats["truncated_clusters"] == 37
