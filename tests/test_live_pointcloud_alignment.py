import numpy as np

from teleop_backends.pointcloud.live_alignment import (
    LiveCorrectionConfig,
    LiveCorrectionTracker,
    estimate_icp_correction,
)


def _asymmetric_cloud() -> np.ndarray:
    xs = np.linspace(-0.09, 0.09, 7, dtype=np.float32)
    ys = np.linspace(-0.06, 0.06, 5, dtype=np.float32)
    zs = np.linspace(0.32, 0.56, 4, dtype=np.float32)
    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                if x > 0.04 and y < -0.02:
                    continue
                points.append([x, y, z])
    return np.asarray(points, dtype=np.float32)


def test_estimate_icp_correction_recovers_small_world_offset():
    anchor = _asymmetric_cloud()
    offset = np.array([0.006, -0.004, 0.003], dtype=np.float32)
    target = anchor + offset

    result = estimate_icp_correction(
        anchor,
        target,
        LiveCorrectionConfig(
            min_points=40,
            max_points=500,
            max_correspondence_m=0.02,
            max_translation_m=0.03,
            max_rotation_deg=2.0,
            max_rmse_m=0.01,
            iterations=6,
        ),
    )

    assert result.accepted is True
    assert result.reason is None
    assert result.inlier_count >= 100
    assert result.rmse_m is not None
    assert result.rmse_m < 0.002
    assert np.allclose(result.transform[:3, 3], -offset, atol=0.002)


def test_estimate_icp_correction_rejects_large_translation():
    anchor = _asymmetric_cloud()
    target = anchor + np.array([0.20, 0.0, 0.0], dtype=np.float32)

    result = estimate_icp_correction(
        anchor,
        target,
        LiveCorrectionConfig(
            min_points=40,
            max_points=500,
            max_correspondence_m=0.30,
            max_translation_m=0.03,
            max_rotation_deg=5.0,
            max_rmse_m=0.10,
            iterations=4,
        ),
    )

    assert result.accepted is False
    assert result.reason == "translation_too_large"
    assert result.translation_m is not None
    assert result.translation_m > 0.03


def test_live_correction_tracker_keeps_last_good_correction_after_rejection():
    anchor = _asymmetric_cloud()
    offset = np.array([0.006, -0.004, 0.003], dtype=np.float32)
    tracker = LiveCorrectionTracker(
        LiveCorrectionConfig(
            min_points=40,
            max_points=500,
            max_correspondence_m=0.02,
            max_translation_m=0.03,
            max_rotation_deg=2.0,
            max_rmse_m=0.01,
            iterations=6,
            smoothing_alpha=1.0,
        )
    )

    first = tracker.update("d435i", anchor, anchor + offset)
    rejected = tracker.update("d435i", anchor, anchor + np.array([0.20, 0.0, 0.0]))

    assert first.accepted is True
    assert rejected.accepted is False
    assert np.allclose(tracker.correction_for("d435i")[:3, 3], -offset, atol=0.002)
    assert tracker.diagnostics()["d435i"]["accepted"] is False
    assert tracker.diagnostics()["d435i"]["active"] is True
