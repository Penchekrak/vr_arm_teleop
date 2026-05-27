import math

import numpy as np
import pytest

from teleop_core.workspace import Workspace


def _yaw_quat(deg: float):
    half = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def test_oriented_workspace_contains_and_clamps_in_local_box_frame():
    workspace = Workspace.from_dict({
        "center": [0.0, 0.0, 0.0],
        "half_extents": [1.0, 0.5, 0.25],
        "orientation": _yaw_quat(90.0),
        "frame": "world",
    })

    assert workspace.contains(np.array([0.0, 0.9, 0.0]))
    assert not workspace.contains(np.array([0.9, 0.0, 0.0]))

    clamped, was_outside = workspace.clamp(np.array([0.9, 0.0, 0.3]))

    assert was_outside is True
    assert clamped == pytest.approx([0.5, 0.0, 0.25], abs=1e-6)


def test_legacy_min_max_workspace_serializes_oriented_fields():
    workspace = Workspace(
        min_corner=np.array([0.1, -0.2, 0.0], dtype=np.float32),
        max_corner=np.array([0.5, 0.2, 0.4], dtype=np.float32),
        frame="world",
    )

    payload = workspace.as_dict()

    assert payload["min"] == pytest.approx([0.1, -0.2, 0.0])
    assert payload["max"] == pytest.approx([0.5, 0.2, 0.4])
    assert payload["center"] == pytest.approx([0.3, 0.0, 0.2])
    assert payload["half_extents"] == pytest.approx([0.2, 0.2, 0.2])
    assert payload["orientation"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
