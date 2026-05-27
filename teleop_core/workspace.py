"""Workspace volume + safety geometry.

The workspace is the axis-aligned box in world coordinates that the
robot is allowed to move within. The tracker clamps every commanded
pose against it; the safety layer raises a warning when the operator's
wrist exits it.

Keeping this independent of the robot/tracker makes it trivial to:
- Render the box in VR for the operator (we serialize it to the client).
- Visualise it offline (matplotlib, blender, etc.) for setup.
- Unit-test clamping logic without spinning up anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Workspace:
    """Oriented box in the world frame.

    ``min_corner`` / ``max_corner`` are kept as the axis-aligned bounding box
    for compatibility with older clients and configs. The authoritative
    safety geometry is ``center`` + ``half_extents`` + ``orientation``.
    """

    min_corner: np.ndarray | None = None   # shape (3,), meters
    max_corner: np.ndarray | None = None   # shape (3,), meters
    frame: str = "world"
    center: np.ndarray | None = None       # shape (3,), meters
    half_extents: np.ndarray | None = None # shape (3,), meters
    orientation: np.ndarray | None = None  # xyzw quaternion

    def __post_init__(self) -> None:
        orientation = _as_quat(
            [0.0, 0.0, 0.0, 1.0]
            if self.orientation is None
            else self.orientation,
            "orientation",
        )
        has_oriented = self.center is not None or self.half_extents is not None
        if has_oriented:
            if self.center is None or self.half_extents is None:
                raise ValueError("workspace center and half_extents must be provided together")
            center = _as_vec3(self.center, "center")
            half_extents = _as_vec3(self.half_extents, "half_extents")
            if np.any(half_extents < 0.0):
                raise ValueError("workspace half_extents must be non-negative")
            min_corner, max_corner = _aabb_for_oriented_box(
                center,
                half_extents,
                _quat_to_matrix(orientation),
            )
        else:
            if self.min_corner is None or self.max_corner is None:
                raise ValueError("workspace requires either min/max or center/half_extents")
            min_corner = _as_vec3(self.min_corner, "min_corner")
            max_corner = _as_vec3(self.max_corner, "max_corner")
            if np.any(max_corner < min_corner):
                raise ValueError("workspace max_corner must be >= min_corner")
            center = (min_corner + max_corner) / 2.0
            half_extents = (max_corner - min_corner) / 2.0

        object.__setattr__(self, "min_corner", min_corner.astype(np.float64))
        object.__setattr__(self, "max_corner", max_corner.astype(np.float64))
        object.__setattr__(self, "center", center.astype(np.float64))
        object.__setattr__(self, "half_extents", half_extents.astype(np.float64))
        object.__setattr__(self, "orientation", orientation.astype(np.float64))

    @classmethod
    def from_dict(cls, data: dict) -> "Workspace":
        """Build from either legacy ``min``/``max`` or oriented-box fields."""
        if "center" in data or "half_extents" in data:
            return cls(
                center=np.asarray(data["center"], dtype=np.float64),
                half_extents=np.asarray(data["half_extents"], dtype=np.float64),
                orientation=np.asarray(
                    data.get("orientation", data.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])),
                    dtype=np.float64,
                ),
                frame=str(data.get("frame", "world")),
            )
        return cls(
            min_corner=np.asarray(data["min"], dtype=np.float64),
            max_corner=np.asarray(data["max"], dtype=np.float64),
            frame=str(data.get("frame", "world")),
        )

    def contains(self, point: np.ndarray) -> bool:
        """True iff ``point`` is inside the box (boundary inclusive)."""
        local = self._to_local(point)
        eps = 1e-9
        return bool(
            np.all(local >= -self.half_extents - eps)
            and np.all(local <= self.half_extents + eps)
        )

    def clamp(self, point: np.ndarray) -> tuple[np.ndarray, bool]:
        """Clamp ``point`` into the box.

        Returns
        -------
        clamped : np.ndarray
            The closest point inside the box (component-wise clamp).
        was_outside : bool
            True iff the input was strictly outside; useful for triggering
            "out of workspace" warnings without re-comparing afterwards.
        """
        local = self._to_local(point)
        clamped_local = np.minimum(
            np.maximum(local, -self.half_extents),
            self.half_extents,
        )
        was_outside = bool(
            np.any(local < -self.half_extents)
            or np.any(local > self.half_extents)
        )
        clamped = self.center + _quat_to_matrix(self.orientation) @ clamped_local
        return clamped, was_outside

    def as_dict(self) -> dict:
        """JSON-friendly representation, for sending to the client."""
        return {
            "min": [float(v) for v in self.min_corner],
            "max": [float(v) for v in self.max_corner],
            "center": [float(v) for v in self.center],
            "half_extents": [float(v) for v in self.half_extents],
            "orientation": [float(v) for v in self.orientation],
            "frame": self.frame,
        }

    def _to_local(self, point: np.ndarray) -> np.ndarray:
        p = _as_vec3(point, "point")
        return _quat_to_matrix(self.orientation).T @ (p - self.center)


def _as_vec3(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite values")
    return arr


def _as_quat(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f"{name} must be a quaternion in xyzw order")
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise ValueError(f"{name} must be a non-zero finite quaternion")
    return arr / norm


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _aabb_for_oriented_box(
    center: np.ndarray,
    half_extents: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    extents = np.abs(rotation) @ half_extents
    return center - extents, center + extents
