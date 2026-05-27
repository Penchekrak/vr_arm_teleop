"""Aero Hand Open actuator-space helpers.

The Aero Hand exposes descriptive 16-joint names, but the hardware command
surface is seven actuators. Keep these helpers pure so planners and drivers
can share the same coupling constants without importing the hardware SDK.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


MOTOR_PULLEY_RADIUS_MM = 9.0

FINGER_MCP_FLEX_COEFF_MM_PER_RAD = 12.4912
FINGER_PIP_COEFF_MM_PER_RAD = 7.3211
FINGER_DIP_COEFF_MM_PER_RAD = 9.0

THUMB_FLEX_CMC_ABD_COEFF_MM_PER_RAD = 2.5
THUMB_FLEX_CMC_FLEX_COEFF_MM_PER_RAD = 12.4931
THUMB_TENDON_CMC_ABD_COEFF_MM_PER_RAD = 2.5
THUMB_TENDON_CMC_FLEX_COEFF_MM_PER_RAD = 2.5
THUMB_TENDON_MCP_COEFF_MM_PER_RAD = 9.4372
THUMB_TENDON_IP_COEFF_MM_PER_RAD = 12.5

AERO_ACTUATION_NAMES: tuple[str, ...] = (
    "thumb_cmc_abd_act",
    "thumb_cmc_flex_act",
    "thumb_tendon_act",
    "index_tendon_act",
    "middle_tendon_act",
    "ring_tendon_act",
    "pinky_tendon_act",
)

AERO_ACTUATION_LOWER_LIMITS_DEG: tuple[float, ...] = (
    0.0,
    0.0,
    -15.2789,
    0.0,
    0.0,
    0.0,
    0.0,
)

AERO_ACTUATION_UPPER_LIMITS_DEG: tuple[float, ...] = (
    100.0,
    104.1250,
    247.1500,
    288.1603,
    288.1603,
    288.1603,
    288.1603,
)

_UINT16_MAX = 65535


@dataclass(frozen=True)
class AeroThumbJointPose:
    """Descriptive thumb joint pose in degrees."""

    cmc_abd_deg: float
    cmc_flex_deg: float
    mcp_deg: float
    ip_deg: float


@dataclass(frozen=True)
class AeroActuatorGoal:
    """Seven Aero actuator targets in SDK/firmware order, degrees."""

    values: tuple[float, ...]

    def __init__(self, values: Iterable[float]) -> None:
        vals = tuple(float(v) for v in values)
        if len(vals) != 7:
            raise ValueError("expected 7 Aero actuator values")
        object.__setattr__(self, "values", vals)

    def clamped(
        self,
        lower: Iterable[float] = AERO_ACTUATION_LOWER_LIMITS_DEG,
        upper: Iterable[float] = AERO_ACTUATION_UPPER_LIMITS_DEG,
    ) -> "AeroActuatorGoal":
        return AeroActuatorGoal(clamp_actuator_degrees(self.values, lower, upper))

    def as_list(self) -> list[float]:
        return list(self.values)


def index_tendon_actuation_rad(
    mcp_flex_rad: float,
    pip_rad: float,
    dip_rad: float,
) -> float:
    """Map index joint angles to the single index tendon actuator angle."""

    tendon_mm = (
        FINGER_MCP_FLEX_COEFF_MM_PER_RAD * float(mcp_flex_rad)
        + FINGER_PIP_COEFF_MM_PER_RAD * float(pip_rad)
        + FINGER_DIP_COEFF_MM_PER_RAD * float(dip_rad)
    )
    return tendon_mm / MOTOR_PULLEY_RADIUS_MM


def index_tendon_actuation_deg(
    mcp_flex: float,
    pip: float,
    dip: float,
    *,
    unit: str = "deg",
) -> float:
    """Map index joint angles to index tendon actuator degrees."""

    mcp_rad, pip_rad, dip_rad = (_to_rad(v, unit) for v in (mcp_flex, pip, dip))
    return math.degrees(index_tendon_actuation_rad(mcp_rad, pip_rad, dip_rad))


def compact_index_actuation_deg(theta: float, *, unit: str = "deg") -> float:
    """Index compact manifold: MCP, PIP, and DIP all share ``theta``."""

    return index_tendon_actuation_deg(theta, theta, theta, unit=unit)


def thumb_actuations_deg(
    *,
    cmc_abd_deg: float,
    cmc_flex_deg: float,
    mcp_deg: float,
    ip_deg: float,
) -> tuple[float, float, float]:
    """Map descriptive thumb joints to the coupled three thumb actuators."""

    cmc_abd_rad = math.radians(float(cmc_abd_deg))
    cmc_flex_rad = math.radians(float(cmc_flex_deg))
    mcp_rad = math.radians(float(mcp_deg))
    ip_rad = math.radians(float(ip_deg))

    thumb_cmc_abd_act = cmc_abd_rad
    thumb_cmc_flex_act = (
        THUMB_FLEX_CMC_ABD_COEFF_MM_PER_RAD * cmc_abd_rad
        + THUMB_FLEX_CMC_FLEX_COEFF_MM_PER_RAD * cmc_flex_rad
    ) / MOTOR_PULLEY_RADIUS_MM
    thumb_tendon_act = (
        THUMB_TENDON_CMC_ABD_COEFF_MM_PER_RAD * cmc_abd_rad
        - THUMB_TENDON_CMC_FLEX_COEFF_MM_PER_RAD * cmc_flex_rad
        + THUMB_TENDON_MCP_COEFF_MM_PER_RAD * mcp_rad
        + THUMB_TENDON_IP_COEFF_MM_PER_RAD * ip_rad
    ) / MOTOR_PULLEY_RADIUS_MM

    return (
        math.degrees(thumb_cmc_abd_act),
        math.degrees(thumb_cmc_flex_act),
        math.degrees(thumb_tendon_act),
    )


def actuator_goal_from_thumb_and_index(
    thumb: AeroThumbJointPose,
    index_theta_rad: float,
) -> AeroActuatorGoal:
    """Build a right-hand two-finger grasp goal; non-participating fingers open."""

    thumb_acts = thumb_actuations_deg(
        cmc_abd_deg=thumb.cmc_abd_deg,
        cmc_flex_deg=thumb.cmc_flex_deg,
        mcp_deg=thumb.mcp_deg,
        ip_deg=thumb.ip_deg,
    )
    index_act = compact_index_actuation_deg(index_theta_rad, unit="rad")
    return AeroActuatorGoal([*thumb_acts, index_act, 0.0, 0.0, 0.0]).clamped()


def clamp_actuator_degrees(
    values: Iterable[float],
    lower: Iterable[float] = AERO_ACTUATION_LOWER_LIMITS_DEG,
    upper: Iterable[float] = AERO_ACTUATION_UPPER_LIMITS_DEG,
) -> tuple[float, ...]:
    vals = np.asarray(tuple(values), dtype=np.float64)
    lo = np.asarray(tuple(lower), dtype=np.float64)
    hi = np.asarray(tuple(upper), dtype=np.float64)
    if vals.shape != (7,) or lo.shape != (7,) or hi.shape != (7,):
        raise ValueError("expected 7 Aero actuator values and limits")
    return tuple(float(v) for v in np.clip(vals, lo, hi))


def actuator_degrees_to_uint16(
    values: Iterable[float],
    lower: Iterable[float] = AERO_ACTUATION_LOWER_LIMITS_DEG,
    upper: Iterable[float] = AERO_ACTUATION_UPPER_LIMITS_DEG,
) -> tuple[int, ...]:
    """Map actuator degrees to the firmware CTRL_POS uint16 range."""

    clamped = np.asarray(clamp_actuator_degrees(values, lower, upper), dtype=np.float64)
    lo = np.asarray(tuple(lower), dtype=np.float64)
    hi = np.asarray(tuple(upper), dtype=np.float64)
    scaled = (clamped - lo) / (hi - lo) * _UINT16_MAX
    return tuple(int(v) for v in np.clip(scaled, 0, _UINT16_MAX))


def _to_rad(value: float, unit: str) -> float:
    if unit == "rad":
        return float(value)
    if unit == "deg":
        return math.radians(float(value))
    raise ValueError("unit must be 'deg' or 'rad'")
