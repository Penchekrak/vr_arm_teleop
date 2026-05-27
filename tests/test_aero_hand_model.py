import math

import pytest

from teleop_core.aero_hand_model import (
    AERO_ACTUATION_LOWER_LIMITS_DEG,
    AERO_ACTUATION_UPPER_LIMITS_DEG,
    AeroActuatorGoal,
    clamp_actuator_degrees,
    compact_index_actuation_deg,
    thumb_actuations_deg,
)


def test_compact_index_actuation_matches_aero_upstream_limits():
    assert compact_index_actuation_deg(0.0) == pytest.approx(0.0)
    assert compact_index_actuation_deg(90.0) == pytest.approx(288.1603, abs=0.05)


def test_compact_index_actuation_accepts_radians_when_requested():
    assert compact_index_actuation_deg(math.pi / 2.0, unit="rad") == pytest.approx(
        288.1603,
        abs=0.05,
    )


def test_clamp_actuator_degrees_uses_aero_actuation_limits():
    values = [-10.0, 200.0, -100.0, 500.0, 0.0, 0.0, 999.0]

    clamped = clamp_actuator_degrees(values)

    assert clamped == pytest.approx(
        [
            AERO_ACTUATION_LOWER_LIMITS_DEG[0],
            AERO_ACTUATION_UPPER_LIMITS_DEG[1],
            AERO_ACTUATION_LOWER_LIMITS_DEG[2],
            AERO_ACTUATION_UPPER_LIMITS_DEG[3],
            0.0,
            0.0,
            AERO_ACTUATION_UPPER_LIMITS_DEG[6],
        ]
    )


def test_thumb_actuations_are_generated_by_coupled_upstream_model():
    actuators = thumb_actuations_deg(
        cmc_abd_deg=45.0,
        cmc_flex_deg=30.0,
        mcp_deg=20.0,
        ip_deg=10.0,
    )

    assert actuators == pytest.approx(
        [
            45.0,
            (2.5 * 45.0 + 12.4931 * 30.0) / 9.0,
            (2.5 * 45.0 - 2.5 * 30.0 + 9.4372 * 20.0 + 12.5 * 10.0) / 9.0,
        ]
    )


def test_aero_actuator_goal_rejects_wrong_length_and_clamps():
    with pytest.raises(ValueError, match="7 Aero actuator"):
        AeroActuatorGoal([0.0, 1.0])

    goal = AeroActuatorGoal(
        [-10.0, 200.0, -100.0, 500.0, 0.0, 0.0, 999.0]
    ).clamped()

    assert goal.values[0] == AERO_ACTUATION_LOWER_LIMITS_DEG[0]
    assert goal.values[1] == AERO_ACTUATION_UPPER_LIMITS_DEG[1]
    assert goal.values[2] == AERO_ACTUATION_LOWER_LIMITS_DEG[2]
    assert goal.values[3] == AERO_ACTUATION_UPPER_LIMITS_DEG[3]
    assert goal.values[6] == AERO_ACTUATION_UPPER_LIMITS_DEG[6]
