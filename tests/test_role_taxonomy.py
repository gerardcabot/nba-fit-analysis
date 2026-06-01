"""Unit tests for industry role taxonomy mapping."""

from __future__ import annotations

import pytest

from nba_fit.models.constants import ARCHETYPE_NOISE_LABEL
from nba_fit.models.role_taxonomy import (
    CTG_POSITION_GROUPS,
    INDUSTRY_OFFENSIVE_ROLES,
    ctg_position_group,
    load_archetype_industry_map,
    map_heuristic_to_industry,
    soft_role_display,
)


@pytest.fixture
def industry_map():
    return load_archetype_industry_map()


def test_industry_map_csv_loads(industry_map) -> None:
    assert len(industry_map) >= 10
    assert {"heuristic_label", "industry_role", "ctg_position_group"}.issubset(industry_map.columns)


@pytest.mark.parametrize(
    ("heuristic", "expected"),
    [
        ("high_usage_creator", "Primary Ball Handler"),
        ("advantage_creator", "Shot Creator"),
        ("movement_shooter", "Movement Shooter"),
        ("low_usage_spacer", "Spot Up Wing"),
        ("connector_wing", "Connector"),
        ("rim_running_big", "Roll/Cut Big"),
        ("stretch_big", "Spot Up Big"),
        ("defensive_guard", "Combo Guard"),
        ("rim_protector", "Versatile Big"),
        ("rebound_finisher", "Roll/Cut Big"),
        ("versatile_forward", "Connector"),
        (ARCHETYPE_NOISE_LABEL, "Unknown"),
    ],
)
def test_map_heuristic_to_industry(heuristic: str, expected: str) -> None:
    assert map_heuristic_to_industry(heuristic) == expected
    assert expected in INDUSTRY_OFFENSIVE_ROLES


def test_map_heuristic_unknown_label() -> None:
    assert map_heuristic_to_industry("nonexistent_archetype") == "Unknown"


@pytest.mark.parametrize(
    ("role_bucket", "expected_group"),
    [
        ("Primary Ball Handler", "Guard"),
        ("Shot Creator", "Guard"),
        ("Movement Shooter", "Wing"),
        ("Connector", "Wing"),
        ("Roll/Cut Big", "Big"),
        ("Versatile Big", "Big"),
        ("high_usage_creator", "Guard"),
        ("stretch_big", "Big"),
        ("Unknown", "Unknown"),
    ],
)
def test_ctg_position_group(role_bucket: str, expected_group: str) -> None:
    assert ctg_position_group(role_bucket) == expected_group
    assert expected_group in CTG_POSITION_GROUPS


def test_soft_role_display_mapping() -> None:
    text = soft_role_display(
        {
            "Shot Creator": 0.62,
            "Connector": 0.24,
            "Movement Shooter": 0.14,
        },
        top_n=2,
    )
    assert "Shot Creator (62%)" in text
    assert "Connector (24%)" in text
    assert "Movement Shooter" not in text


def test_soft_role_display_sequence_and_empty() -> None:
    assert soft_role_display([("Primary Ball Handler", 1.0)]) == "Primary Ball Handler (100%)"
    assert soft_role_display({}) == "Unknown"
