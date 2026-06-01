"""Golden regression for LeBron × Lakers fit card structure and tolerances."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nba_fit.features.season_context import DEMO_PLAYER_ID, DEMO_TEAM_ID, SeasonFitContext
from nba_fit.models.role_context import RoleFitContext
from nba_fit.scoring.fit_card import build_fit_card
from nba_fit.scoring.fit_index import build_fit_index_table

_GOLDEN = Path(__file__).parent / "golden" / "fit_card_2544_1610612747.json"


@pytest.fixture
def golden_spec() -> dict:
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


def test_golden_fit_card_keys_and_tolerances(golden_spec: dict) -> None:
    context = SeasonFitContext.from_synthetic("2025-26", n_players=80)
    role_ctx = RoleFitContext.from_synthetic(context)
    table = build_fit_index_table(context, synthetic=True, role_context=role_ctx)
    card = build_fit_card(
        DEMO_PLAYER_ID,
        DEMO_TEAM_ID,
        table,
        context,
        role_context=role_ctx,
        lineup_synthetic=True,
    )

    for key in golden_spec["required_keys"]:
        assert key in card, f"missing fit card key: {key}"

    ref = golden_spec["reference"]
    tol = golden_spec["tolerances"]

    assert card["player_id"] == ref["player_id"]
    assert card["team_id"] == ref["team_id"]

    assert card.get("overall_fit_percentile") is not None
    assert 0.0 <= float(card["overall_fit_percentile"]) <= 100.0
    assert card.get("raw_fit_score") is not None
    assert 0.0 <= float(card["raw_fit_score"]) <= 1.0

    assert card.get("soft_role_display")
    assert isinstance(card.get("industry_role_probs"), dict)
    assert card.get("industry_role")

    if card.get("data_source") == "interim":
        if card.get("overall_fit_percentile") is not None:
            assert card["overall_fit_percentile"] == pytest.approx(
                ref["overall_fit_percentile"],
                abs=tol["overall_fit_percentile"],
            )
        if card.get("raw_fit_score") is not None:
            assert card["raw_fit_score"] == pytest.approx(
                ref["raw_fit_score"],
                abs=tol["raw_fit_score"],
            )
