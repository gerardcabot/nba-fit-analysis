"""Option C lineup simulation tests (synthetic / offline)."""

from __future__ import annotations

import pytest

from nba_fit.features.season_context import DEMO_PLAYER_ID, DEMO_TEAM_ID
from nba_fit.models.lineup_impact import (
    aggregate_projected_delta,
    project_lineup_replacement,
)
from nba_fit.scoring.fit_card import build_fit_card
from nba_fit.scoring.fit_index import build_fit_index_table
from nba_fit.scoring.lineup_sim import run_lineup_sim


def test_project_lineup_replacement_positive_delta() -> None:
    impact = {1: 2.0, 2: 1.0, 3: 0.0, 4: -1.0, 5: -2.0}
    proj = project_lineup_replacement(
        lineup_player_ids=(1, 2, 3, 4, 5),
        candidate_id=99,
        impact_by_player={**impact, 99: 8.0},
        baseline_net_rating=4.0,
        minutes=100.0,
    )
    assert proj is not None
    assert proj.replaced_player_id == 5
    assert proj.projected_net_rating_delta > 0


def test_run_lineup_sim_synthetic_returns_units() -> None:
    result = run_lineup_sim(
        DEMO_PLAYER_ID,
        DEMO_TEAM_ID,
        "2025-26",
        top_n=3,
        synthetic=True,
    )
    assert result.top_lineups
    assert len(result.top_lineups) <= 3
    assert result.projected_net_rating_delta is not None


def test_aggregate_projected_delta_weighted() -> None:
    from nba_fit.models.lineup_impact import LineupUnitProjection

    units = [
        LineupUnitProjection("a", "a", (1, 2, 3, 4, 5), 0.0, 2.0, 100.0, 5),
        LineupUnitProjection("b", "b", (1, 2, 3, 4, 6), 0.0, 4.0, 50.0, 6),
    ]
    agg = aggregate_projected_delta(units)
    assert agg == pytest.approx((2.0 * 100 + 4.0 * 50) / 150, rel=1e-6)


def test_fit_card_includes_lineup_synergy() -> None:
    from nba_fit.features.season_context import SeasonFitContext

    context = SeasonFitContext.from_synthetic("2025-26", n_players=40)
    table = build_fit_index_table(context)
    card = build_fit_card(
        DEMO_PLAYER_ID,
        DEMO_TEAM_ID,
        table,
        context,
        lineup_synthetic=True,
    )
    assert "lineup_synergy" in card
    assert "projected_net_rating_delta" in card
    assert card["lineup_synergy"]["top_lineups"]
