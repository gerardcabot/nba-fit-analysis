"""Tests for Option D ensemble and evaluation smoke."""

from __future__ import annotations

import pytest

from nba_fit.evaluation.holdout_season import next_season, run_holdout_season_smoke
from nba_fit.data.fetchers.transactions import synthetic_movements
from nba_fit.evaluation.movement_backtest import run_movement_backtest
from nba_fit.features.season_context import DEMO_PLAYER_ID, DEMO_TEAM_ID, SeasonFitContext
from nba_fit.scoring.constants import (
    ENSEMBLE_COMPONENT_WEIGHTS,
    PROFILE_SUBMETRIC_WEIGHTS,
)
from nba_fit.scoring.ensemble import (
    calibrated_ensemble,
    extract_ensemble_components,
    raw_ensemble_score,
    uncertainty_band,
)
from nba_fit.scoring.fit_card import build_fit_card
from nba_fit.scoring.fit_index import build_fit_index_table
from nba_fit.scoring.submetrics import compute_all_submetrics


@pytest.fixture
def context() -> SeasonFitContext:
    return SeasonFitContext.from_synthetic("2025-26", n_players=40)


def test_ensemble_weights_sum_to_one() -> None:
    assert abs(sum(ENSEMBLE_COMPONENT_WEIGHTS.values()) - 1.0) < 1e-9
    assert abs(sum(PROFILE_SUBMETRIC_WEIGHTS.values()) - 1.0) < 1e-9


def test_raw_ensemble_in_unit_interval(context: SeasonFitContext) -> None:
    player = context.players[DEMO_PLAYER_ID]
    team = context.teams[DEMO_TEAM_ID]
    sub = compute_all_submetrics(player, team)
    comps = extract_ensemble_components(sub, player=player)
    raw = raw_ensemble_score(comps)
    assert 0.0 <= raw <= 1.0


def test_calibrated_ensemble_api() -> None:
    raw, comps = calibrated_ensemble(0.7, 0.6, 0.8, 0.75, 0.65, 0.1)
    assert 0.0 <= raw <= 1.0
    assert comps["risk_penalty"] == 0.1


def test_fit_index_has_ensemble_and_uncertainty(context: SeasonFitContext) -> None:
    table = build_fit_index_table(context)
    assert table.is_calibrated
    assert "ensemble_profile_fit" in table.pairs.columns
    assert "contrib_profile_fit" in table.pairs.columns
    row = table.pairs.loc[
        (table.pairs["player_id"] == DEMO_PLAYER_ID)
        & (table.pairs["team_id"] == DEMO_TEAM_ID)
    ].iloc[0]
    assert row["fit_uncertainty_low"] <= row["overall_fit_percentile"]
    assert row["fit_uncertainty_high"] >= row["overall_fit_percentile"]


def test_fit_card_uncertainty_bands(context: SeasonFitContext) -> None:
    table = build_fit_index_table(context)
    card = build_fit_card(
        DEMO_PLAYER_ID,
        DEMO_TEAM_ID,
        table,
        context,
        lineup_synthetic=True,
    )
    unc = card["fit_uncertainty"]
    assert unc["low_percentile"] is not None
    assert unc["high_percentile"] is not None
    assert "ensemble" in card
    assert "ensemble_contributions" in card


def test_uncertainty_band_clamps() -> None:
    low, high = uncertainty_band(2.0, {"profile_fit": 0.2, "role_fit": 0.9})
    assert low >= 0.0
    assert high <= 100.0


def test_holdout_season_smoke() -> None:
    result = run_holdout_season_smoke("2024-25", synthetic=True)
    assert result.predict_season == next_season("2024-25")
    assert result.train_pairs > 0
    assert result.predict_pairs > 0


def test_movement_backtest_smoke(context: SeasonFitContext) -> None:
    movements = synthetic_movements(context.season, n_moves=3)
    result = run_movement_backtest(context, movements)
    assert result.n_movements >= 1
    assert "calibrated_fit_percentile" in result.rows.columns
