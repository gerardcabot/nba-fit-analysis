"""Synthetic tests for Option A fit index and submetrics."""

from __future__ import annotations

import numpy as np
import pytest

from nba_fit.features.season_context import DEMO_PLAYER_ID, DEMO_TEAM_ID, SeasonFitContext
from nba_fit.scoring.constants import SUBMETRIC_NAMES, SUBMETRIC_WEIGHTS
from nba_fit.scoring.fit_card import build_fit_card
from nba_fit.scoring.fit_index import build_fit_index_table, raw_scores_to_percentiles
from nba_fit.scoring.ranker import FitRanker
from nba_fit.scoring.submetrics import compute_all_submetrics, weighted_raw_score


@pytest.fixture
def context() -> SeasonFitContext:
    return SeasonFitContext.from_synthetic("2025-26", n_players=40)


def test_submetrics_in_unit_interval(context: SeasonFitContext) -> None:
    player = context.players[DEMO_PLAYER_ID]
    team = context.teams[DEMO_TEAM_ID]
    sub = compute_all_submetrics(player, team)
    assert set(sub.keys()) == set(SUBMETRIC_NAMES)
    for name, value in sub.items():
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"


def test_submetric_weights_sum_to_one() -> None:
    assert abs(sum(SUBMETRIC_WEIGHTS.values()) - 1.0) < 1e-9


def test_percentile_calibration_monotonic() -> None:
    raw = np.array([0.1, 0.2, 0.2, 0.9])
    pct = raw_scores_to_percentiles(raw)
    assert pct.min() > 0
    assert pct.max() < 100
    assert pct[np.argmax(raw)] == max(pct)


def test_fit_index_table_has_all_pairs(context: SeasonFitContext) -> None:
    table = build_fit_index_table(context)
    n_players = len(context.players)
    n_teams = len(context.teams)
    assert len(table.pairs) == n_players * n_teams
    assert table.is_calibrated
    assert table.pairs["overall_fit_percentile"].between(0, 100).all()


def test_rank_player_includes_demo_ids(context: SeasonFitContext) -> None:
    ranker = FitRanker(context=context, table=build_fit_index_table(context))
    dest = ranker.rank_destinations_for_player(DEMO_PLAYER_ID, top_n=5)
    assert len(dest) == 5
    assert dest.iloc[0]["rank"] == 1
    assert "overall_fit_percentile" in dest.columns


def test_rank_team_orders_players(context: SeasonFitContext) -> None:
    ranker = FitRanker(context=context, table=build_fit_index_table(context))
    targets = ranker.rank_players_for_team(DEMO_TEAM_ID, top_n=5)
    assert len(targets) == 5
    pct = targets["overall_fit_percentile"].tolist()
    assert pct == sorted(pct, reverse=True)


def test_fit_card_structure(context: SeasonFitContext) -> None:
    table = build_fit_index_table(context)
    card = build_fit_card(DEMO_PLAYER_ID, DEMO_TEAM_ID, table, context)
    assert card["player_id"] == DEMO_PLAYER_ID
    assert card["team_id"] == DEMO_TEAM_ID
    assert "comps" in card
    assert "archetype" in card
    assert set(card["submetrics"].keys()) == set(SUBMETRIC_NAMES)
    assert card["overall_fit_percentile"] is not None


def test_weighted_raw_score_matches_manual(context: SeasonFitContext) -> None:
    player = context.players[DEMO_PLAYER_ID]
    team = context.teams[DEMO_TEAM_ID]
    sub = compute_all_submetrics(player, team)
    expected = sum(SUBMETRIC_WEIGHTS[k] * sub[k] for k in SUBMETRIC_NAMES)
    assert weighted_raw_score(sub, SUBMETRIC_WEIGHTS) == pytest.approx(expected, rel=1e-6)
