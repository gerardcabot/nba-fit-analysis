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
    table = build_fit_index_table(context, synthetic=True)
    card = build_fit_card(
        DEMO_PLAYER_ID,
        DEMO_TEAM_ID,
        table,
        context,
        lineup_synthetic=True,
    )
    assert card["player_id"] == DEMO_PLAYER_ID
    assert card["team_id"] == DEMO_TEAM_ID
    assert "comps" in card
    assert "archetype" in card
    assert "lineup_synergy" in card
    assert "projected_net_rating_delta" in card
    assert "team_need_fit" in card
    assert card["role_fit"] == card["team_need_fit"]
    assert set(card["submetrics"].keys()) == set(SUBMETRIC_NAMES)
    assert card["overall_fit_percentile"] is not None
    assert "fit_uncertainty" in card
    assert "ensemble" in card
    assert "ensemble_contributions" in card
    assert "components_degraded" in card
    assert "fallbacks" in card
    lineup = card["lineup_synergy"]
    assert lineup is not None
    assert "headline" in lineup
    assert "projected_net_rating_delta" in lineup["headline"]


def test_fit_card_reports_degraded_without_role_context(
    context: SeasonFitContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nba_fit.scoring.fit_index as fit_index_mod

    monkeypatch.setattr(fit_index_mod, "_resolve_role_context", lambda *_a, **_k: None)
    monkeypatch.setattr(fit_index_mod, "_resolve_impact_context", lambda *_a, **_k: None)

    table = build_fit_index_table(context)
    card = build_fit_card(
        DEMO_PLAYER_ID,
        DEMO_TEAM_ID,
        table,
        context,
        lineup_synthetic=True,
    )
    degraded = card["components_degraded"]
    assert "team_need_fit" in degraded
    assert "lineup_impact_fit" in degraded
    assert card["fallbacks"] == degraded


def test_build_fit_index_table_tries_from_season_before_synthetic(
    context: SeasonFitContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nba_fit.models.impact_context import ImpactFitContext
    from nba_fit.models.role_context import RoleFitContext

    role_calls: list[str] = []
    impact_calls: list[str] = []
    orig_role_synthetic = RoleFitContext.from_synthetic
    orig_impact_synthetic = ImpactFitContext.from_synthetic

    def fake_role_from_season(*_args, **_kwargs):
        role_calls.append("from_season")
        return orig_role_synthetic(context)

    def fake_role_from_synthetic(_ctx):
        role_calls.append("from_synthetic")
        raise ValueError("skip synthetic in test")

    def fake_impact_from_season(*_args, **_kwargs):
        impact_calls.append("from_season")
        rc = orig_role_synthetic(context)
        return orig_impact_synthetic(rc)

    def fake_impact_from_synthetic(_rc):
        impact_calls.append("from_synthetic")
        raise ValueError("skip synthetic in test")

    monkeypatch.setattr(RoleFitContext, "from_season", fake_role_from_season)
    monkeypatch.setattr(RoleFitContext, "from_synthetic", fake_role_from_synthetic)
    monkeypatch.setattr(ImpactFitContext, "from_season", fake_impact_from_season)
    monkeypatch.setattr(ImpactFitContext, "from_synthetic", fake_impact_from_synthetic)

    build_fit_index_table(context, synthetic=False)

    assert role_calls == ["from_season"]
    assert impact_calls == ["from_season"]


def test_fit_ranker_loads_persisted_contexts(
    context: SeasonFitContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nba_fit.models.impact_context import ImpactFitContext
    from nba_fit.models.role_context import RoleFitContext

    role_loaded = False
    impact_loaded = False

    def fake_build(season=None, **kwargs):
        return context

    def fake_role_from_season(*_args, **_kwargs):
        nonlocal role_loaded
        role_loaded = True
        return RoleFitContext.from_synthetic(context)

    def fake_impact_from_season(*_args, **_kwargs):
        nonlocal impact_loaded
        impact_loaded = True
        rc = RoleFitContext.from_synthetic(context)
        return ImpactFitContext.from_synthetic(rc)

    monkeypatch.setattr(SeasonFitContext, "build", staticmethod(fake_build))
    monkeypatch.setattr(RoleFitContext, "from_season", fake_role_from_season)
    monkeypatch.setattr(ImpactFitContext, "from_season", fake_impact_from_season)

    ranker = FitRanker.from_season(context.season, prefer_interim=True, synthetic=False)

    assert role_loaded
    assert impact_loaded
    assert not ranker.table.pairs.empty


def test_weighted_raw_score_matches_manual(context: SeasonFitContext) -> None:
    player = context.players[DEMO_PLAYER_ID]
    team = context.teams[DEMO_TEAM_ID]
    sub = compute_all_submetrics(player, team)
    expected = sum(SUBMETRIC_WEIGHTS[k] * sub[k] for k in SUBMETRIC_NAMES)
    assert weighted_raw_score(sub, SUBMETRIC_WEIGHTS) == pytest.approx(expected, rel=1e-6)
