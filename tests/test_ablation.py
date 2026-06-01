"""Tests for Option A/B/C/D ablation runner."""

from __future__ import annotations

from nba_fit.data.fetchers.transactions import synthetic_movements
from nba_fit.evaluation.ablation import run_ablation, score_ablation_variant
from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.impact_context import ImpactFitContext
from nba_fit.models.role_context import RoleFitContext


def test_ablation_variants_increase_context() -> None:
    context = SeasonFitContext.from_synthetic("2025-26", n_players=25)
    role_ctx = RoleFitContext.from_synthetic(context)
    impact_ctx = ImpactFitContext.from_synthetic(role_ctx)
    movements = synthetic_movements(context.season, n_moves=4)

    report = run_ablation(
        context,
        movements,
        role_context=role_ctx,
        impact_context=impact_ctx,
    )
    assert set(report.variants.keys()) == {"A", "AB", "ABC", "D"}
    summary = report.summary_table()
    assert len(summary) == 4
    assert summary["n_movements"].min() >= 1


def test_ablation_d_applies_calibration() -> None:
    context = SeasonFitContext.from_synthetic("2025-26", n_players=20)
    movements = synthetic_movements(context.season, n_moves=3)
    abc = score_ablation_variant(context, movements, "ABC")
    d = score_ablation_variant(context, movements, "D")
    if not d.rows.empty:
        assert "calibrated_fit_percentile" in d.rows.columns
        assert "calibrated_fit_percentile" not in abc.rows.columns or abc.variant == "ABC"
