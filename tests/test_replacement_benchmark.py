"""Tests for replacement-player benchmark."""

from __future__ import annotations

from nba_fit.data.fetchers.transactions import synthetic_movements
from nba_fit.evaluation.replacement_benchmark import rank_signing_vs_position_pool
from nba_fit.features.season_context import SeasonFitContext
from nba_fit.scoring.fit_index import build_fit_index_table


def test_replacement_benchmark_ranks_signing_in_pool() -> None:
    context = SeasonFitContext.from_synthetic("2024-25", n_players=30)
    movements = synthetic_movements("2024-25", n_moves=3)
    table = build_fit_index_table(context, synthetic=True)
    result = rank_signing_vs_position_pool(movements, context, table)
    assert result.n_movements >= 1
    for row in result.rows:
        assert 0.0 < row.pool_rank_pct <= 1.0
        assert row.pool_size >= 1
