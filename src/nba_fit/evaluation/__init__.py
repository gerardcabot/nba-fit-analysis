"""Backtests and calibration (Phase 5 / Option D)."""

from nba_fit.evaluation.holdout_season import (
    HoldoutSeasonResult,
    holdout_rank_stability,
    next_season,
    run_holdout_season_smoke,
)
from nba_fit.evaluation.movement_backtest import (
    MovementBacktestResult,
    freeze_pre_move_features,
    label_post_move_outcomes,
    run_movement_backtest,
)

__all__ = [
    "HoldoutSeasonResult",
    "MovementBacktestResult",
    "freeze_pre_move_features",
    "holdout_rank_stability",
    "label_post_move_outcomes",
    "next_season",
    "run_holdout_season_smoke",
    "run_movement_backtest",
]
