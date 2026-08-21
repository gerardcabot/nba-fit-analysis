"""Backtests and calibration (Phase 5 / Option D)."""

from nba_fit.evaluation.ablation import (
    AblationReport,
    ablation_report_to_dict,
    run_ablation,
    write_ablation_metrics,
)
from nba_fit.evaluation.calibration_metrics import (
    HoldoutCalibrationResult,
    evaluate_holdout_calibration,
    fit_isotonic_on_train_rows,
    split_movements_by_season,
)
from nba_fit.evaluation.holdout_season import (
    HoldoutSeasonResult,
    holdout_rank_stability,
    next_season,
    run_holdout_season,
    run_holdout_season_smoke,
)
from nba_fit.evaluation.movement_backtest import (
    MovementBacktestResult,
    freeze_pre_move_features,
    label_post_move_outcomes,
    run_movement_backtest,
)
from nba_fit.evaluation.replacement_benchmark import (
    ReplacementBenchmarkResult,
    rank_signing_vs_position_pool,
)

__all__ = [
    "AblationReport",
    "HoldoutCalibrationResult",
    "HoldoutSeasonResult",
    "MovementBacktestResult",
    "ReplacementBenchmarkResult",
    "ablation_report_to_dict",
    "evaluate_holdout_calibration",
    "fit_isotonic_on_train_rows",
    "freeze_pre_move_features",
    "holdout_rank_stability",
    "label_post_move_outcomes",
    "next_season",
    "rank_signing_vs_position_pool",
    "run_ablation",
    "run_holdout_season",
    "run_holdout_season_smoke",
    "run_movement_backtest",
    "split_movements_by_season",
    "write_ablation_metrics",
]
