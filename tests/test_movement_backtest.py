"""Phase 5 movement backtest, calibration, and uncertainty tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_fit.data.fetchers.transactions import (
    MovementEvent,
    ensure_stub_template,
    infer_movements_from_gamelogs,
    load_manual_movements,
    synthetic_movements,
)
from nba_fit.evaluation.movement_backtest import (
    freeze_pre_move_features,
    label_post_move_outcomes,
    run_movement_backtest,
)
from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.calibration import FitCalibrator, calibrate_fit_table
from nba_fit.scoring.uncertainty import (
    bootstrap_disagreement_ci,
    fit_uncertainty,
    sample_size_penalty,
    widen_ci_for_sample,
)


@pytest.fixture
def context() -> SeasonFitContext:
    return SeasonFitContext.from_synthetic("2025-26", n_players=30)


def test_synthetic_movements_schema() -> None:
    mv = synthetic_movements("2025-26", n_moves=2)
    assert len(mv) == 2
    assert set(mv.columns) >= {"player_id", "to_team_id", "move_date"}


def test_infer_team_change_from_gamelogs() -> None:
    logs = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1, 2],
            "TEAM_ID": [10, 10, 20, 20],
            "GAME_DATE": ["2024-10-01", "2024-10-15", "2024-11-01", "2024-10-20"],
        }
    )
    mv = infer_movements_from_gamelogs(logs, season="2024-25")
    assert len(mv) == 1
    assert mv.iloc[0]["player_id"] == 1
    assert mv.iloc[0]["from_team_id"] == 10
    assert mv.iloc[0]["to_team_id"] == 20


def test_stub_template_created() -> None:
    path = ensure_stub_template()
    assert path.exists()
    manual = load_manual_movements()
    assert isinstance(manual, pd.DataFrame)


def test_freeze_pre_move_features(context: SeasonFitContext) -> None:
    mv = synthetic_movements(context.season, n_moves=2)
    frozen = freeze_pre_move_features(context, mv)
    assert len(frozen) >= 1
    assert "raw_fit_score" in frozen.columns
    assert frozen["raw_fit_score"].between(0, 1).all()


def test_label_post_move_outcomes(context: SeasonFitContext) -> None:
    mv = synthetic_movements(context.season, n_moves=2)
    frozen = freeze_pre_move_features(context, mv)
    labeled = label_post_move_outcomes(frozen, mv, context)
    assert "post_move_minutes" in labeled.columns
    assert "post_move_outcome" in labeled.columns
    assert labeled["post_move_outcome"].between(0, 1).all()


def test_run_movement_backtest_calibrates(context: SeasonFitContext) -> None:
    result = run_movement_backtest(context, synthetic_movements(context.season))
    assert result.n_movements >= 1
    assert "calibrated_fit_percentile" in result.rows.columns
    assert "uncertainty_ci_low" in result.rows.columns
    cal = result.calibration_frame()
    assert "post_move_outcome" in cal.columns


def test_isotonic_calibrator_monotonic() -> None:
    raw = np.linspace(0.1, 0.9, 20)
    outcome = raw**1.5 * 100
    cal = FitCalibrator(method="isotonic")
    pct = cal.fit_transform(raw, outcome)
    assert pct.min() >= 0
    assert pct.max() <= 100
    assert pct[np.argmax(raw)] >= pct[np.argmin(raw)]


def test_percentile_calibrator() -> None:
    raw = np.array([0.2, 0.2, 0.5, 0.9])
    cal = FitCalibrator(method="percentile")
    pct = cal.fit_transform(raw)
    assert len(pct) == 4
    assert pct.max() > pct.min()


def test_calibrate_fit_table() -> None:
    df = pd.DataFrame(
        {
            "raw_fit_score": [0.2, 0.5, 0.8],
            "post_move_outcome": [20.0, 50.0, 80.0],
        }
    )
    out = calibrate_fit_table(df, method="isotonic")
    assert "calibrated_fit_percentile" in out.columns


def test_sample_size_penalty_increases_when_low_minutes() -> None:
    low = sample_size_penalty(100.0)
    high = sample_size_penalty(2000.0)
    assert low > high


def test_bootstrap_ci_contains_point() -> None:
    scores = np.array([[0.4, 0.5, 0.6, 0.55]])
    point, lo, hi = bootstrap_disagreement_ci(scores, n_bootstrap=100)
    assert lo <= point <= hi


def test_widen_ci_for_low_sample() -> None:
    lo, hi, pen = widen_ci_for_sample(0.4, 0.6, minutes=50.0)
    assert pen > 0.5
    assert hi - lo > 0.2


def test_fit_uncertainty_result_fields() -> None:
    scores = np.array([[0.3, 0.5, 0.7]])
    u = fit_uncertainty(scores, minutes=200.0)
    assert u.ci_low <= u.point <= u.ci_high
    assert u.sample_penalty > 0


def test_movement_event_roundtrip() -> None:
    ev = MovementEvent(
        player_id=1,
        from_team_id=10,
        to_team_id=20,
        move_date=pd.Timestamp("2024-07-01"),
        season="2024-25",
    )
    d = ev.to_dict()
    assert d["player_id"] == 1
