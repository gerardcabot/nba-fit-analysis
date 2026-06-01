"""Tests for calibration metrics (ECE, Brier, decile lift, Spearman)."""

from __future__ import annotations

import numpy as np
import pytest

from nba_fit.evaluation.calibration_metrics import (
    brier_score,
    calibration_report,
    decile_lift,
    expected_calibration_error,
    spearman_rank_corr,
)


def test_brier_score_perfect_and_worst() -> None:
    y = np.array([0.0, 1.0, 1.0, 0.0])
    assert brier_score(y, y) == pytest.approx(0.0)
    assert brier_score(y, 1.0 - y) == pytest.approx(1.0)


def test_ece_well_calibrated_near_zero() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.2, 0.8, size=200)
    y = (rng.uniform(0, 1, size=200) < p).astype(float)
    ece = expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.15


def test_ece_miscalibrated_higher() -> None:
    y = np.array([1.0, 1.0, 0.0, 0.0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert expected_calibration_error(y, p, n_bins=2) > 0.3


def test_decile_lift_positive_when_scores_rank_outcomes() -> None:
    y = np.linspace(0, 1, 100)
    s = y + np.random.default_rng(1).normal(0, 0.01, size=100)
    lift = decile_lift(y, s, n_deciles=10)
    assert lift > 0.0


def test_spearman_monotonic() -> None:
    y = np.arange(20, dtype=float)
    s = y * 2 + 1
    assert spearman_rank_corr(y, s) == pytest.approx(1.0)


def test_calibration_report_bins() -> None:
    y = np.array([0.0, 1.0, 0.5, 0.5])
    p = np.array([0.1, 0.9, 0.4, 0.6])
    report = calibration_report(y, p, n_bins=2)
    assert len(report) == 2
    assert report["count"].sum() == 4
