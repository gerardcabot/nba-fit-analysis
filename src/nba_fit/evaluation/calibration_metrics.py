"""Calibration and ranking metrics for fit-score validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Mean squared error between binary outcomes and predicted probabilities.

    Both inputs are clipped to [0, 1].
    """
    y = np.clip(np.asarray(y_true, dtype=float), 0.0, 1.0)
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if y.size == 0:
        return float("nan")
    return float(np.mean((y - p) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """
    Expected calibration error (ECE) with equal-width probability bins.

    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|
    """
    y = np.clip(np.asarray(y_true, dtype=float), 0.0, 1.0)
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    if y.size == 0:
        return float("nan")
    if y.size == 1:
        return float(abs(y[0] - p[0]))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def decile_lift(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_deciles: int = 10,
) -> float:
    """
    Lift of top decile mean outcome vs overall mean.

    Higher is better when *y_score* ranks positive outcomes.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    if y.size == 0:
        return float("nan")
    if y.size == 1:
        return 0.0

    n = min(n_deciles, len(y))
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(len(s), dtype=float)
    decile = np.floor(ranks / len(s) * n).astype(int)
    decile = np.clip(decile, 0, n - 1)

    overall = float(y.mean())
    top = decile == (n - 1)
    if not top.any():
        return 0.0
    top_mean = float(y[top].mean())
    if overall == 0.0:
        return top_mean
    return float((top_mean - overall) / abs(overall))


def spearman_rank_corr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Spearman rank correlation between outcomes and scores."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    if y.size < 2:
        return float("nan")
    if np.std(y) == 0.0 or np.std(s) == 0.0:
        return float("nan")
    result = stats.spearmanr(y, s)
    return float(result.correlation)  # type: ignore[union-attr]


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-bin reliability table for calibration plots."""
    y = np.clip(np.asarray(y_true, dtype=float), 0.0, 1.0)
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        rows.append(
            {
                "bin": i,
                "bin_lo": lo,
                "bin_hi": hi,
                "count": count,
                "mean_predicted": float(p[mask].mean()) if count else float("nan"),
                "mean_observed": float(y[mask].mean()) if count else float("nan"),
            }
        )
    return pd.DataFrame(rows)
