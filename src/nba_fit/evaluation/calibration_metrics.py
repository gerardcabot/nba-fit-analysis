"""Calibration and ranking metrics for fit-score validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from nba_fit.models.calibration import FitCalibrator


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


@dataclass(frozen=True)
class HoldoutCalibrationResult:
    """Isotonic calibrator fit on train-season movements, evaluated on holdout."""

    train_seasons: tuple[str, ...]
    eval_season: str
    n_train: int
    n_eval: int
    brier: float | None
    ece: float | None
    decile_lift: float | None
    spearman: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "train_seasons": list(self.train_seasons),
            "eval_season": self.eval_season,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "brier": self.brier,
            "ece": self.ece,
            "decile_lift": self.decile_lift,
            "spearman": self.spearman,
        }


def split_movements_by_season(
    movements: pd.DataFrame,
    *,
    eval_season: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition movement rows into train (season < eval) and eval (== eval) sets."""
    if movements.empty or "season" not in movements.columns:
        return movements.iloc[0:0].copy(), movements.copy()
    eval_start = int(eval_season.split("-")[0])
    seasons = movements["season"].astype(str)

    def _season_start(label: str) -> int:
        return int(str(label).split("-")[0])

    train_mask = seasons.map(_season_start) < eval_start
    eval_mask = seasons == eval_season
    return movements.loc[train_mask].copy(), movements.loc[eval_mask].copy()


def fit_isotonic_on_train_rows(
    train_rows: pd.DataFrame,
    *,
    raw_col: str = "raw_fit_score",
    outcome_col: str = "post_move_outcome",
) -> FitCalibrator | None:
    """Fit isotonic regression on train-season movement rows only."""
    if train_rows.empty or raw_col not in train_rows.columns:
        return None
    outcomes = (
        train_rows[outcome_col].to_numpy(dtype=float)
        if outcome_col in train_rows.columns
        else None
    )
    if outcomes is None:
        return None
    calibrator = FitCalibrator(method="isotonic")
    calibrator.fit(train_rows[raw_col].to_numpy(dtype=float), outcomes)
    return calibrator if calibrator.is_fitted_ else None


def evaluate_holdout_calibration(
    calibrator: FitCalibrator | None,
    eval_rows: pd.DataFrame,
    *,
    raw_col: str = "raw_fit_score",
    outcome_col: str = "post_move_outcome",
    train_seasons: tuple[str, ...] = (),
    eval_season: str = "",
    n_train: int = 0,
) -> HoldoutCalibrationResult:
    """
    Apply a train-fit isotonic calibrator to holdout rows and report metrics.

    Intended workflow: fit on pre-2024-25 movements, evaluate on 2024-25 holdout.
    """
    empty = HoldoutCalibrationResult(
        train_seasons=train_seasons,
        eval_season=eval_season,
        n_train=0,
        n_eval=len(eval_rows),
        brier=None,
        ece=None,
        decile_lift=None,
        spearman=None,
    )
    if calibrator is None or eval_rows.empty or outcome_col not in eval_rows.columns:
        return empty

    raw = eval_rows[raw_col].to_numpy(dtype=float)
    calibrated = calibrator.transform(raw)
    y = eval_rows[outcome_col].to_numpy(dtype=float)
    p = np.clip(calibrated / 100.0, 0.0, 1.0)

    return HoldoutCalibrationResult(
        train_seasons=train_seasons,
        eval_season=eval_season,
        n_train=n_train,
        n_eval=len(eval_rows),
        brier=brier_score(y, p),
        ece=expected_calibration_error(y, p),
        decile_lift=decile_lift(y, p),
        spearman=spearman_rank_corr(y, p),
    )
