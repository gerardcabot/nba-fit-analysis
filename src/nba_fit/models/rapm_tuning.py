"""Blocked time-series CV for RAPM ridge alpha and recency half-life."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nba_fit.models.constants import RAPM_RIDGE_ALPHA
from nba_fit.models.rapm import (
    build_player_dummy_matrix,
    fit_ridge_impact,
    recency_weights,
)

BIBLIOGRAPHY_RIDGE_ANCHOR = (
    "Deep RAPM / luck-adjusted RAPM literature — ridge L2 in 500–5000+ range "
    "([BIBLIOGRAPHY.md#deep-rapm--luck-adjusted-rapm-literature](../../docs/BIBLIOGRAPHY.md))"
)
BIBLIOGRAPHY_RECENCY_ANCHOR = (
    "DARKO recency weighting — exponential half-life on stint exposure "
    "([BIBLIOGRAPHY.md#darko-kostya-medvedovsky](../../docs/BIBLIOGRAPHY.md))"
)

DEFAULT_ALPHA_GRID: tuple[float, ...] = (500.0, 1000.0, 2500.0, 5000.0)
DEFAULT_HALF_LIFE_GRID: tuple[float, ...] = (10.0, 15.0, 20.0)


@dataclass(frozen=True)
class RapmTuneResult:
    """Selected hyperparameters and CV diagnostics."""

    ridge_alpha: float
    recency_half_life_games: float
    cv_mse: float
    n_folds: int
    n_stints: int
    alpha_grid: tuple[float, ...]
    half_life_grid: tuple[float, ...]
    bibliography_ridge: str = BIBLIOGRAPHY_RIDGE_ANCHOR
    bibliography_recency: str = BIBLIOGRAPHY_RECENCY_ANCHOR

    def metadata_fields(self) -> dict[str, Any]:
        return {
            "ridge_alpha": self.ridge_alpha,
            "recency_half_life_games": self.recency_half_life_games,
            "rapm_tune_cv_mse": self.cv_mse,
            "rapm_tune_n_folds": self.n_folds,
            "rapm_tune_n_stints": self.n_stints,
            "rapm_tune_alpha_grid": list(self.alpha_grid),
            "rapm_tune_half_life_grid": list(self.half_life_grid),
            "bibliography_ridge_alpha": self.bibliography_ridge,
            "bibliography_recency_half_life": self.bibliography_recency,
            "rapm_tune_method": "blocked_time_series_cv",
        }


def _stint_order_key(stints: pd.DataFrame) -> np.ndarray:
    """Sort key for blocked CV: min game_id when present, else row order."""
    if "min_game_id" in stints.columns:
        return stints["min_game_id"].astype(str).to_numpy()
    if "game_order" in stints.columns:
        return stints["game_order"].to_numpy(dtype=float)
    return np.arange(len(stints), dtype=float)


def _predict_stint_targets(
    design: np.ndarray,
    coef: np.ndarray,
    intercept: float,
) -> np.ndarray:
    return design @ coef + intercept


def _blocked_fold_indices(n_stints: int, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_stints < n_folds * 2:
        n_folds = max(2, min(n_folds, n_stints // 2 or 1))
    order = np.arange(n_stints)
    fold_size = max(1, n_stints // n_folds)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        val_start = k * fold_size
        val_end = n_stints if k == n_folds - 1 else (k + 1) * fold_size
        val_idx = order[val_start:val_end]
        train_idx = np.concatenate([order[:val_start], order[val_end:]])
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))
    return folds


def tune_rapm_hyperparameters(
    stints: pd.DataFrame,
    *,
    alpha_grid: tuple[float, ...] | None = None,
    half_life_grid: tuple[float, ...] | None = None,
    n_folds: int = 5,
    min_stints: int = 40,
) -> RapmTuneResult:
    """
    Select ridge alpha and recency half-life via blocked time-series CV on stint targets.

    Fits offensive ridge on train folds and scores weighted MSE on held-out ``off_rating``
    and ``def_rating`` rows (defense uses negated coefficients like production RAPM).
    """
    alphas = alpha_grid or DEFAULT_ALPHA_GRID
    half_lives = half_life_grid or DEFAULT_HALF_LIFE_GRID

    if stints.empty or len(stints) < min_stints:
        return RapmTuneResult(
            ridge_alpha=RAPM_RIDGE_ALPHA,
            recency_half_life_games=half_lives[len(half_lives) // 2],
            cv_mse=float("nan"),
            n_folds=0,
            n_stints=int(len(stints)),
            alpha_grid=alphas,
            half_life_grid=half_lives,
        )

    ordered = stints.iloc[np.argsort(_stint_order_key(stints))].reset_index(drop=True)
    stint_players = [list(map(int, s)) for s in ordered["player_ids"]]
    off_targets = ordered["off_rating"].to_numpy(dtype=float)
    def_targets = ordered["def_rating"].to_numpy(dtype=float)
    base_weights = ordered["stint_weight"].to_numpy(dtype=float)
    design, _ = build_player_dummy_matrix(stint_players)
    n = design.shape[0]
    folds = _blocked_fold_indices(n, n_folds)

    best_mse = float("inf")
    best_alpha = RAPM_RIDGE_ALPHA
    best_hl = half_lives[len(half_lives) // 2]

    for hl in half_lives:
        recency = recency_weights(n, hl)
        for alpha in alphas:
            fold_mses: list[float] = []
            for train_idx, val_idx in folds:
                w_train = base_weights[train_idx] * recency[train_idx]
                w_val = base_weights[val_idx] * recency[val_idx]
                orapm = fit_ridge_impact(
                    design[train_idx],
                    off_targets[train_idx],
                    w_train,
                    alpha=alpha,
                )
                drapm_raw = fit_ridge_impact(
                    design[train_idx],
                    def_targets[train_idx],
                    w_train,
                    alpha=alpha,
                )
                # Recover intercept from train predictions for val scoring
                off_pred_train = design[train_idx] @ orapm
                def_pred_train = design[train_idx] @ drapm_raw
                off_intercept = float(np.average(off_targets[train_idx] - off_pred_train, weights=w_train))
                def_intercept = float(np.average(def_targets[train_idx] - def_pred_train, weights=w_train))

                off_pred_val = _predict_stint_targets(design[val_idx], orapm, off_intercept)
                def_pred_val = _predict_stint_targets(design[val_idx], drapm_raw, def_intercept)
                mse_off = float(np.average((off_targets[val_idx] - off_pred_val) ** 2, weights=w_val))
                mse_def = float(np.average((def_targets[val_idx] - def_pred_val) ** 2, weights=w_val))
                fold_mses.append(0.5 * (mse_off + mse_def))

            if not fold_mses:
                continue
            mse = float(np.mean(fold_mses))
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
                best_hl = hl

    return RapmTuneResult(
        ridge_alpha=best_alpha,
        recency_half_life_games=best_hl,
        cv_mse=best_mse,
        n_folds=len(folds),
        n_stints=n,
        alpha_grid=alphas,
        half_life_grid=half_lives,
    )
