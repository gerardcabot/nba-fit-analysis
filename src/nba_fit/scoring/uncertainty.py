"""Uncertainty bands from model disagreement and sample-size penalties."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Default ensemble members when only a scalar raw score is available
DEFAULT_MODEL_KEYS: tuple[str, ...] = (
    "ensemble_profile_fit",
    "ensemble_role_fit",
    "ensemble_team_need_fit",
    "ensemble_projected_impact",
    "ensemble_replacement_upgrade",
)


@dataclass(frozen=True)
class UncertaintyResult:
    """Point estimate with bootstrap CI and penalty-adjusted width."""

    point: float
    ci_low: float
    ci_high: float
    disagreement_std: float
    sample_penalty: float
    effective_n: float


def model_disagreement_std(predictions: np.ndarray) -> float:
    """
    Standard deviation across model components (ensemble spread).

    For matrix *predictions* with shape ``(n_models, n_obs)`` or
    ``(n_obs, n_models)``, returns per-observation std; pass 1d for scalar spread.
    """
    arr = np.asarray(predictions, dtype=float)
    if arr.ndim == 1:
        return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if arr.shape[0] == 1:
        row = arr.reshape(-1)
        return float(np.std(row, ddof=1)) if row.size > 1 else 0.0
    if arr.shape[1] == 1:
        col = arr[:, 0]
        return float(np.std(col, ddof=1)) if col.size > 1 else 0.0
    per_row = np.std(arr, axis=1, ddof=1)
    return float(np.nanmean(per_row))


def sample_size_penalty(
    minutes: float | np.ndarray,
    *,
    floor_minutes: float = 500.0,
    steepness: float = 2.0,
) -> float | np.ndarray:
    """
    Penalty in [0, 1]: 0 when minutes >> floor, →1 as minutes → 0.

    Formula::

        penalty = (1 - min(1, minutes / floor_minutes)) ** steepness

    Inspired by RAPM low-sample flags: sub-500 minute seasons inflate variance.
    """
    mins = np.asarray(minutes, dtype=float)
    ratio = np.clip(mins / floor_minutes, 0.0, 1.0)
    penalty = (1.0 - ratio) ** steepness
    if np.ndim(penalty) == 0:
        return float(penalty)
    return penalty


def effective_sample_size(
    minutes: float,
    games: float | None = None,
    *,
    minutes_weight: float = 1.0,
    games_weight: float = 0.25,
) -> float:
    """
    Heuristic effective *n* for width scaling::

        n_eff = minutes_weight * minutes + games_weight * games

    Defaults games to ``minutes / 36`` when omitted (≈1 game per 36 min).
    """
    g = games if games is not None else minutes / 36.0
    return max(minutes_weight * minutes + games_weight * g, 1.0)


def bootstrap_disagreement_ci(
    model_scores: np.ndarray,
    *,
    n_bootstrap: int = 200,
    alpha: float = 0.10,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Bootstrap CI on the ensemble mean from model disagreement.

    Procedure:
    1. Treat each column of *model_scores* (shape ``n_obs × n_models``) as one
       model draw for the same observation.
    2. Resample model indices with replacement *n_bootstrap* times.
    3. Point = mean of per-model means; CI = ``alpha/2`` and ``1-alpha/2``
       quantiles of bootstrap ensemble means.

    Returns ``(point, ci_low, ci_high)``.
    """
    arr = np.asarray(model_scores, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        return 0.0, 0.0, 0.0

    rng = rng or np.random.default_rng(42)
    n_models = arr.shape[1]
    per_model_means = np.nanmean(arr, axis=0)
    point = float(np.nanmean(per_model_means))

    if n_models < 2:
        return point, point, point

    boot_means = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_models, size=n_models)
        boot_means[b] = np.nanmean(per_model_means[idx])

    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return point, lo, hi


def widen_ci_for_sample(
    ci_low: float,
    ci_high: float,
    *,
    minutes: float,
    floor_minutes: float = 500.0,
    steepness: float = 2.0,
) -> tuple[float, float, float]:
    """
    Inflate CI half-width by ``1 + sample_size_penalty(minutes)``.

    Returns ``(ci_low, ci_high, penalty)``.
    """
    penalty = float(sample_size_penalty(minutes, floor_minutes=floor_minutes, steepness=steepness))
    point = 0.5 * (ci_low + ci_high)
    half = 0.5 * (ci_high - ci_low) * (1.0 + penalty)
    return point - half, point + half, penalty


def fit_uncertainty(
    model_scores: np.ndarray,
    *,
    minutes: float = 1000.0,
    games: float | None = None,
    n_bootstrap: int = 200,
    alpha: float = 0.10,
    floor_minutes: float = 500.0,
) -> UncertaintyResult:
    """
    Combine disagreement bootstrap CI with sample-size widening.

    Final width scales additionally by ``1 / sqrt(n_eff)`` so low-minute
    players get wider bands even after bootstrap.
    """
    point, lo, hi = bootstrap_disagreement_ci(model_scores, n_bootstrap=n_bootstrap, alpha=alpha)
    lo, hi, penalty = widen_ci_for_sample(
        lo, hi, minutes=minutes, floor_minutes=floor_minutes
    )
    n_eff = effective_sample_size(minutes, games)
    spread = float(hi - lo)
    width_scale = 1.0 + 1.0 / np.sqrt(n_eff)
    mid = float(0.5 * (lo + hi))
    half = float(0.5 * spread * width_scale)
    return UncertaintyResult(
        point=float(point),
        ci_low=float(mid - half),
        ci_high=float(mid + half),
        disagreement_std=float(model_disagreement_std(model_scores)),
        sample_penalty=float(penalty),
        effective_n=float(n_eff),
    )


def uncertainty_from_submetrics(
    row: pd.Series,
    *,
    minutes: float | None = None,
    weights: dict[str, float] | None = None,
) -> UncertaintyResult:
    """Build model matrix from fit submetric columns on a pair row."""
    from nba_fit.scoring.constants import SUBMETRIC_NAMES, SUBMETRIC_WEIGHTS

    w = weights or SUBMETRIC_WEIGHTS
    keys = [k for k in DEFAULT_MODEL_KEYS if k in row.index]
    if not keys:
        keys = [f"ensemble_{k}" for k in ("profile_fit", "role_fit", "team_need_fit", "projected_impact", "replacement_upgrade") if f"ensemble_{k}" in row.index]
    if not keys:
        keys = [k for k in SUBMETRIC_NAMES if k in row.index]
    scores = np.array([[float(row[k]) for k in keys]], dtype=float)
    mins = minutes
    if mins is None and "pre_move_minutes" in row.index:
        mins = float(row["pre_move_minutes"])
    if mins is None:
        mins = 800.0
    return fit_uncertainty(scores, minutes=mins)
