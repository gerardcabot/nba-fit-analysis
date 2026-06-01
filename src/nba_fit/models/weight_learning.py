"""Learn nonnegative ensemble weights that sum to one from labeled outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nba_fit.models.constants import WEIGHT_LEARNING_DEFAULT_L2
from nba_fit.scoring.constants import ENSEMBLE_COMPONENT_NAMES, ENSEMBLE_COMPONENT_WEIGHTS


@dataclass(frozen=True)
class WeightLearningResult:
    """Fitted ensemble weights and training diagnostics."""

    weights: dict[str, float]
    prior_weights: dict[str, float]
    l2: float
    train_mse: float
    n_samples: int


def _normalize_weights(raw: np.ndarray) -> np.ndarray:
    raw = np.clip(raw, 0.0, None)
    total = raw.sum()
    if total <= 0:
        return np.ones_like(raw) / len(raw)
    return raw / total


def learn_ensemble_weights(
    X: np.ndarray,
    y: np.ndarray,
    prior_weights: dict[str, float] | None = None,
    *,
    l2: float = WEIGHT_LEARNING_DEFAULT_L2,
    n_iter: int = 500,
    learning_rate: float = 0.05,
) -> WeightLearningResult:
    """
    Projected gradient descent for nonnegative weights summing to one.

    Minimizes ``||X @ w - y||^2 + l2 * ||w - w_prior||^2`` subject to
    ``w >= 0`` and ``sum(w) == 1``.

    Parameters
    ----------
    X
        Shape ``(n_samples, n_components)`` — model component predictions.
    y
        Shape ``(n_samples,)`` — target outcomes in [0, 1] or percentile scale.
    prior_weights
        Named prior aligned to ``ENSEMBLE_COMPONENT_NAMES`` order; defaults to
        ``ENSEMBLE_COMPONENT_WEIGHTS``.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if X_arr.ndim != 2:
        raise ValueError("X must be 2-dimensional (n_samples, n_components)")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have matching n_samples")
    if X_arr.shape[0] == 0:
        raise ValueError("X must have at least one sample")

    n_components = X_arr.shape[1]
    names = list(ENSEMBLE_COMPONENT_NAMES[:n_components])
    if len(names) < n_components:
        names.extend(f"component_{i}" for i in range(len(names), n_components))

    prior = prior_weights or ENSEMBLE_COMPONENT_WEIGHTS
    w_prior = np.array([float(prior.get(n, 1.0 / n_components)) for n in names], dtype=float)
    w_prior = _normalize_weights(w_prior)

    w = w_prior.copy()
    for _ in range(n_iter):
        pred = X_arr @ w
        grad = 2.0 * X_arr.T @ (pred - y_arr) / len(y_arr) + 2.0 * l2 * (w - w_prior)
        w = _normalize_weights(w - learning_rate * grad)

    pred_final = X_arr @ w
    mse = float(np.mean((pred_final - y_arr) ** 2))
    weight_dict = {names[i]: float(w[i]) for i in range(n_components)}

    return WeightLearningResult(
        weights=weight_dict,
        prior_weights={names[i]: float(w_prior[i]) for i in range(n_components)},
        l2=l2,
        train_mse=mse,
        n_samples=int(len(y_arr)),
    )
