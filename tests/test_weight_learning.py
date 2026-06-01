"""Tests for ensemble weight learning."""

from __future__ import annotations

import numpy as np
import pytest

from nba_fit.models.weight_learning import learn_ensemble_weights
from nba_fit.scoring.constants import ENSEMBLE_COMPONENT_WEIGHTS


def test_learned_weights_nonnegative_sum_to_one() -> None:
    rng = np.random.default_rng(42)
    n = 80
    true_w = np.array([0.3, 0.2, 0.15, 0.2, 0.15])
    X = rng.uniform(0, 1, size=(n, 5))
    y = X @ true_w + rng.normal(0, 0.02, size=n)
    y = np.clip(y, 0, 1)

    result = learn_ensemble_weights(X, y, ENSEMBLE_COMPONENT_WEIGHTS, l2=0.001)
    vals = np.array(list(result.weights.values()))
    assert vals.min() >= -1e-9
    assert vals.sum() == pytest.approx(1.0, abs=1e-6)


def test_learned_weights_improve_mse_vs_prior() -> None:
    rng = np.random.default_rng(7)
    w_true = np.array([0.05, 0.05, 0.1, 0.4, 0.4])
    X = rng.uniform(0.2, 0.8, size=(100, 5))
    y = np.clip(X @ w_true, 0, 1)

    prior = np.array(list(ENSEMBLE_COMPONENT_WEIGHTS.values()))
    mse_prior = float(np.mean((X @ prior - y) ** 2))
    result = learn_ensemble_weights(X, y, l2=0.0001)
    assert result.train_mse <= mse_prior + 0.01
