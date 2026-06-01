"""Tests for blocked CV RAPM hyperparameter tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_fit.models.rapm import synthetic_stint_matrix
from nba_fit.models.rapm_tuning import tune_rapm_hyperparameters


def test_tune_rapm_hyperparameters_selects_from_grid() -> None:
    stint_players, off_t, def_t, weights = synthetic_stint_matrix(
        n_players=12,
        n_stints=100,
    )
    stints = pd.DataFrame(
        {
            "player_ids": stint_players,
            "off_rating": off_t,
            "def_rating": def_t,
            "stint_weight": weights,
            "min_game_id": [f"g{i:04d}" for i in range(len(stint_players))],
        }
    )
    result = tune_rapm_hyperparameters(
        stints,
        alpha_grid=(100.0, 500.0),
        half_life_grid=(5.0, 15.0),
        n_folds=4,
        min_stints=20,
    )
    assert result.ridge_alpha in (100.0, 500.0)
    assert result.recency_half_life_games in (5.0, 15.0)
    assert result.n_folds >= 1
    assert "bibliography_ridge_alpha" in result.metadata_fields()
