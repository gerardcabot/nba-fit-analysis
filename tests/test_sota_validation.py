"""SOTA validation gates for degenerate RAPM."""

from __future__ import annotations

import numpy as np
import pytest

from nba_fit.evaluation.sota_validation import SotaValidationError, assert_non_degenerate_rapm
from nba_fit.models.rapm import RapmArtifacts, check_degenerate_rapm


def test_assert_non_degenerate_rapm_raises_on_flat() -> None:
    flat = check_degenerate_rapm(
        RapmArtifacts(
            season="2025-26",
            player_ids=np.array([1, 2, 3]),
            orapm=np.zeros(3),
            drapm=np.zeros(3),
            net_rapm=np.zeros(3),
            stint_possessions=np.ones(3),
            low_sample_flag=np.zeros(3, dtype=bool),
            ridge_alpha=2500.0,
            recency_half_life_games=15.0,
        )
    )
    with pytest.raises(SotaValidationError, match="Degenerate"):
        assert_non_degenerate_rapm(flat)


def test_assert_non_degenerate_rapm_passes_with_spread() -> None:
    ok = RapmArtifacts(
        season="2025-26",
        player_ids=np.array([1, 2]),
        orapm=np.array([1.0, -1.0]),
        drapm=np.array([0.0, 0.0]),
        net_rapm=np.array([1.0, -1.0]),
        stint_possessions=np.array([10.0, 10.0]),
        low_sample_flag=np.array([False, False]),
        ridge_alpha=2500.0,
        recency_half_life_games=15.0,
        metadata={"net_rapm_std": 1.0},
    )
    assert_non_degenerate_rapm(ok)
