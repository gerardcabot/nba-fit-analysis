"""Synthetic tests for RAPM stint matrix and ridge off/def split."""

from __future__ import annotations

import numpy as np
import pytest

from nba_fit.config.settings import RAPM_RECENCY_HALF_LIFE_GAMES
from nba_fit.models.rapm import (
    RapmArtifacts,
    build_player_dummy_matrix,
    fit_rapm_from_stints,
    parse_lineup_player_ids,
    recency_weights,
    save_rapm,
    synthetic_stint_matrix,
)


def test_parse_lineup_player_ids_from_group_id() -> None:
    assert parse_lineup_player_ids("2544-201939-203076") == [2544, 201939, 203076]
    assert parse_lineup_player_ids("") == []
    assert parse_lineup_player_ids(None) == []


def test_player_dummy_matrix_shape() -> None:
    stints = [[1, 2, 3], [2, 3, 4], [1, 4, 5]]
    matrix, player_ids = build_player_dummy_matrix(stints)
    assert matrix.shape == (3, 5)
    assert list(player_ids) == [1, 2, 3, 4, 5]
    assert matrix[0, 0] == 1.0
    assert matrix[0, 3] == 0.0


def test_recency_weights_newest_last() -> None:
    w = recency_weights(4, half_life_games=2.0)
    assert len(w) == 4
    assert w[-1] == pytest.approx(1.0)
    assert w[0] < w[-1]


def test_recency_half_life_from_settings() -> None:
    w = recency_weights(3, RAPM_RECENCY_HALF_LIFE_GAMES)
    assert w[2] == pytest.approx(1.0)
    assert np.all(w > 0)


def test_fit_rapm_recovers_synthetic_signal() -> None:
    stint_players, off_t, def_t, weights = synthetic_stint_matrix(
        n_players=10,
        n_stints=120,
    )
    artifacts = fit_rapm_from_stints(
        stint_players,
        off_t,
        def_t,
        season="2025-26",
        stint_weights=weights,
        alpha=500.0,
    )
    assert isinstance(artifacts, RapmArtifacts)
    assert len(artifacts.player_ids) == 10
    assert artifacts.orapm.shape == (10,)
    assert artifacts.drapm.shape == (10,)
    assert np.allclose(artifacts.net_rapm, artifacts.orapm + artifacts.drapm)
    assert artifacts.stint_possessions.sum() > 0


def test_save_load_rapm_roundtrip(tmp_path) -> None:
    stint_players, off_t, def_t, weights = synthetic_stint_matrix(n_players=8, n_stints=60)
    artifacts = fit_rapm_from_stints(
        stint_players,
        off_t,
        def_t,
        season="2025-26",
        stint_weights=weights,
    )
    out = save_rapm(artifacts, tmp_path / "rapm" / "season=2025-26")
    assert (out / "player_rapm.parquet").is_file()
    from nba_fit.models.rapm import load_rapm

    loaded = load_rapm("2025-26", root=tmp_path)
    assert np.allclose(loaded.net_rapm, artifacts.net_rapm)
