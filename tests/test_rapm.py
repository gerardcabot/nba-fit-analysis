"""Synthetic tests for RAPM stint matrix and ridge off/def split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_fit.config.settings import RAPM_RECENCY_HALF_LIFE_GAMES
from nba_fit.models.rapm import (
    RapmArtifacts,
    build_player_dummy_matrix,
    check_degenerate_rapm,
    fit_rapm_from_possessions,
    fit_rapm_from_stints,
    parse_lineup_player_ids,
    recency_weights,
    save_rapm,
    stints_from_possessions,
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


def test_stints_from_possessions_with_scoring() -> None:
    possessions = pd.DataFrame(
        {
            "lineup_id": ["1-2-3", "1-2-3", "4-5-6", "4-5-6"],
            "offense_team_id": [10, 10, 20, 20],
            "points_scored": [2, 0, 3, 1],
            "opponent_points_scored": [0, 2, 1, 0],
        }
    )
    stints = stints_from_possessions(possessions)
    assert len(stints) == 2
    assert stints.iloc[0]["off_rating"] == pytest.approx(100.0)


def test_fit_rapm_from_possessions_non_degenerate() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for lineup, team in [("1-2-3", 10), ("2-3-4", 10), ("5-6-7", 20)]:
        for _ in range(30):
            rows.append(
                {
                    "lineup_id": lineup,
                    "offense_team_id": team,
                    "points_scored": float(rng.integers(0, 3)),
                    "opponent_points_scored": float(rng.integers(0, 3)),
                }
            )
    possessions = pd.DataFrame(rows)
    artifacts = fit_rapm_from_possessions(possessions, season="2025-26", alpha=100.0)
    assert artifacts.rapm_source == "possessions"
    assert len(artifacts.player_ids) >= 4
    assert np.std(artifacts.net_rapm) > 0
    assert not artifacts.degenerate


def test_check_degenerate_rapm_flags_constant_net() -> None:
    stint_players, off_t, def_t, weights = synthetic_stint_matrix(n_players=6, n_stints=40)
    artifacts = fit_rapm_from_stints(
        stint_players,
        off_t,
        def_t,
        season="2025-26",
        stint_weights=weights,
    )
    flat = RapmArtifacts(
        season=artifacts.season,
        player_ids=artifacts.player_ids,
        orapm=np.zeros_like(artifacts.orapm),
        drapm=np.zeros_like(artifacts.drapm),
        net_rapm=np.zeros_like(artifacts.net_rapm),
        stint_possessions=artifacts.stint_possessions,
        low_sample_flag=artifacts.low_sample_flag,
        ridge_alpha=artifacts.ridge_alpha,
        recency_half_life_games=artifacts.recency_half_life_games,
    )
    checked = check_degenerate_rapm(flat)
    assert checked.degenerate
    assert checked.metadata["net_rapm_std"] == 0.0

