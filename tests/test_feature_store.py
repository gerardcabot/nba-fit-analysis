"""Feature store path and materialization tests (mocked offline)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from nba_fit.config.settings import Settings
from nba_fit.data.storage import feature_path
from nba_fit.features.store import (
    FEATURE_MATRIX_PLAYER,
    FEATURE_MATRIX_SCALING,
    FEATURE_MATRIX_TEAM,
    materialize_features,
    write_player_features_scaled,
    write_scaling_params,
    write_team_features_scaled,
)


def _fake_settings(tmp_path: Path) -> Settings:
    return Settings(
        root=tmp_path,
        data_root=None,
        data_raw=tmp_path / "data" / "raw",
        data_interim=tmp_path / "data" / "interim",
        data_features=tmp_path / "data" / "features",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        probe_results_path=tmp_path / "probe_all_results.json",
        seasons=("2025-26",),
        cache_ttl_hours=24,
        default_season="2025-26",
    )


def test_feature_path_layout():
    path = feature_path(FEATURE_MATRIX_PLAYER, "2025-26")
    assert path.name == "data.parquet"
    assert "player_features_scaled" in str(path)
    assert "season=2025-26" in str(path)


def test_write_feature_store_paths(tmp_path):
    settings = _fake_settings(tmp_path)
    season = settings.default_season
    player_df = pd.DataFrame({"PLAYER_ID": [1], "z_pf_pts": [0.5]})
    team_df = pd.DataFrame({"TEAM_ID": [1610612747], "z_tf_pts": [0.1]})
    scaling_df = pd.DataFrame(
        {"season": [season], "entity": ["player"], "feature": ["pf_pts"], "center": [0.0], "scale": [1.0]}
    )

    with patch("nba_fit.features.store.get_settings", return_value=settings), patch(
        "nba_fit.data.storage.get_settings", return_value=settings
    ):
        player_path = write_player_features_scaled(player_df, season)
        team_path = write_team_features_scaled(team_df, season)
        scaling_path = write_scaling_params(scaling_df, season)
        expected_player = feature_path(FEATURE_MATRIX_PLAYER, season)
        expected_team = feature_path(FEATURE_MATRIX_TEAM, season)
        expected_scaling = feature_path(FEATURE_MATRIX_SCALING, season)

    assert player_path == expected_player
    assert team_path == expected_team
    assert scaling_path == expected_scaling
    assert player_path.exists()
    assert team_path.exists()
    assert scaling_path.exists()


@patch("nba_fit.features.store.load_teams_table")
@patch("nba_fit.features.store.load_players_table")
def test_materialize_features_writes_parquet(mock_players, mock_teams, tmp_path):
    settings = _fake_settings(tmp_path)
    season = settings.default_season
    mock_players.return_value = pd.DataFrame(
        {
            "PLAYER_ID": [1629029],
            "TEAM_ID": [1610612747],
            "GP": [50],
            "MIN": [600.0],
            "PTS": [28.5],
            "USG_PCT": [0.36],
            "TS_PCT": [0.58],
        }
    )
    mock_teams.return_value = pd.DataFrame(
        {
            "TEAM_ID": [1610612747],
            "TEAM_NAME": ["Los Angeles Lakers"],
            "GP": [50],
            "PTS": [112.0],
        }
    )

    with patch("nba_fit.features.store.get_settings", return_value=settings), patch(
        "nba_fit.data.storage.get_settings", return_value=settings
    ):
        result = materialize_features(season)

    assert result.player_rows == 1
    assert result.team_rows == 1
    assert result.scaling_rows > 0
    assert result.player_path.exists()
    assert result.team_path.exists()
    assert result.scaling_path.exists()

    player_scaled = pd.read_parquet(result.player_path)
    assert any(c.startswith("z_pf_") for c in player_scaled.columns)


def test_materialize_features_missing_interim(tmp_path):
    settings = _fake_settings(tmp_path)
    with patch("nba_fit.features.store.get_settings", return_value=settings), patch(
        "nba_fit.data.storage.get_settings", return_value=settings
    ):
        with pytest.raises(FileNotFoundError):
            materialize_features(settings.default_season)
