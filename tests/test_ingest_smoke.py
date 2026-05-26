"""Ingest pipeline smoke tests (mocked offline)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from nba_fit.config.settings import (
    INGEST_TIER_MVP,
    OPTION_A_MVP_ENDPOINTS,
    Settings,
)
from nba_fit.data.client import FetchResult
from nba_fit.data.ingest import resolve_endpoints, run_ingest


def _fake_settings(tmp_path: Path) -> Settings:
    return Settings(
        root=tmp_path,
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


def _player_stats_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PLAYER_ID": [1629029, 2544],
            "PLAYER_NAME": ["Luka Doncic", "LeBron James"],
            "TEAM_ID": [1610612747, 1610612747],
            "TEAM_ABBREVIATION": ["LAL", "LAL"],
            "GP": [50, 48],
            "MIN": [600.0, 550.0],
            "PTS": [28.5, 24.1],
        }
    )


def _player_bio_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PLAYER_ID": [1629029, 2544],
            "PLAYER_NAME": ["Luka Doncic", "LeBron James"],
            "TEAM_ID": [1610612747, 1610612747],
            "USG_PCT": [0.36, 0.29],
            "TS_PCT": [0.58, 0.61],
        }
    )


def _player_est_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PLAYER_ID": [1629029, 2544],
            "PLAYER_NAME": ["Luka Doncic", "LeBron James"],
            "E_USG_PCT": [0.35, 0.28],
            "E_NET_RATING": [5.0, 3.0],
        }
    )


def _player_shot_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PLAYER_ID": [1629029, 2544],
            "PLAYER_NAME": ["Luka Doncic", "LeBron James"],
            "TEAM_ID": [1610612747, 1610612747],
            "Restricted Area | FGA": [5.0, 4.0],
        }
    )


def _team_stats_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TEAM_ID": [1610612747],
            "TEAM_NAME": ["Los Angeles Lakers"],
            "TEAM_ABBREVIATION": ["LAL"],
            "GP": [50],
            "PTS": [112.0],
        }
    )


def _team_est_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TEAM_ID": [1610612747],
            "TEAM_NAME": ["Los Angeles Lakers"],
            "E_NET_RATING": [2.5],
        }
    )


def _team_shot_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TEAM_ID": [1610612747],
            "TEAM_NAME": ["Los Angeles Lakers"],
            "Restricted Area | FGA": [30.0],
        }
    )


def _mock_fetch_result(endpoint: str, season: str) -> FetchResult:
    frames: dict[str, pd.DataFrame]
    if endpoint == "leaguedashplayerstats":
        frames = {"LeagueDashPlayerStats": _player_stats_df()}
    elif endpoint == "leaguedashplayerbiostats":
        frames = {"LeagueDashPlayerBioStats": _player_bio_df()}
    elif endpoint == "playerestimatedmetrics":
        frames = {"PlayerEstimatedMetrics": _player_est_df()}
    elif endpoint == "leaguedashplayershotlocations":
        frames = {"ShotLocations": _player_shot_df()}
    elif endpoint == "leaguedashteamstats":
        frames = {"LeagueDashTeamStats": _team_stats_df()}
    elif endpoint == "teamestimatedmetrics":
        frames = {"TeamEstimatedMetrics": _team_est_df()}
    elif endpoint == "leaguedashteamshotlocations":
        frames = {"ShotLocations": _team_shot_df()}
    else:
        raise AssertionError(f"unexpected endpoint {endpoint}")
    return FetchResult(endpoint=endpoint, season=season, frames=frames)


def test_resolve_endpoints_mvp():
    eps = resolve_endpoints(INGEST_TIER_MVP)
    assert eps == OPTION_A_MVP_ENDPOINTS


def test_resolve_endpoints_unknown():
    with pytest.raises(ValueError, match="Unknown ingest tier"):
        resolve_endpoints("full")


@patch("nba_fit.data.ingest.fetch_option_a_mvp")
def test_run_ingest_writes_interim_parquet(mock_fetch, tmp_path):
    settings = _fake_settings(tmp_path)
    season = settings.default_season

    def _side_effect(_client, *, season: str, use_cache: bool, endpoints):  # noqa: ARG001
        return {ep: _mock_fetch_result(ep, season) for ep in endpoints}

    mock_fetch.side_effect = _side_effect

    with (
        patch("nba_fit.data.ingest.get_settings", return_value=settings),
        patch("nba_fit.data.storage.get_settings", return_value=settings),
        patch("nba_fit.data.ingest.NBAClient"),
    ):
        result = run_ingest(season=season, tier=INGEST_TIER_MVP, use_cache=True)

    assert result.player_rows == 2
    assert result.team_rows == 1
    players_path = Path(result.players_path)
    teams_path = Path(result.teams_path)
    assert players_path.exists()
    assert teams_path.exists()

    players = pd.read_parquet(players_path)
    assert "PTS" in players.columns
    assert "USG_PCT" in players.columns
    assert players["player_id"].notna().all()

    teams = pd.read_parquet(teams_path)
    assert teams["team_id"].notna().all()
    assert mock_fetch.call_count == 1
