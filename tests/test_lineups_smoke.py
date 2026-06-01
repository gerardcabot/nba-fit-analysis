"""Option B lineup/on-off ingest smoke tests (mocked offline)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from nba_fit.config.settings import (
    INGEST_TIER_ROLE,
    ONOFF_COL_COURT_STATUS,
    ONOFF_COL_VS_PLAYER_ID,
    ONOFF_COL_VS_PLAYER_NAME,
    ONOFF_STAT_MIN,
    OPTION_B_ROLE_ENDPOINTS,
    Settings,
)
from nba_fit.data.client import FetchResult
from nba_fit.data.ingest import resolve_endpoints, run_ingest


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


def _lineup_df(*, team_id: int = 1610612747) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GROUP_SET": ["5 Man"],
            "GROUP_ID": [1],
            "GROUP_NAME": ["Player A - Player B - Player C - Player D - Player E"],
            "TEAM_ID": [team_id],
            "TEAM_ABBREVIATION": ["LAL"],
            "MIN": [120.5],
            "PLUS_MINUS": [15.0],
        }
    )


def _onoff_df(*, court_status: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GROUP_SET": ["Overall"],
            "TEAM_ID": [1610612747],
            "TEAM_ABBREVIATION": ["LAL"],
            "VS_PLAYER_ID": [1629029],
            "VS_PLAYER_NAME": ["Luka Doncic"],
            ONOFF_COL_COURT_STATUS: [court_status],
            "GP": [40],
            ONOFF_STAT_MIN: [500.0 if court_status == "On" else 200.0],
            "PLUS_MINUS": [8.0 if court_status == "On" else -2.0],
            "NET_RATING": [5.5 if court_status == "On" else -1.0],
        }
    )


def _mock_fetch_result(endpoint: str, season: str) -> FetchResult:
    ep = endpoint.lower()
    if ep == "leaguedashlineups":
        frames = {"Lineups": _lineup_df()}
    elif ep == "teamdashlineups":
        frames = {"Lineups": _lineup_df(), "Overall": _lineup_df()}
    elif ep == "teamplayeronoffsummary":
        frames = {
            "PlayersOnCourtTeamPlayerOnOffSummary": _onoff_df(court_status="On"),
            "PlayersOffCourtTeamPlayerOnOffSummary": _onoff_df(court_status="Off"),
            "OverallTeamPlayerOnOffSummary": _onoff_df(court_status="On"),
        }
    elif ep == "teamplayeronoffdetails":
        frames = {
            "PlayersOnCourtTeamPlayerOnOffDetails": _onoff_df(court_status="On"),
            "PlayersOffCourtTeamPlayerOnOffDetails": _onoff_df(court_status="Off"),
        }
    else:
        raise AssertionError(f"unexpected endpoint {endpoint}")
    return FetchResult(endpoint=endpoint, season=season, frames=frames)


def test_resolve_endpoints_role():
    eps = resolve_endpoints(INGEST_TIER_ROLE)
    assert eps == OPTION_B_ROLE_ENDPOINTS


@patch("nba_fit.data.ingest.fetch_option_b_role")
def test_run_ingest_role_writes_interim_parquet(mock_fetch, tmp_path):
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
        result = run_ingest(season=season, tier=INGEST_TIER_ROLE, use_cache=True)

    assert result.lineup_units_rows >= 2
    assert result.onoff_rows >= 2
    lineup_path = Path(result.lineup_units_path)
    onoff_path = Path(result.onoff_path)
    assert lineup_path.exists()
    assert onoff_path.exists()

    lineups = pd.read_parquet(lineup_path)
    assert "lineup_group_name" in lineups.columns
    assert lineups["team_id"].notna().all()

    onoff = pd.read_parquet(onoff_path)
    assert ONOFF_STAT_MIN in onoff.columns
    assert onoff["focal_player_id"].notna().all()
    assert set(onoff[ONOFF_COL_COURT_STATUS].dropna().unique()) >= {"On", "Off"}
    assert mock_fetch.call_count == 1
