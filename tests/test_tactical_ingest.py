"""Tactical ingest tier tests (mocked offline)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from nba_fit.config.settings import (
    INGEST_TIER_TACTICAL,
    OPTION_D_TACTICAL_ENDPOINTS,
    Settings,
)
from nba_fit.data.client import FetchResult
from nba_fit.data.fetchers.tactical import (
    fetch_option_d_tactical,
    primary_dataset_name,
    tactical_kwargs,
)
from nba_fit.data.ingest import resolve_endpoints, resolve_impact_max_games, run_ingest


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


def _mock_tactical_result(endpoint: str, season: str) -> FetchResult:
    datasets = {
        "leaguehustlestatsplayer": "HustleStatsPlayer",
        "leaguehustlestatsteam": "HustleStatsTeam",
        "leaguedashptdefend": "LeagueDashPTDefend",
        "gravityleaders": "leaders",
    }
    ds_name = datasets[endpoint]
    return FetchResult(
        endpoint=endpoint,
        season=season,
        frames={ds_name: pd.DataFrame({"PLAYER_ID": [1], "TEAM_ID": [1610612747]})},
    )


def test_tactical_primary_dataset_names():
    assert primary_dataset_name("leaguehustlestatsplayer") == "HustleStatsPlayer"
    assert primary_dataset_name("gravityleaders") == "leaders"


def test_tactical_kwargs_full_league():
    assert tactical_kwargs("leaguehustlestatsplayer") == {"team_id_nullable": ""}
    assert tactical_kwargs("gravityleaders") == {}


def test_resolve_endpoints_tactical():
    assert resolve_endpoints(INGEST_TIER_TACTICAL) == OPTION_D_TACTICAL_ENDPOINTS


def test_resolve_impact_max_games():
    from nba_fit.config.settings import INGEST_IMPACT_MAX_GAMES_DEV

    assert resolve_impact_max_games(max_games=None, full_season=False) == INGEST_IMPACT_MAX_GAMES_DEV
    assert resolve_impact_max_games(max_games=None, full_season=True) is None
    assert resolve_impact_max_games(max_games=0, full_season=False) is None
    assert resolve_impact_max_games(max_games=10, full_season=False) == 10


@patch("nba_fit.data.fetchers.tactical.fetch_tactical")
def test_fetch_option_d_tactical(mock_fetch):
    season = "2025-26"
    mock_fetch.side_effect = lambda client, endpoint, **kw: _mock_tactical_result(endpoint, season)  # noqa: ARG005

    results = fetch_option_d_tactical(None, season=season, use_cache=True)
    assert set(results) == set(OPTION_D_TACTICAL_ENDPOINTS)
    assert mock_fetch.call_count == len(OPTION_D_TACTICAL_ENDPOINTS)


@patch("nba_fit.data.ingest.fetch_option_d_tactical")
def test_run_ingest_tactical_tier(mock_fetch, tmp_path):
    settings = _fake_settings(tmp_path)
    season = settings.default_season

    mock_fetch.return_value = {
        ep: _mock_tactical_result(ep, season) for ep in OPTION_D_TACTICAL_ENDPOINTS
    }

    with (
        patch("nba_fit.data.ingest.get_settings", return_value=settings),
        patch("nba_fit.data.storage.get_settings", return_value=settings),
        patch("nba_fit.data.ingest.NBAClient"),
    ):
        result = run_ingest(season=season, tier=INGEST_TIER_TACTICAL, use_cache=True)

    assert result.tier == INGEST_TIER_TACTICAL
    assert set(result.fetched) == set(OPTION_D_TACTICAL_ENDPOINTS)
    assert mock_fetch.call_count == 1
