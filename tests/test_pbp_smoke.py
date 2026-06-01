"""Option C play-by-play / possession ingest smoke tests (mocked offline)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
from nba_fit.config.settings import (
    INGEST_TIER_IMPACT,
    OPTION_C_IMPACT_ENDPOINTS,
    PBP_COL_ACTION_NUMBER,
    PBP_COL_ACTION_TYPE,
    PBP_COL_CLOCK,
    PBP_COL_PERIOD,
    PBP_COL_SHOT_RESULT,
    PBP_COL_TEAM_ID,
    PBP_COL_IS_FIELD_GOAL,
    PBP_PLAYBYPLAY_DATASET,
    ROTATION_COL_IN_TIME,
    ROTATION_COL_OUT_TIME,
    ROTATION_COL_PERSON_ID,
    ROTATION_COL_TEAM_ID,
    Settings,
)
from nba_fit.data.client import FetchResult
from nba_fit.data.ingest import resolve_endpoints, run_ingest
from nba_fit.data.pbpstats_adapter import (
    build_possession_rows_for_game,
    possessions_from_events_and_rotation,
)
from nba_fit.normalize.possessions import interim_possessions_game_path, load_possessions_table


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


def _pbp_events(game_id: str = "0022501198") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gameId": [game_id] * 4,
            PBP_COL_ACTION_NUMBER: [1, 2, 3, 4],
            PBP_COL_PERIOD: [1, 1, 1, 1],
            PBP_COL_CLOCK: ["PT12M00.00S", "PT11M50.00S", "PT11M40.00S", "PT11M30.00S"],
            PBP_COL_TEAM_ID: [1610612747, 1610612747, 1610612747, 1610612747],
            PBP_COL_ACTION_TYPE: ["period", "Made Shot", "Rebound", "Turnover"],
            PBP_COL_IS_FIELD_GOAL: [0, 1, 0, 0],
            PBP_COL_SHOT_RESULT: ["", "Made", "", ""],
        }
    )


def _rotation_stints(game_id: str = "0022501198") -> pd.DataFrame:
    players = [2544, 1629029, 1628366, 1629216, 1630559]
    rows = []
    for pid in players:
        rows.append(
            {
                "GAME_ID": game_id,
                ROTATION_COL_TEAM_ID: 1610612747,
                ROTATION_COL_PERSON_ID: pid,
                ROTATION_COL_IN_TIME: 0.0,
                ROTATION_COL_OUT_TIME: 28800.0,
            }
        )
    return pd.DataFrame(rows)


def test_resolve_endpoints_impact():
    eps = resolve_endpoints(INGEST_TIER_IMPACT)
    assert eps == OPTION_C_IMPACT_ENDPOINTS


def test_possessions_from_events_and_rotation():
    season = "2025-26"
    game_id = "0022501198"
    df = possessions_from_events_and_rotation(
        _pbp_events(game_id),
        _rotation_stints(game_id),
        game_id=game_id,
        season=season,
    )
    assert len(df) >= 2
    assert df["has_lineup_on_floor"].any()
    assert (df["source"] == "events_rotation").all()


def test_build_possession_rows_for_game_fallback():
    season = "2025-26"
    game_id = "0022501198"
    pbp_result = FetchResult(
        endpoint="playbyplayv3",
        season=season,
        frames={PBP_PLAYBYPLAY_DATASET: _pbp_events(game_id)},
    )
    rot_result = FetchResult(
        endpoint="gamerotation",
        season=season,
        frames={"HomeTeam": _rotation_stints(game_id)},
    )
    settings = _fake_settings(Path("/tmp/unused"))
    with patch(
        "nba_fit.data.pbpstats_adapter.possessions_from_pbpstats_game",
        return_value=None,
    ):
        df = build_possession_rows_for_game(
            game_id=game_id,
            season=season,
            pbp_result=pbp_result,
            rotation_result=rot_result,
            pbpstats_cache_dir=settings.data_raw / "pbpstats",
        )
    assert not df.empty
    assert "lineup_id" in df.columns


@patch("nba_fit.data.ingest.fetch_option_c_impact")
def test_run_ingest_impact_writes_partitioned_parquet(mock_fetch, tmp_path):
    settings = _fake_settings(tmp_path)
    season = settings.default_season
    game_id = "0022501198"

    finder = FetchResult(
        endpoint="leaguegamefinder",
        season=season,
        frames={
            "LeagueGameFinderResults": pd.DataFrame(
                {"GAME_ID": [game_id], "GAME_DATE": ["2025-10-21"]}
            )
        },
    )
    pbp = FetchResult(
        endpoint="playbyplayv3",
        season=season,
        frames={PBP_PLAYBYPLAY_DATASET: _pbp_events(game_id)},
    )
    rot = FetchResult(
        endpoint="gamerotation",
        season=season,
        frames={"HomeTeam": _rotation_stints(game_id)},
    )

    mock_fetch.return_value = {
        "leaguegamefinder": finder,
        "game_ids": [game_id],
        "playbyplayv3": {game_id: pbp},
        "gamerotation": {game_id: rot},
    }

    with (
        patch("nba_fit.data.ingest.get_settings", return_value=settings),
        patch("nba_fit.data.storage.get_settings", return_value=settings),
        patch("nba_fit.normalize.possessions.get_settings", return_value=settings),
        patch("nba_fit.data.ingest.NBAClient"),
        patch(
            "nba_fit.data.pbpstats_adapter.possessions_from_pbpstats_game",
            return_value=None,
        ),
    ):
        result = run_ingest(
            season=season,
            tier=INGEST_TIER_IMPACT,
            use_cache=True,
            max_games=1,
        )
        assert result.possessions_rows > 0
        assert result.games_ingested == 1
        path = interim_possessions_game_path(season, game_id)
        assert path.exists()
        loaded = load_possessions_table(season, game_id=game_id)
        assert len(loaded) == result.possessions_rows

    assert mock_fetch.call_count == 1
