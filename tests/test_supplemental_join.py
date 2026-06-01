"""Tests for tactical / on-off / lineup supplemental joins."""

from __future__ import annotations

import pandas as pd

from nba_fit.features.constants import COL_PLAYER_ID
from nba_fit.features.player_vector import (
    FEATURE_GROUP_ONOFF,
    FEATURE_GROUP_TACTICAL,
    build_player_features,
)
from nba_fit.features.supplemental_join import (
    join_onoff_into_players,
    join_tactical_into_players,
)
from nba_fit.features.team_vector import build_team_features


def _base_players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_PLAYER_ID: [2544],
            "MIN": [35.0],
            "GP": [70],
            "FGA": [18.0],
            "FG3A": [5.0],
            "PTS": [25.0],
            "AST": [8.0],
            "TOV": [3.0],
            "USG_PCT": [0.30],
            "PLAYER_HEIGHT_INCHES": [81.0],
        }
    )


def test_tactical_join_adds_columns_when_cache_present(monkeypatch) -> None:
    hustle = pd.DataFrame(
        {
            "PLAYER_ID": [2544],
            "DEFLECTIONS": [42.0],
            "CHARGES_DRAWN": [3.0],
        }
    )

    def fake_read(_season: str, endpoint: str) -> pd.DataFrame | None:
        if endpoint == "leaguehustlestatsplayer":
            return hustle
        return None

    monkeypatch.setattr(
        "nba_fit.features.supplemental_join._read_raw_endpoint",
        fake_read,
    )
    enriched = join_tactical_into_players(_base_players(), "2024-25")
    assert "DEFLECTIONS" in enriched.columns
    feats = build_player_features(enriched)
    assert any(c.startswith(f"pf_{FEATURE_GROUP_TACTICAL}__") for c in feats.columns)


def test_onoff_join_adds_delta(monkeypatch) -> None:
    onoff = pd.DataFrame(
        {
            "focal_player_id": [2544, 2544],
            "COURT_STATUS": ["On", "Off"],
            "NET_RATING": [5.0, -2.0],
            "MIN": [500.0, 480.0],
            "onoff_dataset": [
                "PlayersOnCourtTeamPlayerOnOffSummary",
                "PlayersOffCourtTeamPlayerOnOffSummary",
            ],
        }
    )

    monkeypatch.setattr(
        "nba_fit.features.supplemental_join.load_onoff_table",
        lambda _s: onoff,
    )
    enriched = join_onoff_into_players(_base_players(), "2024-25")
    assert "ONOFF_NET_DELTA" in enriched.columns
    feats = build_player_features(enriched)
    assert any(c.startswith(f"pf_{FEATURE_GROUP_ONOFF}__") for c in feats.columns)


def test_team_lineup_weakness_columns(monkeypatch) -> None:
    lineups = pd.DataFrame(
        {
            "TEAM_ID": [1610612747, 1610612747],
            "NET_RATING": [-8.0, 4.0],
            "MIN": [120.0, 400.0],
        }
    )
    monkeypatch.setattr(
        "nba_fit.features.supplemental_join.load_lineup_units_table",
        lambda _s: lineups,
    )
    teams = pd.DataFrame({"TEAM_ID": [1610612747], "PACE": [100.0], "OFF_RATING": [115.0]})
    from nba_fit.features.supplemental_join import join_lineup_into_teams

    enriched = join_lineup_into_teams(teams, "2024-25")
    feats = build_team_features(enriched)
    assert "tf_weaknesses__lineup_worst_net" in feats.columns
