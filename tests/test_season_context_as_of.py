"""SeasonFitContext temporal as_of tests."""

from datetime import date

import pandas as pd

from nba_fit.evaluation.movement_backtest import freeze_pre_move_features
from nba_fit.features._synthetic import synthetic_player_df, synthetic_team_df
from nba_fit.features.season_context import (
    SeasonFitContext,
    aggregate_gamelogs_to_player_stats,
    filter_gamelogs_as_of,
)
from nba_fit.data.fetchers.transactions import infer_movements_from_gamelogs


def test_as_of_sets_cutoff_date():
    ctx = SeasonFitContext.from_synthetic("2025-26", n_players=5)
    move = date(2026, 2, 6)
    scoped = ctx.as_of(move)
    assert scoped.as_of_date == move
    assert scoped.season == ctx.season
    assert scoped.as_of(None).as_of_date is None


def test_filter_gamelogs_as_of():
    logs = pd.DataFrame(
        {
            "GAME_DATE": ["2026-01-01", "2026-02-10", "2026-03-01"],
            "PTS": [20, 25, 30],
        }
    )
    filtered = filter_gamelogs_as_of(logs, date(2026, 2, 6))
    assert len(filtered) == 1
    assert filtered.iloc[0]["PTS"] == 20


def test_aggregate_gamelogs_excludes_post_move_window():
    logs = pd.DataFrame(
        {
            "PLAYER_ID": [99] * 5 + [99] * 3,
            "TEAM_ID": [10] * 5 + [20] * 3,
            "GAME_DATE": [
                "2024-10-01",
                "2024-10-15",
                "2024-11-01",
                "2024-11-15",
                "2024-12-01",
                "2024-12-15",
                "2025-01-01",
                "2025-01-15",
            ],
            "MIN": [30.0] * 8,
            "PTS": [20.0] * 8,
        }
    )
    move = date(2024, 12, 10)
    pre_only = filter_gamelogs_as_of(logs, move)
    stats = aggregate_gamelogs_to_player_stats(pre_only)
    assert int(stats.iloc[0]["GP"]) == 5
    assert int(stats.iloc[0]["TEAM_ID"]) == 10


def test_as_of_rebuilds_vectors_when_gamelogs_attached():
    players_raw = synthetic_player_df(n_rows=1)
    players_raw.loc[0, "PLAYER_ID"] = 99
    players_raw.loc[0, "GP"] = 82
    teams_raw = synthetic_team_df([10, 20])
    logs = pd.DataFrame(
        {
            "PLAYER_ID": [99] * 8,
            "TEAM_ID": [10] * 5 + [20] * 3,
            "GAME_DATE": [
                "2024-10-01",
                "2024-10-15",
                "2024-11-01",
                "2024-11-15",
                "2024-12-01",
                "2024-12-15",
                "2025-01-01",
                "2025-01-15",
            ],
            "MIN": [32.0] * 8,
            "PTS": [18.0] * 8,
            "USG_PCT": [0.25] * 8,
            "TS_PCT": [0.55] * 8,
            "AST": [5.0] * 8,
            "TOV": [2.0] * 8,
            "FGA": [12.0] * 8,
            "FG3A": [4.0] * 8,
        }
    )
    ctx = SeasonFitContext._from_frames(
        players_raw, teams_raw, "2024-25", source="test"
    ).with_gamelogs(logs)
    scoped = ctx.as_of(date(2024, 12, 10))
    assert scoped.as_of_date == date(2024, 12, 10)
    assert scoped.source.endswith("_as_of")
    assert 99 in scoped.players


def test_freeze_pre_move_uses_temporal_gp():
    players_raw = synthetic_player_df(n_rows=1)
    players_raw.loc[0, "PLAYER_ID"] = 1
    players_raw.loc[0, "GP"] = 82
    teams_raw = synthetic_team_df([10, 20])
    logs = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1, 1, 1, 1],
            "TEAM_ID": [10, 10, 10, 20, 20, 20],
            "GAME_DATE": [
                "2024-10-01",
                "2024-10-20",
                "2024-11-01",
                "2024-11-15",
                "2024-12-01",
                "2024-12-20",
            ],
            "MIN": [30.0] * 6,
            "PTS": [15.0] * 6,
        }
    )
    ctx = SeasonFitContext._from_frames(
        players_raw, teams_raw, "2024-25", source="test"
    ).with_gamelogs(logs)
    movements = infer_movements_from_gamelogs(logs, season="2024-25")
    frozen = freeze_pre_move_features(ctx, movements)
    assert frozen.iloc[0]["pre_move_gp"] == 3
