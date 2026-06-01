"""Synthetic leaguedash-shaped tables for offline scoring tests."""

from __future__ import annotations

import pandas as pd

from nba_fit.features.constants import COL_PLAYER_ID, COL_TEAM_ID


def synthetic_player_df(*, n_rows: int = 2) -> pd.DataFrame:
    """Minimal player table matching leaguedashplayerstats column stubs."""
    base = pd.DataFrame(
      {
          COL_PLAYER_ID: [1, 2],
          COL_TEAM_ID: [10, 10],
          "PLAYER_POSITION": ["Guard", "Center"],
          "MIN": [34.0, 12.0],
          "GP": [60, 30],
          "FGA": [18.0, 8.0],
          "FG3A": [7.0, 0.5],
          "PTS": [28.0, 12.0],
          "AST": [8.0, 1.0],
          "TOV": [3.5, 1.0],
          "USG_PCT": [0.32, 0.16],
          "PACE": [101.0, 98.0],
          "TS_PCT": [0.61, 0.55],
          "EFG_PCT": [0.56, 0.52],
          "OFF_RATING": [118.0, 105.0],
          "DEF_RATING": [110.0, 108.0],
          "NET_RATING": [8.0, -3.0],
          "AST_PCT": [0.35, 0.05],
          "AST_RATIO": [2.1, 0.8],
          "OREB_PCT": [0.02, 0.12],
          "DREB_PCT": [0.10, 0.28],
          "REB_PCT": [0.06, 0.20],
          "STL": [1.2, 0.4],
          "BLK": [0.2, 1.5],
          "OREB": [0.5, 2.5],
          "DREB": [4.0, 6.0],
          "REB": [4.5, 8.5],
          "FREQ": [0.75, 0.40],
          "PCT_PLUSMINUS": [-0.02, 0.01],
          "PLAYER_HEIGHT_INCHES": [75.0, 84.0],
          "PLAYER_WEIGHT": [200.0, 250.0],
          "AGE": [26.0, 30.0],
          "Restricted Area | FGA": [4.0, 3.0],
          "In The Paint (Non-RA) | FGA": [3.0, 2.0],
          "Mid-Range | FGA": [2.0, 1.0],
          "Left Corner 3 | FGA": [1.0, 0.0],
          "Right Corner 3 | FGA": [1.0, 0.0],
          "Above the Break 3 | FGA": [7.0, 2.0],
      }
    )
    if n_rows <= len(base):
        return base.iloc[:n_rows].copy()
    rows = [base.iloc[i % len(base)].copy() for i in range(n_rows)]
    return pd.DataFrame(rows).reset_index(drop=True)


def synthetic_team_df(team_ids: list[int] | None = None) -> pd.DataFrame:
    """One row per team with leaguedashteamstats-shaped columns."""
    if team_ids is None:
        team_ids = [1610612737 + i for i in range(30)]
    rows = []
    for tid in team_ids:
        rows.append(
          {
              COL_TEAM_ID: tid,
              "PACE": 101.0,
              "OFF_RATING": 115.0,
              "DEF_RATING": 111.0,
              "NET_RATING": 4.0,
              "FGA": 88.0,
              "FG3A": 35.0,
              "AST": 25.0,
              "TOV": 13.0,
              "EFG_PCT": 0.54,
              "TS_PCT": 0.58,
              "AST_RATIO": 1.9,
              "AST_PCT": 0.62,
              "TM_TOV_PCT": 0.13,
              "Restricted Area | FGA": 25.0,
              "In The Paint (Non-RA) | FGA": 15.0,
              "Mid-Range | FGA": 8.0,
              "Left Corner 3 | FGA": 4.0,
              "Right Corner 3 | FGA": 4.0,
              "Above the Break 3 | FGA": 34.0,
            }
        )
    return pd.DataFrame(rows)
