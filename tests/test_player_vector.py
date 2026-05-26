"""Tests for interpretable player feature vectors (synthetic leaguedash-shaped data)."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_fit.features.constants import (
    COL_MINUTES_STABLE,
    COL_PLAYER_ID,
    COL_ROLE_BUCKET,
    MIN_SEASON_MINUTES_STABLE_RATES,
    PLAYER_FEATURE_PREFIX,
)
from nba_fit.features.player_vector import (
    FEATURE_GROUP_DEFENSE,
    FEATURE_GROUP_ROLE_VOLUME,
    PLAYER_FEATURE_GROUPS,
    build_player_features,
    infer_role_bucket,
)
from nba_fit.features.scaling import scale_player_features
from nba_fit.features.team_vector import aggregate_roster_composition, build_team_features


def _synthetic_player_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_PLAYER_ID: [1, 2],
            "TEAM_ID": [10, 10],
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


def test_build_player_features_shape_and_groups():
    raw = _synthetic_player_df()
    feats = build_player_features(raw)

    assert len(feats) == 2
    assert COL_PLAYER_ID in feats.columns
    assert COL_ROLE_BUCKET in feats.columns
    assert COL_MINUTES_STABLE in feats.columns

    pf_cols = [c for c in feats.columns if c.startswith(PLAYER_FEATURE_PREFIX)]
    assert pf_cols, "expected pf_* feature columns"

    for group in PLAYER_FEATURE_GROUPS:
        assert any(c.startswith(f"{PLAYER_FEATURE_PREFIX}{group}__") for c in pf_cols), group

    assert bool(feats.loc[0, COL_MINUTES_STABLE])
    assert feats.loc[0, COL_MINUTES_STABLE]  # 34*60 > 500
    total_min_1 = 12.0 * 30.0
    assert total_min_1 < MIN_SEASON_MINUTES_STABLE_RATES
    assert not feats.loc[1, COL_MINUTES_STABLE]


def test_role_bucket_from_position():
    raw = _synthetic_player_df()
    buckets = infer_role_bucket(raw)
    assert buckets.iloc[0] == "G"
    assert buckets.iloc[1] == "B"


def test_scale_player_features_adds_z_columns():
    feats = build_player_features(_synthetic_player_df())
    scaled = scale_player_features(feats)
    z_cols = [c for c in scaled.columns if c.startswith("z_pf_")]
    assert len(z_cols) == len([c for c in feats.columns if c.startswith(PLAYER_FEATURE_PREFIX)])


def test_team_features_and_roster_agg():
    players = _synthetic_player_df()
    roster = aggregate_roster_composition(players)
    assert len(roster) == 1
    assert roster.loc[0, "roster_n_players"] == 2
    assert roster.loc[0, "roster_top_usg_pct"] == pytest.approx(0.32)

    team_raw = pd.DataFrame(
        {
            "TEAM_ID": [10],
            "PACE": [101.5],
            "OFF_RATING": [116.0],
            "DEF_RATING": [111.0],
            "NET_RATING": [5.0],
            "FGA": [90.0],
            "FG3A": [36.0],
            "AST": [25.0],
            "TOV": [14.0],
            "EFG_PCT": [0.54],
            "TS_PCT": [0.58],
            "AST_RATIO": [1.9],
            "AST_PCT": [0.62],
            "TM_TOV_PCT": [0.13],
            "Restricted Area | FGA": [25.0],
            "In The Paint (Non-RA) | FGA": [15.0],
            "Mid-Range | FGA": [8.0],
            "Left Corner 3 | FGA": [4.0],
            "Right Corner 3 | FGA": [4.0],
            "Above the Break 3 | FGA": [34.0],
        }
    )
    team_feats = build_team_features(team_raw, roster_agg=roster)
    assert "tf_roster_composition__n_players" in team_feats.columns
    assert team_feats.loc[0, "tf_roster_composition__n_players"] == 2
