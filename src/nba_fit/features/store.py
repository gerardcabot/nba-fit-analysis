"""Feature-store writers: scaled player/team matrices and scaling params."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nba_fit.config.settings import get_settings
from nba_fit.data.storage import ensure_dirs, feature_path
from nba_fit.features.constants import (
    COL_PLAYER_ID,
    COL_ROLE_BUCKET,
    COL_TEAM_ID,
    MIN_BUCKET_SIZE_FOR_SCALING,
    PLAYER_FEATURE_PREFIX,
    TEAM_FEATURE_PREFIX,
)
from nba_fit.features.player_vector import build_player_features
from nba_fit.features.scaling import robust_median_mad, scale_player_features, scale_team_features
from nba_fit.features.team_vector import aggregate_roster_composition, build_team_features
from nba_fit.normalize.players import load_players_table
from nba_fit.normalize.teams import load_teams_table

FEATURE_MATRIX_PLAYER: str = "player_features_scaled"
FEATURE_MATRIX_TEAM: str = "team_features_scaled"
FEATURE_MATRIX_SCALING: str = "scaling_params"


@dataclass(frozen=True)
class FeatureStoreResult:
    """Paths and row counts from one feature materialization run."""

    season: str
    player_path: Path
    team_path: Path
    scaling_path: Path
    player_rows: int
    team_rows: int
    scaling_rows: int


def _extract_scaling_params(
    player_features: pd.DataFrame,
    team_features: pd.DataFrame,
    *,
    season: str,
    bucket_col: str = COL_ROLE_BUCKET,
) -> pd.DataFrame:
    """Build long-form center/scale table for player (by bucket) and team features."""
    rows: list[dict[str, object]] = []

    player_cols = [c for c in player_features.columns if c.startswith(PLAYER_FEATURE_PREFIX)]
    if bucket_col not in player_features.columns:
        player_features = player_features.copy()
        player_features[bucket_col] = "ALL"

    for col in player_cols:
        global_center, global_scale = robust_median_mad(player_features[col])
        for bucket, idx in player_features.groupby(bucket_col, dropna=False).groups.items():
            series = player_features.loc[idx, col]
            if len(series.dropna()) >= MIN_BUCKET_SIZE_FOR_SCALING:
                center, scale = robust_median_mad(series)
            else:
                center, scale = global_center, global_scale
            rows.append(
                {
                    "season": season,
                    "entity": "player",
                    "feature": col,
                    "bucket": str(bucket),
                    COL_PLAYER_ID: None,
                    COL_TEAM_ID: None,
                    "center": center,
                    "scale": scale,
                }
            )

    team_cols = [c for c in team_features.columns if c.startswith(TEAM_FEATURE_PREFIX)]
    for col in team_cols:
        center, scale = robust_median_mad(team_features[col])
        rows.append(
            {
                "season": season,
                "entity": "team",
                "feature": col,
                "bucket": "ALL",
                COL_PLAYER_ID: None,
                COL_TEAM_ID: None,
                "center": center,
                "scale": scale,
            }
        )

    return pd.DataFrame(rows)


def write_player_features_scaled(df: pd.DataFrame, season: str) -> Path:
    """Persist scaled player feature matrix under ``data/features/``."""
    out_path = feature_path(FEATURE_MATRIX_PLAYER, season)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def write_team_features_scaled(df: pd.DataFrame, season: str) -> Path:
    """Persist scaled team feature matrix under ``data/features/``."""
    out_path = feature_path(FEATURE_MATRIX_TEAM, season)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def write_scaling_params(df: pd.DataFrame, season: str) -> Path:
    """Persist robust scaling center/scale params under ``data/features/``."""
    out_path = feature_path(FEATURE_MATRIX_SCALING, season)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def materialize_features(season: str | None = None) -> FeatureStoreResult:
    """
    Build raw features from interim player/team tables, scale, and write Parquet.

    Requires ``data/interim/players`` and ``data/interim/teams`` for *season*.
    """
    settings = get_settings()
    season = season or settings.default_season
    ensure_dirs()

    players_raw = load_players_table(season)
    teams_raw = load_teams_table(season)

    player_feats = build_player_features(players_raw)
    player_scaled = scale_player_features(player_feats)
    roster = aggregate_roster_composition(players_raw)
    team_feats = build_team_features(teams_raw, roster_agg=roster)
    team_scaled = scale_team_features(team_feats)
    scaling = _extract_scaling_params(player_feats, team_feats, season=season)

    player_path = write_player_features_scaled(player_scaled, season)
    team_path = write_team_features_scaled(team_scaled, season)
    scaling_path = write_scaling_params(scaling, season)

    return FeatureStoreResult(
        season=season,
        player_path=player_path,
        team_path=team_path,
        scaling_path=scaling_path,
        player_rows=len(player_scaled),
        team_rows=len(team_scaled),
        scaling_rows=len(scaling),
    )
