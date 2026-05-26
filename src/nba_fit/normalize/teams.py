"""Canonical team-season table in data/interim/teams/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_fit.config.settings import INTERIM_TABLE_TEAMS
from nba_fit.data.client import FetchResult
from nba_fit.data.fetchers.league_dash import primary_frame
from nba_fit.data.storage import interim_path
from nba_fit.normalize.ids import canonical_team_id

_INTERIM_PARQUET_NAME = "data.parquet"


def _season_partition_dir(season: str) -> Path:
    return interim_path(INTERIM_TABLE_TEAMS) / f"season={season}"


def interim_teams_path(season: str) -> Path:
    """Path to canonical team-season Parquet for *season*."""
    return _season_partition_dir(season) / _INTERIM_PARQUET_NAME


def _with_ids(df: pd.DataFrame, season: str) -> pd.DataFrame:
    out = df.copy()
    if "TEAM_ID" in out.columns:
        out["team_id"] = out["TEAM_ID"].map(canonical_team_id)
    out["season"] = season
    return out


def _merge_on_team(
    base: pd.DataFrame,
    other: pd.DataFrame,
    *,
    suffix: str,
) -> pd.DataFrame:
    overlap = set(base.columns) & set(other.columns) - {"team_id", "season"}
    other_cols = [c for c in other.columns if c not in overlap or c in ("team_id", "season")]
    return base.merge(other[other_cols], on=["team_id", "season"], how="left", suffixes=("", suffix))


def build_teams_table(
    season: str,
    fetched: dict[str, FetchResult],
) -> pd.DataFrame:
    """
    Merge Option A team endpoints into one wide team-season table.

    Base: leaguedashteamstats; joined: estimated metrics, shot locations.
    """
    if "leaguedashteamstats" not in fetched:
        raise ValueError("Missing required endpoint for teams table: leaguedashteamstats")

    base = _with_ids(primary_frame(fetched["leaguedashteamstats"]), season)

    for endpoint, suffix in (
        ("teamestimatedmetrics", "_est"),
        ("leaguedashteamshotlocations", "_shot"),
    ):
        if endpoint not in fetched:
            continue
        extra = _with_ids(primary_frame(fetched[endpoint]), season)
        base = _merge_on_team(base, extra, suffix=suffix)

    return base


def write_teams_table(df: pd.DataFrame, season: str) -> Path:
    """Write canonical teams Parquet; return output path."""
    path = interim_teams_path(season)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_teams_table(season: str) -> pd.DataFrame:
    """Load canonical teams table for *season*."""
    path = interim_teams_path(season)
    if not path.exists():
        raise FileNotFoundError(
            f"No teams interim table for {season} at {path}. "
            f"Run: python -m nba_fit ingest --season {season} --tier mvp"
        )
    return pd.read_parquet(path)
