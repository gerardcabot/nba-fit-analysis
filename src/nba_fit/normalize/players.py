"""Canonical player-season table in data/interim/players/."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from nba_fit.config.settings import INTERIM_TABLE_PLAYERS, PLAYER_STAT_MIN
from nba_fit.data.client import FetchResult
from nba_fit.data.fetchers.league_dash import primary_frame
from nba_fit.data.storage import interim_path
from nba_fit.normalize.ids import canonical_player_id, canonical_team_id

_INTERIM_PARQUET_NAME = "data.parquet"


def _season_partition_dir(season: str) -> Path:
    return interim_path(INTERIM_TABLE_PLAYERS) / f"season={season}"


def interim_players_path(season: str) -> Path:
    """Path to canonical player-season Parquet for *season*."""
    return _season_partition_dir(season) / _INTERIM_PARQUET_NAME


def _with_ids(df: pd.DataFrame, season: str) -> pd.DataFrame:
    out = df.copy()
    if "PLAYER_ID" in out.columns:
        out["player_id"] = out["PLAYER_ID"].map(canonical_player_id)
    if "TEAM_ID" in out.columns:
        out["team_id"] = out["TEAM_ID"].map(canonical_team_id)
    out["season"] = season
    return out


def _merge_on_player(
    base: pd.DataFrame,
    other: pd.DataFrame,
    *,
    suffix: str,
) -> pd.DataFrame:
    """Left-join *other* onto *base* by player_id, avoiding duplicate ID/name cols."""
    overlap = set(base.columns) & set(other.columns) - {"player_id", "season"}
    other_cols = [c for c in other.columns if c not in overlap or c in ("player_id", "season")]
    return base.merge(other[other_cols], on=["player_id", "season"], how="left", suffixes=("", suffix))


def build_players_table(
    season: str,
    fetched: dict[str, FetchResult],
) -> pd.DataFrame:
    """
    Merge Option A player endpoints into one wide player-season table.

    Base: leaguedashplayerstats; joined: estimated metrics, bio, shot locations.
    """
    required = ("leaguedashplayerstats",)
    for key in required:
        if key not in fetched:
            raise ValueError(f"Missing required endpoint for players table: {key}")

    base = _with_ids(primary_frame(fetched["leaguedashplayerstats"]), season)

    optional_sources: tuple[tuple[str, str], ...] = (
        ("playerestimatedmetrics", "_est"),
        ("leaguedashplayerbiostats", "_bio"),
        ("leaguedashplayershotlocations", "_shot"),
    )
    for endpoint, suffix in optional_sources:
        if endpoint not in fetched:
            continue
        extra = _with_ids(primary_frame(fetched[endpoint]), season)
        base = _merge_on_player(base, extra, suffix=suffix)

    return base


def write_players_table(df: pd.DataFrame, season: str) -> Path:
    """Write canonical players Parquet; return output path."""
    path = interim_players_path(season)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_players_table(season: str) -> pd.DataFrame:
    """Load canonical players table for *season*."""
    path = interim_players_path(season)
    if not path.exists():
        from nba_fit.config.settings import INGEST_TIER_MVP

        raise FileNotFoundError(
            f"No players interim table for {season} at {path}. "
            f"Run: python -m nba_fit ingest --season {season} --tier {INGEST_TIER_MVP}"
        )
    return pd.read_parquet(path)


def rotation_minutes_from_players(players_raw: pd.DataFrame) -> dict[int, float]:
    """Map player_id -> season minutes from league-dash player table."""
    pid_col = "player_id" if "player_id" in players_raw.columns else "PLAYER_ID"
    min_col = PLAYER_STAT_MIN if PLAYER_STAT_MIN in players_raw.columns else "MIN"
    if pid_col not in players_raw.columns or min_col not in players_raw.columns:
        return {}
    out: dict[int, float] = {}
    for _, row in players_raw.iterrows():
        val = row[min_col]
        if pd.isna(val):
            continue
        if isinstance(val, str) and ":" in val:
            parts = val.split(":")
            val = float(parts[0]) + float(parts[1]) / 60.0
        out[int(row[pid_col])] = float(val)
    return out


def players_with_stable_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to players meeting MIN_SEASON_MINUTES_STABLE_RATES (feature eligibility)."""
    from nba_fit.config.settings import MIN_SEASON_MINUTES_STABLE_RATES

    if PLAYER_STAT_MIN not in df.columns:
        return df
    minutes = pd.to_numeric(df[PLAYER_STAT_MIN], errors="coerce")
    return df.loc[minutes >= MIN_SEASON_MINUTES_STABLE_RATES].copy()
