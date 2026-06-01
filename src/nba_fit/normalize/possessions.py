"""Canonical possession / stint rows for Option C interim Parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from nba_fit.config.settings import (
    INGEST_TIER_IMPACT,
    INTERIM_TABLE_POSSESSIONS,
    get_settings,
)
from nba_fit.data.client import FetchResult
from nba_fit.data.fetchers.pbp import fetch_option_c_impact
from nba_fit.data.pbpstats_adapter import build_possession_rows
from nba_fit.data.storage import interim_path
from nba_fit.normalize.ids import canonical_game_id

_INTERIM_PARQUET_NAME = "data.parquet"


def _season_partition_dir(season: str) -> Path:
    return interim_path(INTERIM_TABLE_POSSESSIONS) / f"season={season}"


def interim_possessions_game_path(season: str, game_id: str) -> Path:
    """Hive path: ``possessions/season={season}/game_id={game_id}/data.parquet``."""
    gid = canonical_game_id(game_id)
    if gid is None:
        raise ValueError(f"Invalid game_id: {game_id!r}")
    return _season_partition_dir(season) / f"game_id={gid}" / _INTERIM_PARQUET_NAME


def build_possessions_table(
    season: str,
    *,
    game_ids: list[str],
    pbp_by_game: dict[str, FetchResult],
    rotation_by_game: dict[str, FetchResult],
    pbpstats_cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Build possession rows for all requested games."""
    settings = get_settings()
    cache_dir = pbpstats_cache_dir or (settings.data_raw / "pbpstats")
    return build_possession_rows(
        season=season,
        game_ids=game_ids,
        pbp_by_game=pbp_by_game,
        rotation_by_game=rotation_by_game,
        pbpstats_cache_dir=cache_dir,
    )


def write_possessions_table(df: pd.DataFrame, season: str) -> list[Path]:
    """
    Write possession Parquet partitioned by ``season`` + ``game_id``.

    Returns list of written file paths.
    """
    if df.empty:
        return []
    if "game_id" not in df.columns:
        raise KeyError("possessions table requires game_id column")

    written: list[Path] = []
    for game_id, group in df.groupby("game_id", sort=False):
        gid = canonical_game_id(game_id)
        if gid is None:
            continue
        path = interim_possessions_game_path(season, gid)
        path.parent.mkdir(parents=True, exist_ok=True)
        group.to_parquet(path, index=False)
        written.append(path)
    return written


def load_possessions_table(
    season: str,
    *,
    game_id: str | None = None,
) -> pd.DataFrame:
    """Load possession rows for *season* (optional single *game_id*)."""
    base = _season_partition_dir(season)
    if game_id is not None:
        path = interim_possessions_game_path(season, game_id)
        if not path.exists():
            raise FileNotFoundError(
                f"No possessions for game {game_id} season {season} at {path}. "
                f"Run: python -m nba_fit ingest --season {season} --tier {INGEST_TIER_IMPACT}"
            )
        return pd.read_parquet(path)

    if not base.exists():
        raise FileNotFoundError(
            f"No possessions interim data for {season} at {base}. "
            f"Run: python -m nba_fit ingest --season {season} --tier {INGEST_TIER_IMPACT}"
        )

    parts: list[pd.DataFrame] = []
    for game_dir in sorted(base.glob("game_id=*")):
        path = game_dir / _INTERIM_PARQUET_NAME
        if path.is_file():
            parts.append(pd.read_parquet(path))
    if not parts:
        raise FileNotFoundError(f"No possession parquet files under {base}")
    return pd.concat(parts, ignore_index=True)


def has_possessions_partition(season: str) -> bool:
    """True when interim possessions hive partition exists with at least one parquet file."""
    base = _season_partition_dir(season)
    return base.exists() and any(base.rglob(_INTERIM_PARQUET_NAME))


def possessions_per_game_counts(df: pd.DataFrame) -> pd.Series:
    """Count possession rows per ``game_id`` (for visual validation)."""
    if df.empty or "game_id" not in df.columns:
        return pd.Series(dtype=int)
    return df.groupby("game_id").size()


def build_and_write_from_fetched(
    season: str,
    fetched: dict[str, Any],
    *,
    pbpstats_cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Normalize fetched Option C bundle and write hive partitions."""
    game_ids: list[str] = list(fetched.get("game_ids") or [])
    pbp_by_game: dict[str, FetchResult] = fetched.get("playbyplayv3") or {}
    rotation_by_game: dict[str, FetchResult] = fetched.get("gamerotation") or {}
    df = build_possessions_table(
        season,
        game_ids=game_ids,
        pbp_by_game=pbp_by_game,
        rotation_by_game=rotation_by_game,
        pbpstats_cache_dir=pbpstats_cache_dir,
    )
    paths = write_possessions_table(df, season)
    return df, paths


def ingest_possessions_from_client(
    client: Any,
    *,
    season: str,
    use_cache: bool = True,
    max_games: int | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Fetch Option C endpoints via *client* and write possession interim tables."""
    fetched = fetch_option_c_impact(
        client,
        season=season,
        use_cache=use_cache,
        max_games=max_games,
    )
    return build_and_write_from_fetched(season, fetched)
