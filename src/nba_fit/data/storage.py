"""Parquet path layout, partitioning rules, and chunked reads for NBA Fit data.

Volume rationale (local probe meta, ``probe_all_results.json``):
    - **143** ``nba_api`` endpoints catalogued; **106** HTTP successes, **100** with rows.
    - **Play-by-play** (``playbyplayv3``): ~**500** events per game in probe sample
      (499 PlayByPlay rows + metadata).
    - **Regular season scale**: ~**1,230** games/season (30 teams × 82 games / 2).
    - **PBP order-of-magnitude**: 1,230 × 500 ≈ **615k** rows/season — fine on disk,
      impractical to hold all seasons in RAM at once.
    - **Shot charts** (``shotchartdetail``): ~**1,457** FGA rows per star season in probe;
      × ~500 rotation players ≈ **0.7M** rows/season if fully materialized.
    - **Lineups** (``leaguedashlineups``): ~**641** rows per team-season in probe;
      × 30 teams ≈ **19k** rows/season.
    - **League dashboards** (100+ endpoints): mostly season-scoped tables from hundreds
      to low millions of rows combined across endpoints — still **single-digit GB** in
      Parquet for a few seasons, vs. tens of GB if duplicated in memory per worker.

Hybrid strategy: cache **raw** API pulls as Parquet (partitioned, append-friendly),
normalize into **interim** hive datasets, materialize **feature** matrices per season,
and keep **ephemeral** scratch (joins, single-game probes) off the durable tree.

Partitioning rules
------------------
* **Default league / dashboard tables** — ``season`` only::
      ``data/raw/nba_api/{endpoint}/season={season}/``
* **shotchartdetail** — ``season`` + ``player_id`` (one player-season per file/chunk)::
      ``.../season={season}/player_id={player_id}/``
* **playbyplayv3** — ``game_id`` (one game per chunk; season encoded in game id)::
      ``.../playbyplayv3/game_id={game_id}/``
* **leaguedashlineups**, **teamdashlineups** — ``season`` + ``team_id``::
      ``.../season={season}/team_id={team_id}/``
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds

from nba_fit.config.settings import get_settings

# Hive partition column order per endpoint (see module docstring).
_ENDPOINT_PARTITION_KEYS: dict[str, tuple[str, ...]] = {
    "shotchartdetail": ("season", "player_id"),
    "playbyplayv3": ("game_id",),
    "leaguedashlineups": ("season", "team_id"),
    "teamdashlineups": ("season", "team_id"),
}
_DEFAULT_PARTITION_KEYS: tuple[str, ...] = ("season",)

_RAW_SOURCE = "nba_api"
_DEFAULT_PARQUET_NAME = "data.parquet"


class StorageStrategy(str, Enum):
    """Where a pipeline stage persists data."""

    RAW_PARQUET = "raw_parquet"
    INTERIM_PARQUET = "interim_parquet"
    FEATURE_PARQUET = "feature_parquet"
    EPHEMERAL = "ephemeral"


def _partition_keys_for(endpoint: str) -> tuple[str, ...]:
    return _ENDPOINT_PARTITION_KEYS.get(endpoint.lower(), _DEFAULT_PARTITION_KEYS)


def _format_partition(key: str, value: Any) -> str:
    return f"{key}={value}"


def _base_for(strategy: StorageStrategy) -> Path:
    settings = get_settings()
    return {
        StorageStrategy.RAW_PARQUET: settings.data_raw,
        StorageStrategy.INTERIM_PARQUET: settings.data_interim,
        StorageStrategy.FEATURE_PARQUET: settings.data_features,
        StorageStrategy.EPHEMERAL: settings.root / "data" / ".ephemeral",
    }[strategy]


def raw_path(
    endpoint: str,
    season: str | None = None,
    *,
    source: str = _RAW_SOURCE,
    filename: str = _DEFAULT_PARQUET_NAME,
    **partition_keys: Any,
) -> Path:
    """Path for a raw cached Parquet object.

    ``season`` is always accepted; endpoints that partition only by ``game_id``
    (``playbyplayv3``) ignore it for the directory layout but callers may still
    pass it for logging or fetch kwargs.

    Additional ``partition_keys`` must supply every non-season key required by the
    endpoint (e.g. ``player_id``, ``team_id``, ``game_id``).
    """
    endpoint = endpoint.lower()
    required = _partition_keys_for(endpoint)
    parts: list[str] = []

    if "season" in required:
        if season is None:
            raise ValueError(f"endpoint {endpoint!r} requires season=")
        parts.append(_format_partition("season", season))

    for key in required:
        if key == "season":
            continue
        if key not in partition_keys:
            raise ValueError(
                f"endpoint {endpoint!r} requires partition key {key!r}; "
                f"got {sorted(partition_keys)}"
            )
        parts.append(_format_partition(key, partition_keys[key]))

    directory = _base_for(StorageStrategy.RAW_PARQUET) / source / endpoint
    for part in parts:
        directory /= part
    return directory / filename


def interim_path(table: str) -> Path:
    """Hive dataset root under ``data/interim/{table}/``."""
    return _base_for(StorageStrategy.INTERIM_PARQUET) / table.lower()


def feature_path(
    matrix_name: str,
    season: str,
    *,
    filename: str = _DEFAULT_PARQUET_NAME,
) -> Path:
    """Path under ``data/features/{matrix_name}/season={season}/``."""
    directory = (
        _base_for(StorageStrategy.FEATURE_PARQUET)
        / matrix_name.lower()
        / _format_partition("season", season)
    )
    return directory / filename


def ephemeral_path(*parts: str) -> Path:
    """Scratch path under ``data/.ephemeral/`` (not for long-term cache)."""
    base = _base_for(StorageStrategy.EPHEMERAL)
    return base.joinpath(*parts) if parts else base


def ensure_dirs() -> None:
    """Create all standard data directories if missing."""
    settings = get_settings()
    for path in (
        settings.data_raw,
        settings.data_interim,
        settings.data_features,
        settings.models_dir,
        settings.reports_dir,
        _base_for(StorageStrategy.EPHEMERAL),
    ):
        path.mkdir(parents=True, exist_ok=True)


def read_chunks(
    path: Path | str,
    *,
    columns: list[str] | None = None,
    batch_size: int = 65_536,
    filter_expr: ds.Expression | None = None,
) -> Iterator[pa.RecordBatch]:
    """Iterate RecordBatches from a Parquet file or hive-partitioned dataset.

    Use for large tables (PBP, shot charts, league dashboards) without loading
    the full table into memory.
    """
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    scanner = dataset.scanner(
        columns=columns,
        batch_size=batch_size,
        filter=filter_expr,
    )
    yield from scanner.to_batches()
