"""Canonical lineup-unit and on/off tables for Option B interim Parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_fit.config.settings import (
    INGEST_TIER_ROLE,
    INTERIM_TABLE_LINEUP_UNITS,
    INTERIM_TABLE_ONOFF,
    LINEUP_COL_GROUP_ID,
    LINEUP_COL_GROUP_NAME,
    ONOFF_COL_COURT_STATUS,
    ONOFF_COL_VS_PLAYER_ID,
    ONOFF_COL_VS_PLAYER_NAME,
    ONOFF_SUMMARY_DATASETS,
)
from nba_fit.data.client import FetchResult
from nba_fit.data.fetchers.lineups_onoff import lineup_units_frame, primary_dataset_name
from nba_fit.data.storage import interim_path
from nba_fit.normalize.ids import canonical_player_id, canonical_team_id

_INTERIM_PARQUET_NAME = "data.parquet"

_COURT_STATUS_FROM_DATASET: dict[str, str] = {
    "PlayersOnCourtTeamPlayerOnOffSummary": "On",
    "PlayersOffCourtTeamPlayerOnOffSummary": "Off",
    "PlayersOnCourtTeamPlayerOnOffDetails": "On",
    "PlayersOffCourtTeamPlayerOnOffDetails": "Off",
}


def _season_partition_dir(table: str, season: str) -> Path:
    return interim_path(table) / f"season={season}"


def interim_lineup_units_path(season: str) -> Path:
    return _season_partition_dir(INTERIM_TABLE_LINEUP_UNITS, season) / _INTERIM_PARQUET_NAME


def interim_onoff_path(season: str) -> Path:
    return _season_partition_dir(INTERIM_TABLE_ONOFF, season) / _INTERIM_PARQUET_NAME


def _with_team_season(df: pd.DataFrame, season: str, *, source_endpoint: str) -> pd.DataFrame:
    out = df.copy()
    if "TEAM_ID" in out.columns:
        out["team_id"] = out["TEAM_ID"].map(canonical_team_id)
    out["season"] = season
    out["source_endpoint"] = source_endpoint
    return out


def _stack_onoff_frames(
    fetched: FetchResult,
    dataset_names: tuple[str, ...],
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for ds_name in dataset_names:
        if ds_name not in fetched.frames:
            continue
        chunk = fetched.frames[ds_name].copy()
        status = _COURT_STATUS_FROM_DATASET.get(ds_name)
        if status and ONOFF_COL_COURT_STATUS not in chunk.columns:
            chunk[ONOFF_COL_COURT_STATUS] = status
        chunk["onoff_dataset"] = ds_name
        chunks.append(chunk)
    if not chunks:
        raise ValueError(
            f"No on/off datasets in {fetched.endpoint}; "
            f"expected one of {dataset_names}, got {list(fetched.frames)}"
        )
    return pd.concat(chunks, ignore_index=True)


def build_lineup_units_table(
    season: str,
    fetched: dict[str, FetchResult],
) -> pd.DataFrame:
    """
    Merge five-man lineup rows from league- and team-dash lineup endpoints.

    Base rows come from ``leaguedashlineups``; ``teamdashlineups`` rows are appended
    with ``source_endpoint`` for traceability.
    """
    required = ("leaguedashlineups",)
    for key in required:
        if key not in fetched:
            raise ValueError(f"Missing required endpoint for lineup units table: {key}")

    parts: list[pd.DataFrame] = []
    for endpoint in ("leaguedashlineups", "teamdashlineups"):
        if endpoint not in fetched:
            continue
        frame = lineup_units_frame(fetched[endpoint])
        parts.append(_with_team_season(frame, season, source_endpoint=endpoint))

    combined = pd.concat(parts, ignore_index=True)
    if LINEUP_COL_GROUP_ID in combined.columns:
        combined["lineup_group_id"] = combined[LINEUP_COL_GROUP_ID]
    if LINEUP_COL_GROUP_NAME in combined.columns:
        combined["lineup_group_name"] = combined[LINEUP_COL_GROUP_NAME]
    return combined


def build_onoff_table(
    season: str,
    fetched: dict[str, FetchResult],
) -> pd.DataFrame:
    """
    Stack on-court / off-court partner rows from summary (and optional details).

    ``VS_PLAYER_ID`` is the focal teammate; ``COURT_STATUS`` is On or Off.
    """
    if "teamplayeronoffsummary" not in fetched:
        raise ValueError("Missing required endpoint for on/off table: teamplayeronoffsummary")

    summary = _stack_onoff_frames(fetched["teamplayeronoffsummary"], ONOFF_SUMMARY_DATASETS)
    out = _with_team_season(summary, season, source_endpoint="teamplayeronoffsummary")

    if ONOFF_COL_VS_PLAYER_ID in out.columns:
        out["focal_player_id"] = out[ONOFF_COL_VS_PLAYER_ID].map(canonical_player_id)
    if ONOFF_COL_VS_PLAYER_NAME in out.columns:
        out["focal_player_name"] = out[ONOFF_COL_VS_PLAYER_NAME]

    return out


def write_lineup_units_table(df: pd.DataFrame, season: str) -> Path:
    path = interim_lineup_units_path(season)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def write_onoff_table(df: pd.DataFrame, season: str) -> Path:
    path = interim_onoff_path(season)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_lineup_units_table(season: str) -> pd.DataFrame:
    path = interim_lineup_units_path(season)
    if not path.exists():
        raise FileNotFoundError(
            f"No lineup_units interim table for {season} at {path}. "
            f"Run: python -m nba_fit ingest --season {season} --tier {INGEST_TIER_ROLE}"
        )
    return pd.read_parquet(path)


def load_onoff_table(season: str) -> pd.DataFrame:
    path = interim_onoff_path(season)
    if not path.exists():
        raise FileNotFoundError(
            f"No onoff interim table for {season} at {path}. "
            f"Run: python -m nba_fit ingest --season {season} --tier {INGEST_TIER_ROLE}"
        )
    return pd.read_parquet(path)


def onoff_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep on-court/off-court summary partner rows (drop Overall aggregates)."""
    if "onoff_dataset" not in df.columns:
        return df
    return df.loc[df["onoff_dataset"].isin(ONOFF_SUMMARY_DATASETS)].copy()


def primary_lineup_dataset(endpoint: str) -> str:
    """Expose primary dataset name for tests and tooling."""
    return primary_dataset_name(endpoint)
