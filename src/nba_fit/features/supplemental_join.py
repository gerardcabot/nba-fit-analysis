"""Join tactical, on/off, and lineup interim tables into leaguedash-wide frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_fit.config.settings import (
    ONOFF_COL_COURT_STATUS,
    ONOFF_COL_VS_PLAYER_ID,
    ONOFF_STAT_MIN,
    ONOFF_STAT_NET_RATING,
    TACTICAL_PRIMARY_DATASET,
)
from nba_fit.data.storage import raw_path
from nba_fit.features._column_utils import find_column, safe_divide
from nba_fit.features.constants import COL_PLAYER_ID, COL_TEAM_ID
from nba_fit.normalize.ids import canonical_player_id, canonical_team_id
from nba_fit.normalize.lineups import (
    load_lineup_units_table,
    load_onoff_table,
    onoff_summary_rows,
)

SUPPLEMENTAL_COVERAGE: dict[str, str] = {}


def _read_raw_endpoint(season: str, endpoint: str) -> pd.DataFrame | None:
    """Load primary dataset from raw Parquet cache if present."""
    path = raw_path(endpoint, season=season)
    if not path.is_file():
        SUPPLEMENTAL_COVERAGE[endpoint] = "missing"
        return None
    combined = pd.read_parquet(path)
    if "_dataset" in combined.columns:
        ds_name = TACTICAL_PRIMARY_DATASET.get(endpoint.lower())
        if ds_name and ds_name in combined["_dataset"].unique():
            frame = combined.loc[combined["_dataset"] == ds_name].drop(
                columns=["_dataset"], errors="ignore"
            )
        else:
            frame = combined.drop(columns=["_dataset"], errors="ignore")
    else:
        frame = combined
    SUPPLEMENTAL_COVERAGE[endpoint] = "joined"
    return frame.reset_index(drop=True)


def _player_id_series(df: pd.DataFrame) -> pd.Series | None:
    col = find_column(df, COL_PLAYER_ID, "PLAYERID", "PLAYER_ID", "CLOSE_DEF_PERSON_ID")
    if col is None:
        return None
    return df[col].map(canonical_player_id)


def join_tactical_into_players(players: pd.DataFrame, season: str) -> pd.DataFrame:
    """Left-join hustle, pt-defend, and gravity columns when raw cache exists."""
    out = players.copy()
    pid_col = find_column(out, COL_PLAYER_ID, "player_id")
    if pid_col is None:
        return out

    hustle = _read_raw_endpoint(season, "leaguehustlestatsplayer")
    if hustle is not None:
        h_pid = _player_id_series(hustle)
        if h_pid is not None:
            hustle = hustle.assign(_pid=h_pid)
            hustle_cols = [
                c
                for c in (
                    "DEFLECTIONS",
                    "CHARGES_DRAWN",
                    "LOOSE_BALLS_RECOVERED",
                    "SCREEN_ASSISTS",
                    "CONTESTED_SHOTS",
                    "BOX_OUTS",
                )
                if c in hustle.columns
            ]
            if hustle_cols:
                agg = hustle.groupby("_pid", dropna=True)[hustle_cols].mean().reset_index()
                agg = agg.rename(columns={"_pid": "_join_pid"})
                out = out.merge(
                    agg,
                    left_on=pid_col,
                    right_on="_join_pid",
                    how="left",
                    suffixes=("", "_hustle"),
                )
                out = out.drop(columns=["_join_pid"], errors="ignore")

    defend = _read_raw_endpoint(season, "leaguedashptdefend")
    if defend is not None:
        d_pid = _player_id_series(defend)
        if d_pid is not None:
            defend = defend.assign(_pid=d_pid)
            defend_cols = [c for c in ("FREQ", "D_FG_PCT", "PCT_PLUSMINUS") if c in defend.columns]
            if defend_cols:
                agg = defend.groupby("_pid", dropna=True)[defend_cols].mean().reset_index()
                rename = {c: f"DEFEND_{c}" for c in defend_cols}
                agg = agg.rename(columns={"_pid": "_join_pid", **rename})
                out = out.merge(agg, left_on=pid_col, right_on="_join_pid", how="left")
                out = out.drop(columns=["_join_pid"], errors="ignore")

    gravity = _read_raw_endpoint(season, "gravityleaders")
    if gravity is not None:
        g_pid = _player_id_series(gravity)
        if g_pid is not None:
            gravity = gravity.assign(_pid=g_pid)
            g_cols = [c for c in ("GRAVITYSCORE", "AVGGRAVITYSCORE") if c in gravity.columns]
            if g_cols:
                agg = gravity.groupby("_pid", dropna=True)[g_cols].mean().reset_index()
                agg = agg.rename(columns={"_pid": "_join_pid"})
                out = out.merge(agg, left_on=pid_col, right_on="_join_pid", how="left")
                out = out.drop(columns=["_join_pid"], errors="ignore")

    return out


def _aggregate_player_onoff(onoff: pd.DataFrame) -> pd.DataFrame:
    """Per focal player: weighted on-court minus off-court NET_RATING."""
    pid_col = find_column(onoff, "focal_player_id", ONOFF_COL_VS_PLAYER_ID)
    if pid_col is None or ONOFF_STAT_NET_RATING not in onoff.columns:
        return pd.DataFrame()

    work = onoff_summary_rows(onoff).copy()
    status_col = ONOFF_COL_COURT_STATUS if ONOFF_COL_COURT_STATUS in work.columns else None
    if status_col is None:
        return pd.DataFrame()

    work["_net"] = pd.to_numeric(work[ONOFF_STAT_NET_RATING], errors="coerce")
    work["_min"] = pd.to_numeric(
        work[ONOFF_STAT_MIN] if ONOFF_STAT_MIN in work.columns else 1.0,
        errors="coerce",
    ).fillna(1.0)

    rows: list[dict[str, float | int]] = []
    for focal_id, group in work.groupby(pid_col, dropna=True):
        on_mask = group[status_col].astype(str).str.lower() == "on"
        off_mask = group[status_col].astype(str).str.lower() == "off"
        on_net = safe_divide(
            (group.loc[on_mask, "_net"] * group.loc[on_mask, "_min"]).sum(),
            group.loc[on_mask, "_min"].sum(),
        )
        off_net = safe_divide(
            (group.loc[off_mask, "_net"] * group.loc[off_mask, "_min"]).sum(),
            group.loc[off_mask, "_min"].sum(),
        )
        on_min = float(group.loc[on_mask, "_min"].sum()) if on_mask.any() else 0.0
        rows.append(
            {
                COL_PLAYER_ID: int(focal_id),
                "ONOFF_NET_RATING_ON": float(on_net) if pd.notna(on_net) else np.nan,
                "ONOFF_NET_RATING_OFF": float(off_net) if pd.notna(off_net) else np.nan,
                "ONOFF_NET_DELTA": float(on_net - off_net)
                if pd.notna(on_net) and pd.notna(off_net)
                else np.nan,
                "ONOFF_ON_MINUTES": on_min,
            }
        )
    return pd.DataFrame(rows)


def join_onoff_into_players(players: pd.DataFrame, season: str) -> pd.DataFrame:
    """Attach on/off NET_RATING deltas when role-tier interim on/off exists."""
    out = players.copy()
    pid_col = find_column(out, COL_PLAYER_ID, "player_id")
    if pid_col is None:
        return out
    try:
        onoff = load_onoff_table(season)
    except FileNotFoundError:
        SUPPLEMENTAL_COVERAGE["onoff"] = "missing"
        return out

    agg = _aggregate_player_onoff(onoff)
    if agg.empty:
        SUPPLEMENTAL_COVERAGE["onoff"] = "empty"
        return out

    SUPPLEMENTAL_COVERAGE["onoff"] = "joined"
    return out.merge(agg, on=pid_col, how="left")


def _aggregate_team_lineup_weakness(lineups: pd.DataFrame) -> pd.DataFrame:
    """Team-level lineup weakness proxies from ``leaguedashlineups`` rows."""
    team_col = find_column(lineups, COL_TEAM_ID, "team_id", "TEAM_ID")
    net_col = find_column(lineups, "NET_RATING", "NET_RATING_RANK")
    min_col = find_column(lineups, "MIN", "MINUTES")
    if team_col is None or net_col is None:
        return pd.DataFrame()

    work = lineups.copy()
    if "team_id" not in work.columns and "TEAM_ID" in work.columns:
        work["team_id"] = work["TEAM_ID"].map(canonical_team_id)
    tid = "team_id" if "team_id" in work.columns else team_col

    work["_net"] = pd.to_numeric(work[net_col], errors="coerce")
    work["_min"] = pd.to_numeric(work[min_col], errors="coerce").fillna(0.0) if min_col else 1.0

    def _team_agg(group: pd.DataFrame) -> pd.Series:
        nets = group["_net"].dropna()
        mins = group["_min"]
        if nets.empty:
            return pd.Series(
                {
                    "LINEUP_WORST_NET_RATING": np.nan,
                    "LINEUP_MEAN_NET_RATING": np.nan,
                    "LINEUP_BAD_MIN_SHARE": np.nan,
                }
            )
        worst = float(nets.min())
        mean_net = float(nets.mean())
        bad_mask = nets < 0
        bad_min = float(mins.loc[bad_mask.index[bad_mask]].sum()) if bad_mask.any() else 0.0
        total_min = float(mins.sum()) or 1.0
        return pd.Series(
            {
                "LINEUP_WORST_NET_RATING": worst,
                "LINEUP_MEAN_NET_RATING": mean_net,
                "LINEUP_BAD_MIN_SHARE": bad_min / total_min,
            }
        )

    agg = work.groupby(tid, dropna=False).apply(_team_agg, include_groups=False).reset_index()
    agg = agg.rename(columns={tid: COL_TEAM_ID})
    return agg


def join_lineup_into_teams(teams: pd.DataFrame, season: str) -> pd.DataFrame:
    """Attach lineup weakness aggregates when role-tier lineup_units interim exists."""
    out = teams.copy()
    team_col = find_column(out, COL_TEAM_ID, "team_id")
    if team_col is None:
        return out
    try:
        lineups = load_lineup_units_table(season)
    except FileNotFoundError:
        SUPPLEMENTAL_COVERAGE["lineup_units"] = "missing"
        return out

    agg = _aggregate_team_lineup_weakness(lineups)
    if agg.empty:
        SUPPLEMENTAL_COVERAGE["lineup_units"] = "empty"
        return out

    SUPPLEMENTAL_COVERAGE["lineup_units"] = "joined"
    return out.merge(agg, on=team_col, how="left")


def enrich_player_frame(players: pd.DataFrame, season: str) -> pd.DataFrame:
    """Apply tactical + on/off joins (no-op when caches/tables missing)."""
    SUPPLEMENTAL_COVERAGE.clear()
    out = join_tactical_into_players(players, season)
    return join_onoff_into_players(out, season)


def enrich_team_frame(teams: pd.DataFrame, season: str) -> pd.DataFrame:
    """Apply lineup weakness joins (no-op when interim missing)."""
    return join_lineup_into_teams(teams, season)


def supplemental_coverage_report() -> dict[str, str]:
    """Return last join coverage flags (endpoint → joined|missing|empty)."""
    return dict(SUPPLEMENTAL_COVERAGE)
