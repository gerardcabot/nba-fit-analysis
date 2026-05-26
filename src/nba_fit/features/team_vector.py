"""Interpretable team need/style vectors from merged ``leaguedash*`` tables."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from nba_fit.features._column_utils import (
    find_column,
    safe_divide,
    series_or_nan,
    total_zone_fga,
    zone_fga_series,
)
from nba_fit.features.constants import (
    COL_TEAM_ID,
    LEAGUE_AVG_CORNER3_SHARE_OF_FG3A,
    LEAGUE_AVG_FG3A_RATE,
    LEAGUE_AVG_PACE,
    LEAGUE_AVG_RIM_ATTEMPT_SHARE,
    TEAM_FEATURE_PREFIX,
    ZONE_ABOVE_BREAK_3,
    ZONE_LEFT_CORNER_3,
    ZONE_MID_RANGE,
    ZONE_PAINT_NON_RA,
    ZONE_RESTRICTED,
    ZONE_RIGHT_CORNER_3,
)

FEATURE_GROUP_PACE_IDENTITY: str = "pace_identity"
FEATURE_GROUP_SHOT_DIET: str = "shot_diet"
FEATURE_GROUP_OFF_IDENTITY: str = "off_identity"
FEATURE_GROUP_DEF_IDENTITY: str = "def_identity"
FEATURE_GROUP_WEAKNESSES: str = "weaknesses"
FEATURE_GROUP_ROSTER_COMP: str = "roster_composition"

TEAM_FEATURE_GROUPS: tuple[str, ...] = (
    FEATURE_GROUP_PACE_IDENTITY,
    FEATURE_GROUP_SHOT_DIET,
    FEATURE_GROUP_OFF_IDENTITY,
    FEATURE_GROUP_DEF_IDENTITY,
    FEATURE_GROUP_WEAKNESSES,
    FEATURE_GROUP_ROSTER_COMP,
)


def _tf(name: str) -> str:
    return f"{TEAM_FEATURE_PREFIX}{name}"


def aggregate_roster_composition(player_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level leaguedash rows to team roster-composition proxies.

    Expects ``TEAM_ID`` plus typical box/usage columns from ``leaguedashplayerstats``
    and/or ``leaguedashplayerbiostats``.
    """
    team_col = find_column(player_df, COL_TEAM_ID)
    if team_col is None:
        raise ValueError("player_df must include TEAM_ID for roster aggregation")

    work = player_df.copy()
    work["_usg"] = series_or_nan(work, "USG_PCT")
    work["_min_pg"] = series_or_nan(work, "MIN")
    work["_gp"] = series_or_nan(work, "GP", "G")
    work["_total_min"] = work["_min_pg"] * work["_gp"]

    agg = (
        work.groupby(team_col, dropna=False)
        .agg(
            roster_n_players=(team_col, "count"),
            roster_mean_usg_pct=("_usg", "mean"),
            roster_std_usg_pct=("_usg", "std"),
            roster_top_usg_pct=("_usg", "max"),
            roster_mean_min_pg=("_min_pg", "mean"),
            roster_total_min=("_total_min", "sum"),
        )
        .reset_index()
    )
    agg["roster_usg_concentration"] = agg["roster_top_usg_pct"] - agg["roster_mean_usg_pct"]
    agg = agg.rename(columns={team_col: COL_TEAM_ID})
    return agg


def build_team_features(
    df: pd.DataFrame,
    *,
    roster_agg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build team need/style feature matrix from leaguedash team tables.

    Parameters
    ----------
    df:
        One row per team from ``leaguedashteamstats`` (Base/Advanced), optional joins
        from ``leaguedashteamshotlocations``, ``leaguedashteamptshot``,
        ``leaguedashptteamdefend``.
    roster_agg:
        Optional output of :func:`aggregate_roster_composition` for composition features.
    """
    if df.empty:
        return pd.DataFrame(columns=[COL_TEAM_ID])

    work = df.copy()
    team_col = find_column(work, COL_TEAM_ID)
    if team_col is None:
        raise ValueError("Input DataFrame must include TEAM_ID")

    out = pd.DataFrame(index=work.index)
    out[COL_TEAM_ID] = work[team_col]

    pace = series_or_nan(work, "PACE", "E_PACE")
    off_rtg = series_or_nan(work, "OFF_RATING", "E_OFF_RATING")
    def_rtg = series_or_nan(work, "DEF_RATING", "E_DEF_RATING")
    net_rtg = series_or_nan(work, "NET_RATING", "E_NET_RATING")
    fga = series_or_nan(work, "FGA")
    fg3a = series_or_nan(work, "FG3A")
    ast = series_or_nan(work, "AST")
    tov = series_or_nan(work, "TOV")

    total_fga_zones = total_zone_fga(work)
    rim = zone_fga_series(work, ZONE_RESTRICTED) + zone_fga_series(work, ZONE_PAINT_NON_RA)
    corner = zone_fga_series(work, ZONE_LEFT_CORNER_3) + zone_fga_series(work, ZONE_RIGHT_CORNER_3)

    pace_identity = {
        "pace": pace,
        "pace_vs_league": pace - LEAGUE_AVG_PACE,
        "poss_pg": series_or_nan(work, "POSS"),
    }

    shot_diet = {
        "rim_freq": safe_divide(rim, total_fga_zones.replace(0, np.nan)),
        "mid_freq": safe_divide(zone_fga_series(work, ZONE_MID_RANGE), total_fga_zones.replace(0, np.nan)),
        "corner3_freq": safe_divide(corner, total_fga_zones.replace(0, np.nan)),
        "ab3_freq": safe_divide(zone_fga_series(work, ZONE_ABOVE_BREAK_3), total_fga_zones.replace(0, np.nan)),
        "fg3a_rate": safe_divide(fg3a, fga.replace(0, np.nan)),
        "fg3a_frequency": series_or_nan(work, "FG3A_FREQUENCY"),
        "efg_pct": series_or_nan(work, "EFG_PCT"),
        "ts_pct": series_or_nan(work, "TS_PCT"),
    }

    off_identity = {
        "off_rating": off_rtg,
        "net_rating": net_rtg,
        "ast_pg": ast,
        "tov_pg": tov,
        "ast_ratio": series_or_nan(work, "AST_RATIO"),
        "ast_pct": series_or_nan(work, "AST_PCT"),
        "tm_tov_pct": series_or_nan(work, "TM_TOV_PCT"),
        "pie": series_or_nan(work, "PIE"),
    }

    def_identity = {
        "def_rating": def_rtg,
        "stl_pg": series_or_nan(work, "STL"),
        "blk_pg": series_or_nan(work, "BLK"),
        "reb_pg": series_or_nan(work, "REB"),
        "oreb_pct": series_or_nan(work, "OREB_PCT"),
        "dreb_pct": series_or_nan(work, "DREB_PCT"),
        "defend_freq": series_or_nan(work, "FREQ"),
        "defend_fg_pct_vs_normal": series_or_nan(work, "PCT_PLUSMINUS"),
    }

    # Weakness proxies: higher is worse for defense/turnovers, lower is worse for offense
    weaknesses = {
        "def_weakness": def_rtg,
        "off_weakness": -off_rtg,
        "tov_weakness": series_or_nan(work, "TM_TOV_PCT"),
        "rim_def_freq_gap": series_or_nan(work, "FREQ"),
    }

    roster_composition: dict[str, pd.Series] = {}
    if roster_agg is not None and not roster_agg.empty:
        merged = out[[COL_TEAM_ID]].merge(roster_agg, on=COL_TEAM_ID, how="left")
        for col in merged.columns:
            if col.startswith("roster_"):
                roster_composition[col.removeprefix("roster_")] = pd.Series(
                    merged[col].to_numpy(), index=work.index
                )
    else:
        for stem in (
            "n_players",
            "mean_usg_pct",
            "std_usg_pct",
            "top_usg_pct",
            "usg_concentration",
            "mean_min_pg",
        ):
            roster_composition[stem] = pd.Series(np.nan, index=work.index, dtype=float)

    groups: dict[str, dict[str, pd.Series]] = {
        FEATURE_GROUP_PACE_IDENTITY: pace_identity,
        FEATURE_GROUP_SHOT_DIET: shot_diet,
        FEATURE_GROUP_OFF_IDENTITY: off_identity,
        FEATURE_GROUP_DEF_IDENTITY: def_identity,
        FEATURE_GROUP_WEAKNESSES: weaknesses,
        FEATURE_GROUP_ROSTER_COMP: roster_composition,
    }

    for group, feats in groups.items():
        for stem, series in feats.items():
            out[_tf(f"{group}__{stem}")] = pd.to_numeric(series, errors="coerce")

    return out


def team_feature_columns(groups: Iterable[str] | None = None) -> list[str]:
    stub = pd.DataFrame(
        {
            COL_TEAM_ID: [1610612747],
            "PACE": [101.0],
            "OFF_RATING": [115.0],
            "DEF_RATING": [112.0],
            "FGA": [88.0],
            "FG3A": [38.0],
            "AST": [26.0],
            "TOV": [13.0],
        }
    )
    built = build_team_features(stub)
    cols = [c for c in built.columns if c.startswith(TEAM_FEATURE_PREFIX)]
    if groups is None:
        return cols
    allowed = set(groups)
    return [c for c in cols if c.split("__", 1)[0].removeprefix(TEAM_FEATURE_PREFIX) in allowed]
