"""Interpretable player feature vectors from merged ``leaguedash*`` tables."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from nba_fit.features._column_utils import (
    find_column,
    safe_divide,
    series_or_nan,
    total_zone_fga,
    with_prefix,
    zone_fga_series,
)
from nba_fit.features.constants import (
    COL_MINUTES_STABLE,
    COL_PLAYER_ID,
    COL_ROLE_BUCKET,
    HEIGHT_INCHES_BIG_MIN,
    HEIGHT_INCHES_GUARD_MAX,
    LEAGUE_AVG_CORNER3_SHARE_OF_FG3A,
    LEAGUE_AVG_FG3A_RATE,
    LEAGUE_AVG_RIM_ATTEMPT_SHARE,
    MIN_GAMES_PLAYED_AVAILABILITY,
    MIN_MINUTES_FEATURE_FLAG,
    MIN_SEASON_MINUTES_STABLE_RATES,
    PLAYER_FEATURE_PREFIX,
    ROLE_BUCKET_BIG,
    ROLE_BUCKET_GUARD,
    ROLE_BUCKET_UNKNOWN,
    ROLE_BUCKET_WING,
    SHOT_LOCATION_ZONES,
    TURNOVER_DENOM_FLOOR,
    USG_PCT_CREATOR_THRESHOLD,
    USG_PCT_ROLE_PLAYER_THRESHOLD,
    ZONE_ABOVE_BREAK_3,
    ZONE_LEFT_CORNER_3,
    ZONE_MID_RANGE,
    ZONE_PAINT_NON_RA,
    ZONE_RESTRICTED,
    ZONE_RIGHT_CORNER_3,
)

# Feature groups returned by build_player_features (column stem after pf_ prefix).
FEATURE_GROUP_ROLE_VOLUME: str = "role_volume"
FEATURE_GROUP_SCORING_STYLE: str = "scoring_style"
FEATURE_GROUP_EFFICIENCY: str = "efficiency"
FEATURE_GROUP_CREATION: str = "creation"
FEATURE_GROUP_DEFENSE: str = "defense"
FEATURE_GROUP_REBOUNDING: str = "rebounding"
FEATURE_GROUP_AVAILABILITY: str = "availability"
FEATURE_GROUP_PHYSICAL: str = "physical"

PLAYER_FEATURE_GROUPS: tuple[str, ...] = (
    FEATURE_GROUP_ROLE_VOLUME,
    FEATURE_GROUP_SCORING_STYLE,
    FEATURE_GROUP_EFFICIENCY,
    FEATURE_GROUP_CREATION,
    FEATURE_GROUP_DEFENSE,
    FEATURE_GROUP_REBOUNDING,
    FEATURE_GROUP_AVAILABILITY,
    FEATURE_GROUP_PHYSICAL,
)


def infer_role_bucket(df: pd.DataFrame) -> pd.Series:
    """Map each row to G / W / B / UNK using position, height, or usage heuristics."""
    pos_col = find_column(df, "PLAYER_POSITION", "POSITION")
    if pos_col is not None:
        pos = df[pos_col].astype(str).str.upper().str.strip()
        out = pd.Series(ROLE_BUCKET_UNKNOWN, index=df.index, dtype=object)
        out[pos.str.contains("GUARD", na=False) | pos.isin(["G", "PG", "SG"])] = ROLE_BUCKET_GUARD
        out[pos.str.contains("FORWARD", na=False) | pos.isin(["F", "SF", "PF", "F-G", "G-F"])] = (
            ROLE_BUCKET_WING
        )
        out[pos.str.contains("CENTER", na=False) | pos.isin(["C", "F-C", "C-F"])] = ROLE_BUCKET_BIG
        known = out != ROLE_BUCKET_UNKNOWN
        if known.any():
            return out

    height = series_or_nan(df, "PLAYER_HEIGHT_INCHES", "HEIGHT_INCHES")
    usg = series_or_nan(df, "USG_PCT")
    out = pd.Series(ROLE_BUCKET_UNKNOWN, index=df.index, dtype=object)
    out[height <= HEIGHT_INCHES_GUARD_MAX] = ROLE_BUCKET_GUARD
    out[height >= HEIGHT_INCHES_BIG_MIN] = ROLE_BUCKET_BIG
    mid = (height > HEIGHT_INCHES_GUARD_MAX) & (height < HEIGHT_INCHES_BIG_MIN)
    out[mid] = ROLE_BUCKET_WING
    out[(out == ROLE_BUCKET_UNKNOWN) & (usg >= USG_PCT_CREATOR_THRESHOLD)] = ROLE_BUCKET_GUARD
    out[(out == ROLE_BUCKET_UNKNOWN) & (usg < USG_PCT_ROLE_PLAYER_THRESHOLD)] = ROLE_BUCKET_BIG
    out[out == ROLE_BUCKET_UNKNOWN] = ROLE_BUCKET_WING
    return out


def _pf(name: str) -> str:
    return f"{PLAYER_FEATURE_PREFIX}{name}"


def build_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build interpretable player feature matrix from a merged leaguedash-wide table.

    Parameters
    ----------
    df:
        One row per player. Columns may come from any combination of
        ``leaguedashplayerstats`` (Base/Advanced/Usage), ``leaguedashplayerbiostats``,
        ``leaguedashplayershotlocations``, ``leaguedashplayerptshot``, and
        ``leaguedashptdefend`` left-joined on ``PLAYER_ID`` / ``CLOSE_DEF_PERSON_ID``.

    Returns
    -------
    DataFrame with ``PLAYER_ID``, ``role_bucket``, stability flag, and ``pf_*`` features.
    See ``MISSING_PROXIES.md`` for capabilities not available from leaguedash alone.
    """
    if df.empty:
        return pd.DataFrame(columns=[COL_PLAYER_ID, COL_ROLE_BUCKET, COL_MINUTES_STABLE])

    work = df.copy()
    player_col = find_column(work, COL_PLAYER_ID, "CLOSE_DEF_PERSON_ID")
    if player_col is None:
        raise ValueError("Input DataFrame must include PLAYER_ID or CLOSE_DEF_PERSON_ID")

    out = pd.DataFrame(index=work.index)
    out[COL_PLAYER_ID] = work[player_col]
    out[COL_ROLE_BUCKET] = infer_role_bucket(work)

    min_pg = series_or_nan(work, "MIN")
    gp = series_or_nan(work, "GP", "G")
    total_min = min_pg * gp
    out[COL_MINUTES_STABLE] = total_min >= MIN_SEASON_MINUTES_STABLE_RATES

    # --- role / volume (leaguedashplayerstats Base + Advanced) ---
    fga = series_or_nan(work, "FGA")
    pts = series_or_nan(work, "PTS")
    ast = series_or_nan(work, "AST")
    tov = series_or_nan(work, "TOV")
    usg = series_or_nan(work, "USG_PCT")
    pace = series_or_nan(work, "PACE")

    role_volume = {
        "min_pg": min_pg,
        "gp": gp,
        "total_min": total_min,
        "usg_pct": usg,
        "pace": pace,
        "fga_pg": fga,
        "pts_pg": pts,
        "ast_pg": ast,
        "tov_pg": tov,
        "ast_to": safe_divide(ast, tov.clip(lower=TURNOVER_DENOM_FLOOR)),
        "ast_ratio": series_or_nan(work, "AST_RATIO"),
        "pie": series_or_nan(work, "PIE"),
    }

    # --- scoring style (shot locations + leaguedashplayerptshot) ---
    total_fga_zones = total_zone_fga(work)
    rim = zone_fga_series(work, ZONE_RESTRICTED) + zone_fga_series(work, ZONE_PAINT_NON_RA)
    corner = zone_fga_series(work, ZONE_LEFT_CORNER_3) + zone_fga_series(work, ZONE_RIGHT_CORNER_3)
    fg3a = series_or_nan(work, "FG3A")
    fg3a_freq = series_or_nan(work, "FG3A_FREQUENCY")

    scoring_style = {
        "rim_freq": safe_divide(rim, total_fga_zones.replace(0, np.nan)).fillna(
            safe_divide(rim, fga.replace(0, np.nan))
        ),
        "mid_freq": safe_divide(zone_fga_series(work, ZONE_MID_RANGE), total_fga_zones.replace(0, np.nan)),
        "corner3_freq": safe_divide(corner, total_fga_zones.replace(0, np.nan)),
        "ab3_freq": safe_divide(zone_fga_series(work, ZONE_ABOVE_BREAK_3), total_fga_zones.replace(0, np.nan)),
        "fg3a_rate": safe_divide(fg3a, fga.replace(0, np.nan)),
        "fg3a_frequency": fg3a_freq,
        "fg2a_frequency": series_or_nan(work, "FG2A_FREQUENCY"),
        "corner3_share_of_fg3a": safe_divide(corner, fg3a.replace(0, np.nan)),
        "rim_freq_vs_league": safe_divide(rim, total_fga_zones.replace(0, np.nan)) - LEAGUE_AVG_RIM_ATTEMPT_SHARE,
        "fg3a_rate_vs_league": safe_divide(fg3a, fga.replace(0, np.nan)) - LEAGUE_AVG_FG3A_RATE,
        "corner3_share_vs_league": safe_divide(corner, fg3a.replace(0, np.nan))
        - LEAGUE_AVG_CORNER3_SHARE_OF_FG3A,
    }

    # --- efficiency ---
    efficiency = {
        "ts_pct": series_or_nan(work, "TS_PCT"),
        "efg_pct": series_or_nan(work, "EFG_PCT"),
        "fg_pct": series_or_nan(work, "FG_PCT"),
        "fg3_pct": series_or_nan(work, "FG3_PCT"),
        "ft_pct": series_or_nan(work, "FT_PCT"),
        "off_rating": series_or_nan(work, "OFF_RATING", "E_OFF_RATING"),
        "def_rating": series_or_nan(work, "DEF_RATING", "E_DEF_RATING"),
        "net_rating": series_or_nan(work, "NET_RATING", "E_NET_RATING"),
        "plus_minus_pg": series_or_nan(work, "PLUS_MINUS"),
    }

    # --- creation ---
    creation = {
        "ast_pct": series_or_nan(work, "AST_PCT"),
        "ast_to": role_volume["ast_to"],
        "ast_ratio": role_volume["ast_ratio"],
        "tov_pct": series_or_nan(work, "E_TOV_PCT", "TM_TOV_PCT"),
    }

    # --- defense proxies (box + leaguedashptdefend) ---
    defense = {
        "stl_pg": series_or_nan(work, "STL"),
        "blk_pg": series_or_nan(work, "BLK"),
        "def_rating": efficiency["def_rating"],
        "defend_freq": series_or_nan(work, "FREQ"),
        "defend_fg_pct": series_or_nan(work, "D_FG_PCT"),
        "defend_fg_pct_vs_normal": series_or_nan(work, "PCT_PLUSMINUS"),
        "foul_rate": safe_divide(series_or_nan(work, "PF"), min_pg.replace(0, np.nan)),
    }

    # --- rebounding ---
    rebounding = {
        "oreb_pg": series_or_nan(work, "OREB"),
        "dreb_pg": series_or_nan(work, "DREB"),
        "reb_pg": series_or_nan(work, "REB"),
        "oreb_pct": series_or_nan(work, "OREB_PCT"),
        "dreb_pct": series_or_nan(work, "DREB_PCT"),
        "reb_pct": series_or_nan(work, "REB_PCT"),
    }

    # --- availability (leaguedash GP/MIN only — no injury feed) ---
    availability = {
        "gp": gp,
        "min_pg": min_pg,
        "games_played_rate": safe_divide(gp, 82.0),
        "low_minutes_flag": (total_min < MIN_MINUTES_FEATURE_FLAG).astype(float),
        "availability_signal": (gp >= MIN_GAMES_PLAYED_AVAILABILITY).astype(float),
    }

    # --- physical (biostats) ---
    physical = {
        "age": series_or_nan(work, "AGE"),
        "height_in": series_or_nan(work, "PLAYER_HEIGHT_INCHES"),
        "weight_lb": series_or_nan(work, "PLAYER_WEIGHT"),
        "draft_year": series_or_nan(work, "DRAFT_YEAR"),
    }

    groups: dict[str, dict[str, pd.Series]] = {
        FEATURE_GROUP_ROLE_VOLUME: role_volume,
        FEATURE_GROUP_SCORING_STYLE: scoring_style,
        FEATURE_GROUP_EFFICIENCY: efficiency,
        FEATURE_GROUP_CREATION: creation,
        FEATURE_GROUP_DEFENSE: defense,
        FEATURE_GROUP_REBOUNDING: rebounding,
        FEATURE_GROUP_AVAILABILITY: availability,
        FEATURE_GROUP_PHYSICAL: physical,
    }

    for group, feats in groups.items():
        for stem, series in feats.items():
            out[_pf(f"{group}__{stem}")] = pd.to_numeric(series, errors="coerce")

    return out


def player_feature_columns(groups: Iterable[str] | None = None) -> list[str]:
    """Return feature column names for the requested groups (materialized from a stub row)."""
    stub = pd.DataFrame(
        {
            COL_PLAYER_ID: [0],
            "MIN": [30.0],
            "GP": [50],
            "FGA": [15.0],
            "PTS": [20.0],
            "AST": [5.0],
            "TOV": [2.0],
            "USG_PCT": [0.25],
            "PACE": [100.0],
            "FG3A": [6.0],
            "PLAYER_HEIGHT_INCHES": [78.0],
            "PLAYER_WEIGHT": [210.0],
            "AGE": [25.0],
        }
    )
    built = build_player_features(stub)
    cols = [c for c in built.columns if c.startswith(PLAYER_FEATURE_PREFIX)]
    if groups is None:
        return cols
    allowed = {f"{PLAYER_FEATURE_PREFIX}{g}__" for g in groups}
    return [c for c in cols if any(c.startswith(prefix) for prefix in allowed)]

