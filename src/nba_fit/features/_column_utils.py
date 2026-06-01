"""Column resolution helpers for merged leaguedash DataFrames."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nba_fit.features.constants import SHOT_COL_FGA_SUFFIX, SHOT_LOCATION_ZONES


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with MultiIndex columns flattened to 'Zone | FGA' style strings."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df.copy()
    flat = [
        " | ".join(str(p) for p in tup if str(p) not in ("", "nan"))
        for tup in df.columns
    ]
    out = df.copy()
    out.columns = flat
    return out


def find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    """Resolve a column by exact match, case-insensitive match, or substring."""
    cols = list(df.columns)
    upper_map = {str(c).upper(): c for c in cols}
    for name in candidates:
        if name in cols:
            return name
        hit = upper_map.get(name.upper())
        if hit is not None:
            return str(hit)
    for name in candidates:
        low = name.lower()
        for col in cols:
            if low in str(col).lower():
                return str(col)
    return None


def series_or_nan(df: pd.DataFrame, *candidates: str) -> pd.Series:
    col = find_column(df, *candidates)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def safe_divide(
    numerator: pd.Series | float,
    denominator: pd.Series | float,
    *,
    floor: float = 1e-9,
) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    if isinstance(den, pd.Series):
        den = den.clip(lower=floor)
    else:
        den = max(float(den), floor)
    return num / den


def zone_fga_column(zone: str) -> str:
    return f"{zone} | {SHOT_COL_FGA_SUFFIX}"


def zone_fga_series(df: pd.DataFrame, zone: str) -> pd.Series:
    flat = flatten_columns(df)
    col = find_column(flat, zone_fga_column(zone), f"{zone} | FGA", zone)
    if col is None:
        return pd.Series(0.0, index=df.index, dtype=float)
    return pd.to_numeric(flat[col], errors="coerce").fillna(0.0)


def total_zone_fga(df: pd.DataFrame) -> pd.Series:
    total = pd.Series(0.0, index=df.index, dtype=float)
    for zone in SHOT_LOCATION_ZONES:
        total = total + zone_fga_series(df, zone)
    return total.clip(lower=0.0)


def with_prefix(names: list[str], prefix: str) -> list[str]:
    return [f"{prefix}{n}" for n in names]
