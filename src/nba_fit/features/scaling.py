"""Robust season z-scores by position/role bucket."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_fit.features.constants import (
    COL_ROLE_BUCKET,
    IQR_TO_SIGMA,
    MAD_EPSILON,
    MIN_BUCKET_SIZE_FOR_SCALING,
    ZSCORE_CLIP_HIGH,
    ZSCORE_CLIP_LOW,
)


def robust_median_mad(series: pd.Series) -> tuple[float, float]:
    """Median and MAD (scaled to pseudo-sigma via 1.4826 factor)."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 0.0, 1.0
    median = float(clean.median())
    mad = float((clean - median).abs().median())
    if mad < MAD_EPSILON:
        iqr = float(clean.quantile(0.75) - clean.quantile(0.25))
        sigma = iqr / IQR_TO_SIGMA if iqr > MAD_EPSILON else 1.0
        return median, sigma
    return median, mad * 1.4826


def robust_zscore(
    series: pd.Series,
    *,
    clip_low: float = ZSCORE_CLIP_LOW,
    clip_high: float = ZSCORE_CLIP_HIGH,
    center: float | None = None,
    scale: float | None = None,
) -> pd.Series:
    """League-wide robust z-score with optional precomputed center/scale."""
    values = pd.to_numeric(series, errors="coerce")
    if center is None or scale is None:
        center, scale = robust_median_mad(values)
    z = (values - center) / max(scale, MAD_EPSILON)
    return z.clip(lower=clip_low, upper=clip_high)


def scale_features_by_bucket(
    features: pd.DataFrame,
    feature_cols: list[str],
    *,
    bucket_col: str = COL_ROLE_BUCKET,
    clip_low: float = ZSCORE_CLIP_LOW,
    clip_high: float = ZSCORE_CLIP_HIGH,
) -> pd.DataFrame:
    """Z-score each feature within role/position buckets; fall back to league-wide.

    Parameters
    ----------
    features:
        Matrix containing ``bucket_col`` and raw feature columns.
    feature_cols:
        Columns to scale in place (``z_`` prefix added to output copy).
    """
    out = features.copy()
    if bucket_col not in out.columns:
        out[bucket_col] = "ALL"

    for col in feature_cols:
        if col not in out.columns:
            continue
        z_col = f"z_{col}" if not col.startswith("z_") else col
        out[z_col] = np.nan
        global_center, global_scale = robust_median_mad(out[col])

        for bucket, idx in out.groupby(bucket_col, dropna=False).groups.items():
            mask = out.index.isin(idx)
            bucket_series = out.loc[mask, col]
            if len(bucket_series.dropna()) >= MIN_BUCKET_SIZE_FOR_SCALING:
                center, scale = robust_median_mad(bucket_series)
            else:
                center, scale = global_center, global_scale
            out.loc[mask, z_col] = robust_zscore(
                bucket_series,
                clip_low=clip_low,
                clip_high=clip_high,
                center=center,
                scale=scale,
            )
    return out


def scale_player_features(
    player_features: pd.DataFrame,
    *,
    bucket_col: str = COL_ROLE_BUCKET,
) -> pd.DataFrame:
    """Scale all ``pf_*`` columns in a player feature matrix."""
    from nba_fit.features.constants import PLAYER_FEATURE_PREFIX

    feat_cols = [c for c in player_features.columns if c.startswith(PLAYER_FEATURE_PREFIX)]
    return scale_features_by_bucket(player_features, feat_cols, bucket_col=bucket_col)


def scale_team_features(team_features: pd.DataFrame) -> pd.DataFrame:
    """League-wide robust z-scores for ``tf_*`` team features (no position bucket)."""
    from nba_fit.features.constants import TEAM_FEATURE_PREFIX

    feat_cols = [c for c in team_features.columns if c.startswith(TEAM_FEATURE_PREFIX)]
    out = team_features.copy()
    for col in feat_cols:
        out[f"z_{col}"] = robust_zscore(out[col])
    return out
