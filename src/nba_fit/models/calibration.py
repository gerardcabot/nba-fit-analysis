"""Map raw ensemble scores to calibrated fit percentiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

CalibrationMethod = Literal["isotonic", "percentile"]


def _average_rank_percentiles(values: np.ndarray) -> np.ndarray:
    """Average-rank percentiles in [0, 100] (handles ties)."""
    n = len(values)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([50.0])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    return 100.0 * (ranks - 0.5) / n


@dataclass
class FitCalibrator:
    """
    Calibrate raw model scores to outcome-aligned percentiles.

    * **isotonic** — monotonic regression of raw score → realized outcome
      (movement backtest labels or held-out performance).
    * **percentile** — rank raw scores within the calibration set only
      (no outcome labels required).
    """

    method: CalibrationMethod = "isotonic"
    _isotonic: IsotonicRegression | None = field(default=None, repr=False)
    _raw_anchor: np.ndarray | None = field(default=None, repr=False)
    _outcome_anchor: np.ndarray | None = field(default=None, repr=False)
    is_fitted_: bool = False

    def fit(
        self,
        raw_scores: np.ndarray | pd.Series,
        outcomes: np.ndarray | pd.Series | None = None,
    ) -> FitCalibrator:
        raw = np.asarray(raw_scores, dtype=float)
        mask = np.isfinite(raw)
        raw = raw[mask]
        if len(raw) < 2:
            self.is_fitted_ = False
            return self

        if self.method == "percentile" or outcomes is None:
            self._isotonic = None
            self._raw_anchor = raw
            self._outcome_anchor = _average_rank_percentiles(raw)
            self.is_fitted_ = True
            return self

        y = np.asarray(outcomes, dtype=float)[mask]
        if len(y) != len(raw):
            raise ValueError("raw_scores and outcomes length mismatch after NaN drop")

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=100.0)
        iso.fit(raw, y)
        self._isotonic = iso
        self._raw_anchor = raw
        self._outcome_anchor = y
        self.is_fitted_ = True
        return self

    def transform(self, raw_scores: np.ndarray | pd.Series) -> np.ndarray:
        """Return calibrated percentiles in [0, 100]."""
        raw = np.asarray(raw_scores, dtype=float)
        if not self.is_fitted_ or len(raw) == 0:
            return _average_rank_percentiles(raw)

        if self.method == "percentile" or self._isotonic is None:
            if self._raw_anchor is not None and len(self._raw_anchor) >= 2:
                anchor = self._outcome_anchor
                if anchor is None:
                    anchor = _average_rank_percentiles(self._raw_anchor)
                return np.interp(
                    raw,
                    np.sort(self._raw_anchor),
                    np.sort(anchor),
                )
            return _average_rank_percentiles(raw)

        return np.clip(self._isotonic.predict(raw), 0.0, 100.0)

    def fit_transform(
        self,
        raw_scores: np.ndarray | pd.Series,
        outcomes: np.ndarray | pd.Series | None = None,
    ) -> np.ndarray:
        return self.fit(raw_scores, outcomes).transform(raw_scores)


def calibrate_fit_table(
    df: pd.DataFrame,
    *,
    raw_col: str = "raw_fit_score",
    outcome_col: str | None = "post_move_outcome",
    method: CalibrationMethod = "isotonic",
    output_col: str = "calibrated_fit_percentile",
) -> pd.DataFrame:
    """Add *output_col* to a movement backtest or fit-index frame."""
    if raw_col not in df.columns:
        raise KeyError(raw_col)
    calibrator = FitCalibrator(method=method)
    outcomes = df[outcome_col].to_numpy() if outcome_col and outcome_col in df.columns else None
    out = df.copy()
    out[output_col] = calibrator.fit_transform(df[raw_col].to_numpy(), outcomes)
    return out
