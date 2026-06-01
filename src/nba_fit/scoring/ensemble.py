"""Option D calibrated ensemble over profile, role, need, impact, replacement, and risk."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from nba_fit.features.player_vector import FEATURE_GROUP_AVAILABILITY
from nba_fit.features.vectors import PlayerVector
from nba_fit.scoring.constants import (
    AVAILABILITY_RISK_GP_WEIGHT,
    AVAILABILITY_RISK_LOW_MIN_WEIGHT,
    AVAILABILITY_RISK_SIGNAL_WEIGHT,
    ENSEMBLE_COMPONENT_NAMES,
    ENSEMBLE_COMPONENT_WEIGHTS,
    ENSEMBLE_DERIVED_NAMES,
    PROFILE_SUBMETRIC_WEIGHTS,
    RISK_DAMPENER_MAX,
    UNCERTAINTY_PERCENTILE_BASE,
    UNCERTAINTY_PERCENTILE_SPREAD_SCALE,
)
from nba_fit.scoring.submetrics import weighted_raw_score


def profile_fit_from_submetrics(submetrics: Mapping[str, float]) -> float:
    """Weighted Option A profile block (off/def/usage/shot/spacing)."""
    return weighted_raw_score(submetrics, PROFILE_SUBMETRIC_WEIGHTS)


def role_fit_from_submetrics(submetrics: Mapping[str, float]) -> float:
    """Role alignment submetric (cosine vs roster role demand)."""
    return float(submetrics.get("role_alignment", 0.5))


def availability_risk_penalty(player: PlayerVector | None) -> float:
    """
    Availability risk in [0, 1]; higher means more penalty.

    Uses scaled availability features (GP rate, low-minutes flag) when present.
    """
    if player is None:
        return 0.0
    avail = player.select_groups(FEATURE_GROUP_AVAILABILITY)
    if len(avail) == 0:
        return 0.0
    # z-scored: gp, min_pg, games_played_rate, low_minutes_flag, availability_signal
    gp_rate = float(avail[2]) if len(avail) > 2 else 0.0
    low_min = float(avail[3]) if len(avail) > 3 else 0.0
    signal = float(avail[4]) if len(avail) > 4 else 0.0
    # Low GP rate and low_minutes_flag increase risk; availability_signal decreases it.
    risk = (
        AVAILABILITY_RISK_GP_WEIGHT * _sigmoid01(-gp_rate)
        + AVAILABILITY_RISK_LOW_MIN_WEIGHT * low_min
        + AVAILABILITY_RISK_SIGNAL_WEIGHT * (1.0 - signal)
    )
    return float(np.clip(risk, 0.0, 1.0))


def _sigmoid01(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def extract_ensemble_components(
    submetrics: Mapping[str, float],
    *,
    player: PlayerVector | None = None,
) -> dict[str, float]:
    """Map submetrics to top-level ensemble inputs plus ``risk_penalty``."""
    return {
        "profile_fit": profile_fit_from_submetrics(submetrics),
        "role_fit": role_fit_from_submetrics(submetrics),
        "team_need_fit": float(submetrics.get("team_need_fit", 0.5)),
        "projected_impact": float(submetrics.get("lineup_impact_fit", 0.5)),
        "replacement_upgrade": float(submetrics.get("replacement_upgrade", 0.5)),
        "risk_penalty": availability_risk_penalty(player),
    }


def apply_risk_dampener(raw_score: float, risk_penalty: float) -> float:
    """Scale raw ensemble down when availability risk is high."""
    dampener = 1.0 - RISK_DAMPENER_MAX * float(np.clip(risk_penalty, 0.0, 1.0))
    return raw_score * dampener


def raw_ensemble_score(components: Mapping[str, float]) -> float:
    """Weighted sum of positive components (excludes risk_penalty)."""
    total = 0.0
    for name, weight in ENSEMBLE_COMPONENT_WEIGHTS.items():
        total += weight * float(components.get(name, 0.5))
    risk = float(components.get("risk_penalty", 0.0))
    return apply_risk_dampener(total, risk)


def component_contributions(components: Mapping[str, float]) -> dict[str, float]:
    """Weighted contribution of each positive component to the pre-risk raw score."""
    out: dict[str, float] = {}
    for name, weight in ENSEMBLE_COMPONENT_WEIGHTS.items():
        out[name] = weight * float(components.get(name, 0.5))
    return out


def uncertainty_half_width(components: Mapping[str, float]) -> float:
    """
    Percentile half-width from disagreement across ensemble components.

    Higher spread among profile/role/need/impact/replacement → wider band.
    """
    vals = [float(components.get(n, 0.5)) for n in ENSEMBLE_COMPONENT_NAMES]
    spread = float(np.std(vals)) if len(vals) > 1 else 0.0
    return UNCERTAINTY_PERCENTILE_BASE + UNCERTAINTY_PERCENTILE_SPREAD_SCALE * spread


def uncertainty_band(
    percentile: float,
    components: Mapping[str, float],
) -> tuple[float, float]:
    """Return (low, high) percentile bounds clamped to [0, 100]."""
    half = uncertainty_half_width(components)
    low = max(0.0, percentile - half)
    high = min(100.0, percentile + half)
    return low, high


def calibrated_ensemble(
    profile_fit: float,
    role_fit: float,
    team_need_fit: float,
    projected_impact: float,
    replacement_upgrade: float,
    risk_penalty: float,
) -> tuple[float, dict[str, float]]:
    """
    Combine ensemble components into a single raw score in [0, 1].

    Returns ``(raw_score, components_dict)`` for calibration and fit cards.
    """
    components = {
        "profile_fit": profile_fit,
        "role_fit": role_fit,
        "team_need_fit": team_need_fit,
        "projected_impact": projected_impact,
        "replacement_upgrade": replacement_upgrade,
        "risk_penalty": risk_penalty,
    }
    return raw_ensemble_score(components), components
