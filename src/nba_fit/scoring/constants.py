"""Scoring constants: submetric weights and feature-group mapping."""

from __future__ import annotations

from nba_fit.features.player_vector import (
    FEATURE_GROUP_CREATION,
    FEATURE_GROUP_DEFENSE,
    FEATURE_GROUP_EFFICIENCY,
    FEATURE_GROUP_REBOUNDING,
    FEATURE_GROUP_ROLE_VOLUME,
    FEATURE_GROUP_SCORING_STYLE,
)
from nba_fit.features.team_vector import (
    FEATURE_GROUP_DEF_IDENTITY,
    FEATURE_GROUP_OFF_IDENTITY,
    FEATURE_GROUP_ROSTER_COMP,
    FEATURE_GROUP_SHOT_DIET,
    FEATURE_GROUP_WEAKNESSES,
)
from nba_fit.models.constants import LINEUP_IMPACT_WEIGHT, ROLE_FIT_WEIGHT

# Player groups used per submetric (similarity vs complementarity in comments)
PLAYER_OFF_GROUPS: tuple[str, ...] = (
    FEATURE_GROUP_SCORING_STYLE,
    FEATURE_GROUP_EFFICIENCY,
    FEATURE_GROUP_CREATION,
)
PLAYER_DEF_GROUPS: tuple[str, ...] = (FEATURE_GROUP_DEFENSE, FEATURE_GROUP_REBOUNDING)
PLAYER_ROLE_GROUPS: tuple[str, ...] = (FEATURE_GROUP_ROLE_VOLUME,)
PLAYER_USAGE_GROUPS: tuple[str, ...] = (FEATURE_GROUP_ROLE_VOLUME,)
PLAYER_SHOT_GROUPS: tuple[str, ...] = (FEATURE_GROUP_SCORING_STYLE,)
PLAYER_SPACING_GROUPS: tuple[str, ...] = (FEATURE_GROUP_SCORING_STYLE,)

TEAM_OFF_NEED_GROUPS: tuple[str, ...] = (
    FEATURE_GROUP_WEAKNESSES,
    FEATURE_GROUP_OFF_IDENTITY,
)
TEAM_DEF_NEED_GROUPS: tuple[str, ...] = (
    FEATURE_GROUP_WEAKNESSES,
    FEATURE_GROUP_DEF_IDENTITY,
)
TEAM_USAGE_GROUPS: tuple[str, ...] = (FEATURE_GROUP_ROSTER_COMP,)
TEAM_SHOT_NEED_GROUPS: tuple[str, ...] = (FEATURE_GROUP_SHOT_DIET,)
TEAM_ROLE_NEED_GROUPS: tuple[str, ...] = (FEATURE_GROUP_ROSTER_COMP,)

# ---------------------------------------------------------------------------
# Submetric combination weights (sum to 1.0) — basketball rationale in comments
# ---------------------------------------------------------------------------
# Option A weights scaled by (1 - ROLE_FIT_WEIGHT - LINEUP_IMPACT_WEIGHT) when
# adding Option B ``team_need_fit`` and Option C ``lineup_impact_fit``.

_OPTION_A_SCALE = 1.0 - ROLE_FIT_WEIGHT - LINEUP_IMPACT_WEIGHT

SUBMETRIC_WEIGHTS: dict[str, float] = {
    # Complementarity: offensive production vs team offensive weaknesses.
    "offensive_fit": 0.18 * _OPTION_A_SCALE,
    # Complementarity: defensive profile vs team defensive gaps.
    "defensive_fit": 0.16 * _OPTION_A_SCALE,
    # Similarity: role/usage archetype vs roster role demand (cosine).
    "role_alignment": 0.14 * _OPTION_A_SCALE,
    # Complementarity: usage headroom — avoid crowding on-ball creators.
    "usage_compatibility": 0.14 * _OPTION_A_SCALE,
    # Complementarity: shot diet vs team shot vacancies.
    "shot_profile_fit": 0.12 * _OPTION_A_SCALE,
    # Complementarity: spacing & gravity vs spacer surplus / three volume needs.
    "spacing_gravity_fit": 0.12 * _OPTION_A_SCALE,
    # Replacement value: upgrade over likely displaced minutes.
    "replacement_upgrade": 0.14 * _OPTION_A_SCALE,
    # Option B: learned archetype gaps + lineup weakness proxies (role_fit.py).
    "team_need_fit": ROLE_FIT_WEIGHT,
    # Option C: projected net rating delta vs bottom-rotation replacement (lineup_fit.py).
    "lineup_impact_fit": LINEUP_IMPACT_WEIGHT,
}

assert abs(sum(SUBMETRIC_WEIGHTS.values()) - 1.0) < 1e-9

SUBMETRIC_NAMES: tuple[str, ...] = tuple(SUBMETRIC_WEIGHTS.keys())

# ---------------------------------------------------------------------------
# Option D — calibrated ensemble (profile, role, need, impact, replacement, risk)
# ---------------------------------------------------------------------------
# Top-level components map to the plan formula:
#   overall_fit = calibrated_ensemble(
#       profile_fit, role_fit, team_need_fit, projected_impact,
#       replacement_upgrade, risk_penalty,
#   )
# Weights sum to 1.0; ``risk_penalty`` is applied as a multiplicative dampener
# (see ``ensemble.apply_risk_dampener``) so high availability risk lowers the score.

ENSEMBLE_COMPONENT_NAMES: tuple[str, ...] = (
    "profile_fit",
    "role_fit",
    "team_need_fit",
    "projected_impact",
    "replacement_upgrade",
)

# Includes availability ``risk_penalty`` (applied as dampener, not a positive weight).
ENSEMBLE_DERIVED_NAMES: tuple[str, ...] = ENSEMBLE_COMPONENT_NAMES + ("risk_penalty",)

# Weight each positive component in the raw ensemble (before risk dampening).
ENSEMBLE_COMPONENT_WEIGHTS: dict[str, float] = {
    # Option A interpretable profile (off/def/usage/shot/spacing).
    "profile_fit": 0.28,
    # Cosine role alignment vs roster role demand (Option A submetric).
    "role_fit": 0.12,
    # Archetype-gap + embedding need alignment (Option B).
    "team_need_fit": 0.18,
    # Projected lineup net-rating delta (Option C).
    "projected_impact": 0.22,
    # Upgrade over displaced rotation minutes.
    "replacement_upgrade": 0.20,
}

assert abs(sum(ENSEMBLE_COMPONENT_WEIGHTS.values()) - 1.0) < 1e-9

# Within profile_fit: renormalize Option A submetrics that feed the profile block.
PROFILE_SUBMETRIC_WEIGHTS: dict[str, float] = {
    "offensive_fit": 0.22,
    "defensive_fit": 0.20,
    "usage_compatibility": 0.18,
    "shot_profile_fit": 0.20,
    "spacing_gravity_fit": 0.20,
}

assert abs(sum(PROFILE_SUBMETRIC_WEIGHTS.values()) - 1.0) < 1e-9

# Availability risk: blend weight on (1 - risk_penalty) multiplier.
# 0.15 → at most 15% downward adjustment when risk_penalty == 1.
RISK_DAMPENER_MAX: float = 0.15

# Uncertainty band half-width in percentile points: base + spread * scale.
UNCERTAINTY_PERCENTILE_BASE: float = 4.0
UNCERTAINTY_PERCENTILE_SPREAD_SCALE: float = 18.0

# ---------------------------------------------------------------------------
# Submetric sigmoid / complementarity tuning (Option A)
# ---------------------------------------------------------------------------
# Shared scale for gap-based submetrics (usage, replacement, complementarity).
SIGMOID_SCALE: float = 0.75

# Usage headroom: bonus when roster has creator surplus relative to player usage.
USAGE_CREATOR_SURPLUS_WEIGHT: float = 0.3

# Replacement upgrade: discount team gap when estimating player level vs need.
REPLACEMENT_TEAM_GAP_WEIGHT: float = 0.25

# Availability risk blend (sum to 1.0) — GP rate, low-minutes flag, signal.
AVAILABILITY_RISK_GP_WEIGHT: float = 0.35
AVAILABILITY_RISK_LOW_MIN_WEIGHT: float = 0.35
AVAILABILITY_RISK_SIGNAL_WEIGHT: float = 0.30

# Submetrics that fall back to neutral when Option B/C artifacts are missing.
DEGRADABLE_SUBMETRICS: tuple[str, ...] = ("team_need_fit", "lineup_impact_fit")
