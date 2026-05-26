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

SUBMETRIC_WEIGHTS: dict[str, float] = {
    # Complementarity: offensive production vs team offensive weaknesses.
    "offensive_fit": 0.18,
    # Complementarity: defensive profile vs team defensive gaps.
    "defensive_fit": 0.16,
    # Similarity: role/usage archetype vs roster role demand (cosine).
    "role_alignment": 0.14,
    # Complementarity: usage headroom — avoid crowding on-ball creators.
    "usage_compatibility": 0.14,
    # Complementarity: shot diet vs team shot vacancies.
    "shot_profile_fit": 0.12,
    # Complementarity: spacing & gravity vs spacer surplus / three volume needs.
    "spacing_gravity_fit": 0.12,
    # Replacement value: upgrade over likely displaced minutes.
    "replacement_upgrade": 0.14,
}

assert abs(sum(SUBMETRIC_WEIGHTS.values()) - 1.0) < 1e-9

SUBMETRIC_NAMES: tuple[str, ...] = tuple(SUBMETRIC_WEIGHTS.keys())
