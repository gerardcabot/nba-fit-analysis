"""Feature engineering (Option A — interpretable leaguedash vectors)."""

from nba_fit.features.constants import (
    LEAGUE_AVG_PACE,
    MIN_SEASON_MINUTES_STABLE_RATES,
    PLAYER_FEATURE_PREFIX,
    TEAM_FEATURE_PREFIX,
    ZSCORE_CLIP_HIGH,
    ZSCORE_CLIP_LOW,
)
from nba_fit.features.player_vector import (
    PLAYER_FEATURE_GROUPS,
    build_player_features,
)
from nba_fit.features.scaling import (
    robust_zscore,
    scale_features_by_bucket,
    scale_player_features,
    scale_team_features,
)
from nba_fit.features.team_vector import (
    TEAM_FEATURE_GROUPS,
    aggregate_roster_composition,
    build_team_features,
)

__all__ = [
    "LEAGUE_AVG_PACE",
    "MIN_SEASON_MINUTES_STABLE_RATES",
    "PLAYER_FEATURE_GROUPS",
    "PLAYER_FEATURE_PREFIX",
    "TEAM_FEATURE_GROUPS",
    "TEAM_FEATURE_PREFIX",
    "ZSCORE_CLIP_HIGH",
    "ZSCORE_CLIP_LOW",
    "aggregate_roster_composition",
    "build_player_features",
    "build_team_features",
    "robust_zscore",
    "scale_features_by_bucket",
    "scale_player_features",
    "scale_team_features",
]
