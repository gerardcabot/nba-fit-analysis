"""Feature-engineering thresholds and coefficients (Option A / leaguedash-only).

Every numeric knob used by player/team vectors and scaling lives here with a short
basketball or robust-statistics rationale. Domain paths/HTTP settings remain in
``nba_fit.config.settings``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pace & possession context (Dean Oliver / modern NBA tracking literature)
# ---------------------------------------------------------------------------

# League-average team pace has hovered ~98–102 possessions per 48 min since 2015;
# 100 is the standard normalization constant for per-100 rates and sanity checks.
LEAGUE_AVG_PACE: float = 100.0

# Pace values below this are usually small-sample or partial-season artifacts.
PACE_FLOOR: float = 90.0

# Pace values above this flag extreme uptempo teams (historical ceiling ~110).
PACE_CEILING: float = 115.0

# ---------------------------------------------------------------------------
# Sample size & rate stability (public analytics practice)
# ---------------------------------------------------------------------------

# ~500 minutes/season is a common floor before trusting per-minute rates for
# rotation players (Kubatko/B-R, tracking blogs); stars need more.
MIN_SEASON_MINUTES_STABLE_RATES: int = 500

# Minimum games played to treat availability signal as meaningful (82-game season).
MIN_GAMES_PLAYED_AVAILABILITY: int = 20

# Flag players below this minute total (not a hard filter — downstream can weight).
MIN_MINUTES_FEATURE_FLAG: int = 100

# Defended-shot samples below this frequency % are noisy in leaguedashptdefend.
MIN_DEFEND_FREQ_PCT: float = 0.05

# ---------------------------------------------------------------------------
# Robust z-score scaling (Huber / IQR practice in sports analytics)
# ---------------------------------------------------------------------------

# Clip standardized features to ±3 to limit outlier leverage in cosine/linear fits.
ZSCORE_CLIP_LOW: float = -3.0
ZSCORE_CLIP_HIGH: float = 3.0

# Add to MAD denominator to avoid division by zero on tiny position buckets.
MAD_EPSILON: float = 1e-6

# When MAD is near zero, fall back to IQR/1.349 as a pseudo-standard deviation.
IQR_TO_SIGMA: float = 1.349

# Minimum bucket size before bucket-specific scaling; else use league-wide.
MIN_BUCKET_SIZE_FOR_SCALING: int = 8

# ---------------------------------------------------------------------------
# Role / position buckets for scaling
# ---------------------------------------------------------------------------

# Canonical buckets aligned with NBA.com defend-position strings where possible.
ROLE_BUCKET_GUARD: str = "G"
ROLE_BUCKET_WING: str = "W"
ROLE_BUCKET_BIG: str = "B"
ROLE_BUCKET_UNKNOWN: str = "UNK"

# Usage% cutoffs (leaguedash Advanced) for heuristic role when position missing.
USG_PCT_CREATOR_THRESHOLD: float = 0.26
USG_PCT_ROLE_PLAYER_THRESHOLD: float = 0.18

# Height inches (leaguedashplayerbiostats) heuristic when position column absent.
HEIGHT_INCHES_GUARD_MAX: float = 76.0
HEIGHT_INCHES_BIG_MIN: float = 80.0

# ---------------------------------------------------------------------------
# Shot diet / zone labels (leaguedash*shotlocations column stems)
# ---------------------------------------------------------------------------

ZONE_RESTRICTED: str = "Restricted Area"
ZONE_PAINT_NON_RA: str = "In The Paint (Non-RA)"
ZONE_MID_RANGE: str = "Mid-Range"
ZONE_LEFT_CORNER_3: str = "Left Corner 3"
ZONE_RIGHT_CORNER_3: str = "Right Corner 3"
ZONE_ABOVE_BREAK_3: str = "Above the Break 3"
ZONE_BACKCOURT: str = "Backcourt"

SHOT_LOCATION_ZONES: tuple[str, ...] = (
    ZONE_RESTRICTED,
    ZONE_PAINT_NON_RA,
    ZONE_MID_RANGE,
    ZONE_LEFT_CORNER_3,
    ZONE_RIGHT_CORNER_3,
    ZONE_ABOVE_BREAK_3,
    ZONE_BACKCOURT,
)

# MultiIndex / flattened suffix for attempt volume in shot-location tables.
SHOT_COL_FGA_SUFFIX: str = "FGA"

# ---------------------------------------------------------------------------
# Derived-rate coefficients
# ---------------------------------------------------------------------------

# Assist-to-turnover and similar ratios: cap denominator to avoid blow-ups.
TURNOVER_DENOM_FLOOR: float = 0.1

# True shooting: league FT possession value approximation (Oliver four factors).
TS_FT_POSSESSION_WEIGHT: float = 0.44

# Free-throw rate proxy: FTA per FGA (team/player foul drawing).
FTA_PER_FGA_SCALE: float = 1.0

# 3-point attempt share of total FGA (modern spacing benchmark ~0.38–0.42 league).
LEAGUE_AVG_FG3A_RATE: float = 0.40

# Corner 3 share of all 3PA — spacing teams often target >0.18.
LEAGUE_AVG_CORNER3_SHARE_OF_FG3A: float = 0.18

# Rim attempt share (restricted + paint) typical team ~0.55–0.62.
LEAGUE_AVG_RIM_ATTEMPT_SHARE: float = 0.58

# ---------------------------------------------------------------------------
# Defense proxy weights (interpretable index, not a learned model)
# ---------------------------------------------------------------------------

# Weight defended-shot frequency in composite defense proxy.
DEF_PROXY_WEIGHT_FREQ: float = 0.25

# Weight opponent FG% suppression (PCT_PLUSMINUS from leaguedashptdefend).
DEF_PROXY_WEIGHT_FG_SUPPRESSION: float = 0.35

# Weight box/impact stats (STL, BLK, DEF_RATING z-scores).
DEF_PROXY_WEIGHT_BOX: float = 0.40

# ---------------------------------------------------------------------------
# Team weakness proxies (relative to league in same season)
# ---------------------------------------------------------------------------

# Team defensive weakness: positive means worse than league median DEF_RATING.
TEAM_DEF_WEAKNESS_SCALE: float = 1.0

# Team offensive weakness: negative OFF_RATING vs league median.
TEAM_OFF_WEAKNESS_SCALE: float = 1.0

# Turnover problem proxy: TM_TOV_PCT above league median.
TEAM_TOV_WEAKNESS_SCALE: float = 1.0

# ---------------------------------------------------------------------------
# Feature column prefixes (stable API for scoring layer)
# ---------------------------------------------------------------------------

PLAYER_FEATURE_PREFIX: str = "pf_"
TEAM_FEATURE_PREFIX: str = "tf_"

# Metadata columns preserved alongside feature matrix
COL_PLAYER_ID: str = "PLAYER_ID"
COL_TEAM_ID: str = "TEAM_ID"
COL_ROLE_BUCKET: str = "role_bucket"
COL_MINUTES_STABLE: str = "minutes_stable_flag"
