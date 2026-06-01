"""Project paths, seasons, cache policy, and domain-derived constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Basketball / analytics domain constants (no magic numbers without rationale)
# ---------------------------------------------------------------------------

# Dean Oliver (Basketball on Paper): ~100 possessions per team per full NBA game
# at league-average pace; used as sanity check for per-100 scaling and stint sizes.
POSSESSIONS_PER_TEAM_PER_GAME_REGULATION: float = 100.0

# Common public-analytics practice (Oliver, Kubatko/B-R, modern tracking blogs):
# ~500 minutes per season is a rough floor before per-minute rate stats stabilize
# for rotation players; stars need more. We use this for feature eligibility flags.
MIN_SEASON_MINUTES_STABLE_RATES: int = 500

# NBA regulation: 48 minutes; OT adds 5 each — used when converting game logs to rates.
REGULATION_GAME_MINUTES: int = 48

# RAPM / lineup models: regularized APM literature often requires thousands of
# possessions; we flag low-sample players below this threshold (not a hard filter).
MIN_POSSESSIONS_RAPM_FLAG: int = 1500

# Exponential recency half-life for stint weighting (games); recent lineups count more.
RAPM_RECENCY_HALF_LIFE_GAMES: float = 15.0

# Probe / smoke-test context IDs (same as probe_all_nba_endpoints.py)
DEFAULT_SEASON: str = "2025-26"
DEFAULT_PLAYER_ID: int = 1629029  # Luka Doncic
DEFAULT_TEAM_ID: int = 1610612747  # Los Angeles Lakers
DEFAULT_GAME_ID: str = "0022501198"
DEFAULT_LEAGUE_ID: str = "00"

# Seasons to backfill in later phases (newest first)
DEFAULT_SEASONS: tuple[str, ...] = ("2025-26", "2024-25", "2023-24")

# ---------------------------------------------------------------------------
# Ingest tiers (Option A interpretable fit index — league-dash MVP tables)
# ---------------------------------------------------------------------------

# CLI tier name for the first milestone ingest bundle
INGEST_TIER_MVP: str = "mvp"

# Option A core league-dash endpoints (season-scoped; no per-game chunking)
OPTION_A_MVP_ENDPOINTS: tuple[str, ...] = (
    "leaguedashplayerstats",
    "leaguedashteamstats",
    "playerestimatedmetrics",
    "teamestimatedmetrics",
    "leaguedashplayerbiostats",
    "leaguedashplayershotlocations",
    "leaguedashteamshotlocations",
)

# League-dash bulk pulls: blank team filter returns all teams (probe uses one team)
LEAGUE_DASH_TEAM_ID_NULLABLE_ALL: str = ""

# Canonical nba_api dataset names inside multi-frame responses
LEAGUE_DASH_PRIMARY_DATASET: dict[str, str] = {
    "leaguedashplayerstats": "LeagueDashPlayerStats",
    "leaguedashteamstats": "LeagueDashTeamStats",
    "playerestimatedmetrics": "PlayerEstimatedMetrics",
    "teamestimatedmetrics": "TeamEstimatedMetrics",
    "leaguedashplayerbiostats": "LeagueDashPlayerBioStats",
    "leaguedashplayershotlocations": "ShotLocations",
    "leaguedashteamshotlocations": "ShotLocations",
}

# ---------------------------------------------------------------------------
# Ingest tiers (Option B — lineups, on/off, role modeling inputs)
# ---------------------------------------------------------------------------

# CLI tier name for lineup + on/off bundle (Option B data layer)
INGEST_TIER_ROLE: str = "role"

# Option B core endpoints (team-scoped pulls loop NBA_TEAM_IDS; league lineups bulk)
OPTION_B_ROLE_ENDPOINTS: tuple[str, ...] = (
    "teamplayeronoffsummary",
    "teamplayeronoffdetails",
    "leaguedashlineups",
    "teamdashlineups",
)

# All 30 NBA franchise TEAM_ID values (stats.nba.com); used for per-team lineup/on-off pulls
NBA_TEAM_IDS: tuple[int, ...] = (
    1610612737,  # ATL
    1610612738,  # BOS
    1610612751,  # BKN
    1610612766,  # CHA
    1610612741,  # CHI
    1610612739,  # CLE
    1610612742,  # DAL
    1610612743,  # DEN
    1610612765,  # DET
    1610612744,  # GSW
    1610612745,  # HOU
    1610612754,  # IND
    1610612746,  # LAC
    1610612747,  # LAL
    1610612763,  # MEM
    1610612748,  # MIA
    1610612749,  # MIL
    1610612750,  # MIN
    1610612740,  # NOP
    1610612752,  # NYK
    1610612760,  # OKC
    1610612753,  # ORL
    1610612755,  # PHI
    1610612756,  # PHX
    1610612757,  # POR
    1610612758,  # SAC
    1610612759,  # SAS
    1610612761,  # TOR
    1610612762,  # UTA
    1610612764,  # WAS
)

# Primary nba_api dataset keys for Option B endpoints (multi-frame responses)
LINEUP_ONOFF_PRIMARY_DATASET: dict[str, str] = {
    "leaguedashlineups": "Lineups",
    "teamdashlineups": "Lineups",
    "teamplayeronoffsummary": "PlayersOnCourtTeamPlayerOnOffSummary",
    "teamplayeronoffdetails": "PlayersOnCourtTeamPlayerOnOffDetails",
}

# On/off frames to stack into interim onoff table (court status inferred from name)
ONOFF_SUMMARY_DATASETS: tuple[str, ...] = (
    "PlayersOnCourtTeamPlayerOnOffSummary",
    "PlayersOffCourtTeamPlayerOnOffSummary",
)
ONOFF_DETAILS_DATASETS: tuple[str, ...] = (
    "PlayersOnCourtTeamPlayerOnOffDetails",
    "PlayersOffCourtTeamPlayerOnOffDetails",
)

# Interim table names (hive roots under data/interim/)
INTERIM_TABLE_PLAYERS: str = "players"
INTERIM_TABLE_TEAMS: str = "teams"
INTERIM_TABLE_LINEUP_UNITS: str = "lineup_units"
INTERIM_TABLE_ONOFF: str = "onoff"

# On/off and lineup columns used in normalization / visuals
ONOFF_STAT_MIN: str = "MIN"
ONOFF_STAT_NET_RATING: str = "NET_RATING"
ONOFF_STAT_PLUS_MINUS: str = "PLUS_MINUS"
ONOFF_COL_VS_PLAYER_ID: str = "VS_PLAYER_ID"
ONOFF_COL_VS_PLAYER_NAME: str = "VS_PLAYER_NAME"
ONOFF_COL_COURT_STATUS: str = "COURT_STATUS"
LINEUP_COL_GROUP_NAME: str = "GROUP_NAME"
LINEUP_COL_GROUP_ID: str = "GROUP_ID"

# ---------------------------------------------------------------------------
# Ingest tiers (Option C — possession / stint foundation for lineup impact)
# ---------------------------------------------------------------------------

# CLI tier name for play-by-play + rotation + possession interim tables
INGEST_TIER_IMPACT: str = "impact"

# Option C endpoints: game index, per-game PBP, rotation stints
OPTION_C_IMPACT_ENDPOINTS: tuple[str, ...] = (
    "leaguegamefinder",
    "playbyplayv3",
    "gamerotation",
)

# Regular season: 30 teams × 82 games / 2 unique matchups per season
REGULAR_SEASON_GAME_COUNT: int = 1230

# stats.nba.com leaguegamefinder column names
LEAGUE_GAME_FINDER_GAME_ID_COL: str = "GAME_ID"
LEAGUE_GAME_FINDER_DATE_COL: str = "GAME_DATE"

# playbyplayv3 / gamerotation primary dataset keys (probe_all_results.json)
PBP_PLAYBYPLAY_DATASET: str = "PlayByPlay"
GAME_ROTATION_TEAM_DATASETS: tuple[str, ...] = ("HomeTeam", "AwayTeam")

# playbyplayv3 event columns (V3 schema)
PBP_COL_GAME_ID: str = "gameId"
PBP_COL_ACTION_NUMBER: str = "actionNumber"
PBP_COL_CLOCK: str = "clock"
PBP_COL_PERIOD: str = "period"
PBP_COL_TEAM_ID: str = "teamId"
PBP_COL_ACTION_TYPE: str = "actionType"
PBP_COL_SUB_TYPE: str = "subType"
PBP_COL_SHOT_RESULT: str = "shotResult"
PBP_COL_IS_FIELD_GOAL: str = "isFieldGoal"

# gamerotation stint columns (seconds from game start, deciseconds in API)
ROTATION_COL_IN_TIME: str = "IN_TIME_REAL"
ROTATION_COL_OUT_TIME: str = "OUT_TIME_REAL"
ROTATION_COL_PERSON_ID: str = "PERSON_ID"
ROTATION_COL_TEAM_ID: str = "TEAM_ID"

# Regulation / OT period length (seconds) for clock ↔ rotation alignment
REGULATION_PERIOD_SECONDS: int = 12 * 60
OT_PERIOD_SECONDS: int = 5 * 60
ROTATION_API_DECISECONDS_PER_SECOND: int = 10

# Possession row provenance labels (interim possessions table)
POSSESSION_SOURCE_PBPSTATS: str = "pbpstats"
POSSESSION_SOURCE_EVENTS_ROTATION: str = "events_rotation"

# Interim hive table for possession / stint rows
INTERIM_TABLE_POSSESSIONS: str = "possessions"

# Dev ingest cap: each game needs PBP + rotation HTTP calls (2× per game minimum).
# Full season (~1,230 games) exceeds polite stats.nba.com pacing for local iteration;
# cap keeps ``ingest --tier impact`` usable during development (see REQUEST_* policy).
INGEST_IMPACT_MAX_GAMES_DEV: int = 50

# ``max_games=0`` or ``None`` (via ``--full-season``) means no cap on impact pulls.
INGEST_IMPACT_MAX_GAMES_UNLIMITED: int = 0

# ---------------------------------------------------------------------------
# Ingest tiers (Option D / Phase 1 — tactical hustle, defend, gravity bulk)
# ---------------------------------------------------------------------------

# CLI tier name for supplemental league-bulk tactical tables
INGEST_TIER_TACTICAL: str = "tactical"

# Option D tactical endpoints (season-scoped league bulk; no per-game chunking)
OPTION_D_TACTICAL_ENDPOINTS: tuple[str, ...] = (
    "leaguehustlestatsplayer",
    "leaguehustlestatsteam",
    "leaguedashptdefend",
    "gravityleaders",
)

# Primary nba_api dataset keys for tactical endpoints (multi-frame responses)
TACTICAL_PRIMARY_DATASET: dict[str, str] = {
    "leaguehustlestatsplayer": "HustleStatsPlayer",
    "leaguehustlestatsteam": "HustleStatsTeam",
    "leaguedashptdefend": "LeagueDashPTDefend",
    "gravityleaders": "leaders",
}

# Box-score columns used in MVP visuals / feature eligibility
PLAYER_STAT_PTS: str = "PTS"
PLAYER_STAT_USG_PCT: str = "USG_PCT"
PLAYER_STAT_TS_PCT: str = "TS_PCT"
PLAYER_STAT_EST_USG_PCT: str = "E_USG_PCT"
PLAYER_STAT_MIN: str = "MIN"

# ---------------------------------------------------------------------------
# HTTP / cache policy (extracted from probe_all_nba_endpoints.py)
# ---------------------------------------------------------------------------

# Initial polite delay between stats.nba.com calls (seconds)
REQUEST_SLEEP_INITIAL_SEC: float = 0.7

# Multiplicative backoff cap per request (mirrors probe script)
REQUEST_SLEEP_BACKOFF_FACTOR: float = 1.08
REQUEST_SLEEP_MAX_SEC: float = 3.5

# Retries on transient HTTP / JSON failures
REQUEST_MAX_RETRIES: int = 3
REQUEST_TIMEOUT_SEC: int = 45

# Raw Parquet cache TTL — stats endpoints refresh nightly in-season
CACHE_TTL_HOURS: int = 24

BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def _project_root() -> Path:
    """Walk up from this file to repo root (contains pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


@dataclass(frozen=True)
class Settings:
    """Resolved paths and runtime options."""

    root: Path
    data_raw: Path
    data_interim: Path
    data_features: Path
    models_dir: Path
    reports_dir: Path
    probe_results_path: Path
    seasons: tuple[str, ...]
    cache_ttl_hours: int
    default_season: str

    def raw_parquet_dir(self, source: str, endpoint: str, season: str) -> Path:
        """data/raw/{source}/{endpoint}/season={season}/"""
        return self.data_raw / source / endpoint / f"season={season}"

    def interim_path(self, name: str, season: str) -> Path:
        return self.data_interim / name / f"season={season}"

    def features_path(self, name: str, season: str) -> Path:
        return self.data_features / name / f"season={season}"


def get_settings() -> Settings:
    root = _project_root()
    return Settings(
        root=root,
        data_raw=root / "data" / "raw",
        data_interim=root / "data" / "interim",
        data_features=root / "data" / "features",
        models_dir=root / "models",
        reports_dir=root / "reports",
        probe_results_path=root / "probe_all_results.json",
        seasons=DEFAULT_SEASONS,
        cache_ttl_hours=CACHE_TTL_HOURS,
        default_season=DEFAULT_SEASON,
    )
