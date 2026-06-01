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

# Interim table names (hive roots under data/interim/)
INTERIM_TABLE_PLAYERS: str = "players"
INTERIM_TABLE_TEAMS: str = "teams"

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
