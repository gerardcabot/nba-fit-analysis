"""Orchestrate raw fetch + interim normalization for ingest tiers."""

from __future__ import annotations

from dataclasses import dataclass, field

from nba_fit.config.settings import (
    INGEST_TIER_MVP,
    INGEST_TIER_ROLE,
    OPTION_A_MVP_ENDPOINTS,
    OPTION_B_ROLE_ENDPOINTS,
    get_settings,
)
from nba_fit.data.client import FetchResult, NBAClient
from nba_fit.data.fetchers.league_dash import fetch_option_a_mvp
from nba_fit.data.fetchers.lineups_onoff import fetch_option_b_role
from nba_fit.data.storage import ensure_dirs
from nba_fit.normalize.lineups import (
    build_lineup_units_table,
    build_onoff_table,
    write_lineup_units_table,
    write_onoff_table,
)
from nba_fit.normalize.players import build_players_table, write_players_table
from nba_fit.normalize.teams import build_teams_table, write_teams_table

_TIER_ENDPOINTS: dict[str, tuple[str, ...]] = {
    INGEST_TIER_MVP: OPTION_A_MVP_ENDPOINTS,
    INGEST_TIER_ROLE: OPTION_B_ROLE_ENDPOINTS,
}


@dataclass
class IngestResult:
    """Summary of one ingest run."""

    season: str
    tier: str
    fetched: dict[str, FetchResult] = field(default_factory=dict)
    players_path: str | None = None
    teams_path: str | None = None
    player_rows: int = 0
    team_rows: int = 0
    lineup_units_path: str | None = None
    onoff_path: str | None = None
    lineup_units_rows: int = 0
    onoff_rows: int = 0


def resolve_endpoints(tier: str) -> tuple[str, ...]:
    """Map CLI tier name to endpoint list."""
    key = tier.lower()
    if key not in _TIER_ENDPOINTS:
        known = ", ".join(sorted(_TIER_ENDPOINTS))
        raise ValueError(f"Unknown ingest tier {tier!r}; known: {known}")
    return _TIER_ENDPOINTS[key]


def _run_ingest_mvp(
    *,
    season: str,
    tier: str,
    use_cache: bool,
    nba_client: NBAClient,
    endpoints: tuple[str, ...],
) -> IngestResult:
    fetched = fetch_option_a_mvp(
        nba_client,
        season=season,
        use_cache=use_cache,
        endpoints=endpoints,
    )
    players_df = build_players_table(season, fetched)
    teams_df = build_teams_table(season, fetched)
    players_out = write_players_table(players_df, season)
    teams_out = write_teams_table(teams_df, season)
    return IngestResult(
        season=season,
        tier=tier,
        fetched=fetched,
        players_path=str(players_out),
        teams_path=str(teams_out),
        player_rows=len(players_df),
        team_rows=len(teams_df),
    )


def _run_ingest_role(
    *,
    season: str,
    tier: str,
    use_cache: bool,
    nba_client: NBAClient,
    endpoints: tuple[str, ...],
) -> IngestResult:
    fetched = fetch_option_b_role(
        nba_client,
        season=season,
        use_cache=use_cache,
        endpoints=endpoints,
    )
    lineup_df = build_lineup_units_table(season, fetched)
    onoff_df = build_onoff_table(season, fetched)
    lineup_out = write_lineup_units_table(lineup_df, season)
    onoff_out = write_onoff_table(onoff_df, season)
    return IngestResult(
        season=season,
        tier=tier,
        fetched=fetched,
        lineup_units_path=str(lineup_out),
        onoff_path=str(onoff_out),
        lineup_units_rows=len(lineup_df),
        onoff_rows=len(onoff_df),
    )


def run_ingest(
    *,
    season: str | None = None,
    tier: str = INGEST_TIER_MVP,
    use_cache: bool = True,
    client: NBAClient | None = None,
) -> IngestResult:
    """
    Fetch tier endpoints and write interim Parquet tables.

    * ``mvp`` — Option A player/team league-dash tables.
    * ``role`` — Option B lineup units and on/off interim tables.

    Returns summary paths and row counts.
    """
    settings = get_settings()
    season = season or settings.default_season
    tier_key = tier.lower()
    endpoints = resolve_endpoints(tier_key)

    ensure_dirs()
    nba_client = client or NBAClient(settings=settings)

    if tier_key == INGEST_TIER_MVP:
        return _run_ingest_mvp(
            season=season,
            tier=tier_key,
            use_cache=use_cache,
            nba_client=nba_client,
            endpoints=endpoints,
        )
    if tier_key == INGEST_TIER_ROLE:
        return _run_ingest_role(
            season=season,
            tier=tier_key,
            use_cache=use_cache,
            nba_client=nba_client,
            endpoints=endpoints,
        )
    known = ", ".join(sorted(_TIER_ENDPOINTS))
    raise ValueError(f"Unknown ingest tier {tier!r}; known: {known}")
