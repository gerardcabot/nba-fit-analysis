"""Orchestrate raw fetch + interim normalization for ingest tiers."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nba_fit.config.settings import (
    INGEST_IMPACT_MAX_GAMES_DEV,
    INGEST_IMPACT_MAX_GAMES_UNLIMITED,
    INGEST_TIER_IMPACT,
    INGEST_TIER_MVP,
    INGEST_TIER_ROLE,
    INGEST_TIER_TACTICAL,
    OPTION_A_MVP_ENDPOINTS,
    OPTION_B_ROLE_ENDPOINTS,
    OPTION_C_IMPACT_ENDPOINTS,
    OPTION_D_TACTICAL_ENDPOINTS,
    get_settings,
)
from nba_fit.data.client import FetchResult, NBAClient
from nba_fit.data.fetchers.league_dash import fetch_option_a_mvp
from nba_fit.data.fetchers.lineups_onoff import fetch_option_b_role
from nba_fit.data.fetchers.pbp import fetch_option_c_impact
from nba_fit.data.fetchers.tactical import fetch_option_d_tactical
from nba_fit.data.storage import ensure_dirs
from nba_fit.features.store import materialize_features
from nba_fit.normalize.lineups import (
    build_lineup_units_table,
    build_onoff_table,
    write_lineup_units_table,
    write_onoff_table,
)
from nba_fit.normalize.players import build_players_table, write_players_table
from nba_fit.normalize.possessions import build_and_write_from_fetched
from nba_fit.normalize.teams import build_teams_table, write_teams_table

_TIER_ENDPOINTS: dict[str, tuple[str, ...]] = {
    INGEST_TIER_MVP: OPTION_A_MVP_ENDPOINTS,
    INGEST_TIER_ROLE: OPTION_B_ROLE_ENDPOINTS,
    INGEST_TIER_IMPACT: OPTION_C_IMPACT_ENDPOINTS,
    INGEST_TIER_TACTICAL: OPTION_D_TACTICAL_ENDPOINTS,
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
    possessions_paths: list[str] = field(default_factory=list)
    possessions_rows: int = 0
    games_ingested: int = 0
    features_materialized: bool = False
    player_features_path: str | None = None
    team_features_path: str | None = None
    scaling_params_path: str | None = None


def resolve_impact_max_games(
    *,
    max_games: int | None,
    full_season: bool = False,
) -> int | None:
    """
    Resolve impact-tier game cap.

    * ``full_season=True`` or ``max_games=0`` → no cap (``None``).
    * ``max_games=None`` (default) → dev cap ``INGEST_IMPACT_MAX_GAMES_DEV``.
    * explicit positive ``max_games`` → that cap.
    """
    if full_season or max_games == INGEST_IMPACT_MAX_GAMES_UNLIMITED:
        return None
    if max_games is None:
        return INGEST_IMPACT_MAX_GAMES_DEV
    return max_games


def _run_ingest_tactical(
    *,
    season: str,
    tier: str,
    use_cache: bool,
    nba_client: NBAClient,
    endpoints: tuple[str, ...],
) -> IngestResult:
    fetched = fetch_option_d_tactical(
        nba_client,
        season=season,
        use_cache=use_cache,
        endpoints=endpoints,
    )
    return IngestResult(season=season, tier=tier, fetched=fetched)


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
    result = IngestResult(
        season=season,
        tier=tier,
        fetched=fetched,
        players_path=str(players_out),
        teams_path=str(teams_out),
        player_rows=len(players_df),
        team_rows=len(teams_df),
    )
    store = materialize_features(season)
    result.features_materialized = True
    result.player_features_path = str(store.player_path)
    result.team_features_path = str(store.team_path)
    result.scaling_params_path = str(store.scaling_path)
    return result


def _run_ingest_impact(
    *,
    season: str,
    tier: str,
    use_cache: bool,
    nba_client: NBAClient,
    endpoints: tuple[str, ...],
    max_games: int | None,
) -> IngestResult:
    fetched = fetch_option_c_impact(
        nba_client,
        season=season,
        use_cache=use_cache,
        endpoints=endpoints,
        max_games=max_games,
    )
    possessions_df, paths = build_and_write_from_fetched(season, fetched)
    flat_fetched: dict[str, FetchResult] = {
        "leaguegamefinder": fetched["leaguegamefinder"],
    }
    for endpoint_key in ("playbyplayv3", "gamerotation"):
        per_game: dict[str, FetchResult] = fetched.get(endpoint_key) or {}
        if not per_game:
            continue
        frames: dict[str, list] = {}
        for result in per_game.values():
            for name, df in result.frames.items():
                frames.setdefault(name, []).append(df)
        merged = {
            name: pd.concat(chunks, ignore_index=True) for name, chunks in frames.items()
        }
        flat_fetched[endpoint_key] = FetchResult(
            endpoint=endpoint_key,
            season=season,
            frames=merged,
        )
    return IngestResult(
        season=season,
        tier=tier,
        fetched=flat_fetched,
        possessions_paths=[str(p) for p in paths],
        possessions_rows=len(possessions_df),
        games_ingested=len(fetched.get("game_ids") or []),
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
    max_games: int | None = None,
    full_season: bool = False,
) -> IngestResult:
    """
    Fetch tier endpoints and write interim Parquet tables.

    * ``mvp`` — Option A player/team league-dash tables (+ feature store materialization).
    * ``role`` — Option B lineup units and on/off interim tables.
    * ``impact`` — Option C play-by-play, rotation, and possession interim tables.
    * ``tactical`` — Option D hustle, defend, and gravity league-bulk raw cache.

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
    if tier_key == INGEST_TIER_TACTICAL:
        return _run_ingest_tactical(
            season=season,
            tier=tier_key,
            use_cache=use_cache,
            nba_client=nba_client,
            endpoints=endpoints,
        )
    if tier_key == INGEST_TIER_IMPACT:
        cap = resolve_impact_max_games(max_games=max_games, full_season=full_season)
        return _run_ingest_impact(
            season=season,
            tier=tier_key,
            use_cache=use_cache,
            nba_client=nba_client,
            endpoints=endpoints,
            max_games=cap,
        )
    known = ", ".join(sorted(_TIER_ENDPOINTS))
    raise ValueError(f"Unknown ingest tier {tier!r}; known: {known}")
