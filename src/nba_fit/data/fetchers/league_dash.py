"""League-dash endpoint fetchers for Option A (season-scoped NBA stats tables)."""

from __future__ import annotations

from typing import Any

from nba_fit.config.settings import (
    LEAGUE_DASH_PRIMARY_DATASET,
    LEAGUE_DASH_TEAM_ID_NULLABLE_ALL,
    OPTION_A_MVP_ENDPOINTS,
)
from nba_fit.data.client import FetchResult, NBAClient

# Endpoints that accept team_id_nullable; empty string = full league (not probe slice).
_TEAM_FILTERED_ENDPOINTS: frozenset[str] = frozenset(
    {
        "leaguedashplayerstats",
        "leaguedashteamstats",
        "leaguedashplayerbiostats",
        "leaguedashplayershotlocations",
        "leaguedashteamshotlocations",
    }
)


def primary_dataset_name(endpoint: str) -> str:
    """Return the main dataset key for *endpoint* (raises KeyError if unknown)."""
    return LEAGUE_DASH_PRIMARY_DATASET[endpoint.lower()]


def primary_frame(result: FetchResult) -> Any:
    """Extract the primary DataFrame from a fetch result."""
    import pandas as pd

    name = primary_dataset_name(result.endpoint)
    if name in result.frames:
        return result.frames[name]
    if len(result.frames) == 1:
        return next(iter(result.frames.values()))
    raise KeyError(
        f"Primary dataset {name!r} not in {result.endpoint} frames: {list(result.frames)}"
    )


def league_dash_kwargs(endpoint: str) -> dict[str, Any]:
    """Extra kwargs so league-dash pulls cover all teams, not probe defaults."""
    if endpoint.lower() in _TEAM_FILTERED_ENDPOINTS:
        return {"team_id_nullable": LEAGUE_DASH_TEAM_ID_NULLABLE_ALL}
    return {}


def fetch_league_dash(
    client: NBAClient,
    endpoint: str,
    *,
    season: str,
    use_cache: bool = True,
) -> FetchResult:
    """Fetch one league-dash endpoint for *season*."""
    return client.fetch(
        endpoint,
        season=season,
        use_cache=use_cache,
        persist_cache=True,
        **league_dash_kwargs(endpoint),
    )


def fetch_option_a_mvp(
    client: NBAClient,
    *,
    season: str,
    use_cache: bool = True,
    endpoints: tuple[str, ...] = OPTION_A_MVP_ENDPOINTS,
) -> dict[str, FetchResult]:
    """Fetch all Option A MVP league-dash endpoints."""
    results: dict[str, FetchResult] = {}
    for endpoint in endpoints:
        results[endpoint] = fetch_league_dash(
            client,
            endpoint,
            season=season,
            use_cache=use_cache,
        )
    return results
