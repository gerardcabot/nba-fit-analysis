"""Tactical endpoint fetchers for Option D (hustle, defend, gravity league bulk)."""

from __future__ import annotations

from typing import Any

from nba_fit.config.settings import (
    LEAGUE_DASH_TEAM_ID_NULLABLE_ALL,
    OPTION_D_TACTICAL_ENDPOINTS,
    TACTICAL_PRIMARY_DATASET,
)
from nba_fit.data.client import FetchResult, NBAClient

# Endpoints that accept team_id_nullable; empty string = full league (not probe slice).
_TEAM_FILTERED_ENDPOINTS: frozenset[str] = frozenset(
    {
        "leaguehustlestatsplayer",
        "leaguehustlestatsteam",
        "leaguedashptdefend",
    }
)


def primary_dataset_name(endpoint: str) -> str:
    """Return the main dataset key for *endpoint* (raises KeyError if unknown)."""
    return TACTICAL_PRIMARY_DATASET[endpoint.lower()]


def tactical_kwargs(endpoint: str) -> dict[str, Any]:
    """Extra kwargs so tactical pulls cover all teams, not probe defaults."""
    if endpoint.lower() in _TEAM_FILTERED_ENDPOINTS:
        return {"team_id_nullable": LEAGUE_DASH_TEAM_ID_NULLABLE_ALL}
    return {}


def fetch_tactical(
    client: NBAClient,
    endpoint: str,
    *,
    season: str,
    use_cache: bool = True,
) -> FetchResult:
    """Fetch one tactical endpoint for *season*."""
    return client.fetch(
        endpoint,
        season=season,
        use_cache=use_cache,
        persist_cache=True,
        **tactical_kwargs(endpoint),
    )


def fetch_option_d_tactical(
    client: NBAClient,
    *,
    season: str,
    use_cache: bool = True,
    endpoints: tuple[str, ...] = OPTION_D_TACTICAL_ENDPOINTS,
) -> dict[str, FetchResult]:
    """Fetch all Option D tactical-tier endpoints (league-bulk hustle/defend/gravity)."""
    results: dict[str, FetchResult] = {}
    for endpoint in endpoints:
        results[endpoint] = fetch_tactical(
            client,
            endpoint,
            season=season,
            use_cache=use_cache,
        )
    return results


def primary_frame(result: FetchResult) -> Any:
    """Extract the primary DataFrame from a tactical fetch result."""
    import pandas as pd

    name = primary_dataset_name(result.endpoint)
    if name in result.frames:
        return result.frames[name]
    if len(result.frames) == 1:
        return next(iter(result.frames.values()))
    raise KeyError(
        f"Primary dataset {name!r} not in {result.endpoint} frames: {list(result.frames)}"
    )
