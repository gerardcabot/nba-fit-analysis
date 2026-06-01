"""Lineup and on/off endpoint fetchers for Option B (role / team-need modeling)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nba_fit.config.settings import (
    LEAGUE_DASH_TEAM_ID_NULLABLE_ALL,
    LINEUP_ONOFF_PRIMARY_DATASET,
    NBA_TEAM_IDS,
    OPTION_B_ROLE_ENDPOINTS,
)
from nba_fit.data.client import FetchResult, NBAClient

_TEAM_SCOPED_ENDPOINTS: frozenset[str] = frozenset(
    {
        "teamdashlineups",
        "teamplayeronoffsummary",
        "teamplayeronoffdetails",
    }
)


def primary_dataset_name(endpoint: str) -> str:
    """Return the main dataset key for *endpoint* (raises KeyError if unknown)."""
    return LINEUP_ONOFF_PRIMARY_DATASET[endpoint.lower()]


def lineup_onoff_kwargs(endpoint: str, *, team_id: int | None = None) -> dict[str, Any]:
    """Extra kwargs for lineup/on-off pulls (team filter or full-league lineups)."""
    ep = endpoint.lower()
    if ep == "leaguedashlineups":
        if team_id is None:
            return {"team_id_nullable": LEAGUE_DASH_TEAM_ID_NULLABLE_ALL}
        return {"team_id_nullable": str(team_id)}
    if ep in _TEAM_SCOPED_ENDPOINTS:
        if team_id is None:
            raise ValueError(f"{endpoint} requires team_id")
        return {"team_id": team_id}
    return {}


def fetch_lineup_onoff(
    client: NBAClient,
    endpoint: str,
    *,
    season: str,
    team_id: int | None = None,
    use_cache: bool = True,
) -> FetchResult:
    """Fetch one lineup/on-off endpoint for *season* (optional *team_id*)."""
    return client.fetch(
        endpoint,
        season=season,
        use_cache=use_cache,
        persist_cache=True,
        **lineup_onoff_kwargs(endpoint, team_id=team_id),
    )


def _merge_fetch_results(
    endpoint: str,
    season: str,
    parts: list[FetchResult],
) -> FetchResult:
    """Concatenate per-team frames into one FetchResult keyed by dataset name."""
    merged_frames: dict[str, list[pd.DataFrame]] = {}
    from_cache = True
    cache_path = None
    elapsed = 0.0
    kwargs_used: dict[str, Any] = {}

    for part in parts:
        from_cache = from_cache and part.from_cache
        elapsed += part.elapsed_sec
        if part.cache_path and cache_path is None:
            cache_path = part.cache_path
        kwargs_used = part.kwargs_used
        for name, df in part.frames.items():
            merged_frames.setdefault(name, []).append(df)

    frames = {
        name: pd.concat(chunks, ignore_index=True) for name, chunks in merged_frames.items()
    }
    return FetchResult(
        endpoint=endpoint,
        season=season,
        frames=frames,
        kwargs_used=kwargs_used,
        from_cache=from_cache,
        cache_path=cache_path,
        elapsed_sec=elapsed,
    )


def _fetch_team_scoped_all(
    client: NBAClient,
    endpoint: str,
    *,
    season: str,
    use_cache: bool,
    team_ids: tuple[int, ...],
) -> FetchResult:
    parts = [
        fetch_lineup_onoff(
            client,
            endpoint,
            season=season,
            team_id=team_id,
            use_cache=use_cache,
        )
        for team_id in team_ids
    ]
    return _merge_fetch_results(endpoint, season, parts)


def fetch_option_b_role(
    client: NBAClient,
    *,
    season: str,
    use_cache: bool = True,
    endpoints: tuple[str, ...] = OPTION_B_ROLE_ENDPOINTS,
    team_ids: tuple[int, ...] = NBA_TEAM_IDS,
) -> dict[str, FetchResult]:
    """Fetch all Option B role-tier endpoints (bulk league lineups + per-team pulls)."""
    results: dict[str, FetchResult] = {}
    for endpoint in endpoints:
        ep = endpoint.lower()
        if ep in _TEAM_SCOPED_ENDPOINTS:
            results[endpoint] = _fetch_team_scoped_all(
                client,
                endpoint,
                season=season,
                use_cache=use_cache,
                team_ids=team_ids,
            )
        else:
            results[endpoint] = fetch_lineup_onoff(
                client,
                endpoint,
                season=season,
                use_cache=use_cache,
            )
    return results


def primary_frame(result: FetchResult) -> pd.DataFrame:
    """Extract the primary DataFrame from a lineup/on-off fetch result."""
    name = primary_dataset_name(result.endpoint)
    if name in result.frames:
        return result.frames[name]
    if len(result.frames) == 1:
        return next(iter(result.frames.values()))
    raise KeyError(
        f"Primary dataset {name!r} not in {result.endpoint} frames: {list(result.frames)}"
    )


def lineup_units_frame(result: FetchResult) -> pd.DataFrame:
    """Primary five-man lineup DataFrame from a lineup endpoint fetch."""
    return primary_frame(result)
