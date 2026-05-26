"""Play-by-play and rotation fetchers for Option C (possession / lineup impact)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nba_fit.config.settings import (
    GAME_ROTATION_TEAM_DATASETS,
    LEAGUE_GAME_FINDER_DATE_COL,
    LEAGUE_GAME_FINDER_GAME_ID_COL,
    OPTION_C_IMPACT_ENDPOINTS,
    PBP_PLAYBYPLAY_DATASET,
)
from nba_fit.data.client import FetchResult, NBAClient
from nba_fit.data.fetchers.league_dash import primary_frame
from nba_fit.normalize.ids import canonical_game_id

_GAME_SCOPED_ENDPOINTS: frozenset[str] = frozenset({"playbyplayv3", "gamerotation"})


def league_game_finder_frame(result: FetchResult) -> pd.DataFrame:
    """Return the leaguegamefinder game list DataFrame."""
    if len(result.frames) == 1:
        return next(iter(result.frames.values()))
    return primary_frame(result)


def fetch_league_game_finder(
    client: NBAClient,
    *,
    season: str,
    use_cache: bool = True,
) -> FetchResult:
    """Fetch leaguegamefinder for *season* (two rows per game — home and away)."""
    return client.fetch("leaguegamefinder", season=season, use_cache=use_cache)


def league_game_ids(
    finder: FetchResult,
    *,
    max_games: int | None = None,
) -> list[str]:
    """
  Unique ``GAME_ID`` values from leaguegamefinder, newest ``GAME_DATE`` first.

  ``max_games`` caps how many games are returned (dev iteration / rate limits).
  """
    df = league_game_finder_frame(finder)
    if LEAGUE_GAME_FINDER_GAME_ID_COL not in df.columns:
        raise KeyError(
            f"leaguegamefinder missing {LEAGUE_GAME_FINDER_GAME_ID_COL!r}; "
            f"got {list(df.columns)}"
        )
    work = df.copy()
    if LEAGUE_GAME_FINDER_DATE_COL in work.columns:
        work = work.sort_values(LEAGUE_GAME_FINDER_DATE_COL, ascending=False)
    ids: list[str] = []
    seen: set[str] = set()
    for raw in work[LEAGUE_GAME_FINDER_GAME_ID_COL]:
        gid = canonical_game_id(raw)
        if gid is None or gid in seen:
            continue
        seen.add(gid)
        ids.append(gid)
        if max_games is not None and len(ids) >= max_games:
            break
    return ids


def playbyplay_frame(result: FetchResult) -> pd.DataFrame:
    """Primary PlayByPlayV3 event DataFrame."""
    if PBP_PLAYBYPLAY_DATASET in result.frames:
        return result.frames[PBP_PLAYBYPLAY_DATASET]
    return primary_frame(result)


def gamerotation_frames(result: FetchResult) -> dict[str, pd.DataFrame]:
    """Home/away rotation stint tables keyed by dataset name."""
    out: dict[str, pd.DataFrame] = {}
    for name in GAME_ROTATION_TEAM_DATASETS:
        if name in result.frames:
            out[name] = result.frames[name]
    if not out and result.frames:
        # Probe layout may differ — keep all non-empty frames
        for name, df in result.frames.items():
            if not df.empty:
                out[name] = df
    if not out:
        raise KeyError(
            f"gamerotation missing rotation datasets; got {list(result.frames)}"
        )
    return out


def gamerotation_stints(result: FetchResult) -> pd.DataFrame:
    """Stack home and away rotation stints with a ``rotation_side`` label."""
    frames = gamerotation_frames(result)
    chunks: list[pd.DataFrame] = []
    for side, df in frames.items():
        chunk = df.copy()
        chunk["rotation_side"] = side
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def fetch_playbyplay_game(
    client: NBAClient,
    game_id: str,
    *,
    season: str,
    use_cache: bool = True,
) -> FetchResult:
    """Fetch playbyplayv3 for a single *game_id*."""
    gid = canonical_game_id(game_id)
    if gid is None:
        raise ValueError(f"Invalid game_id: {game_id!r}")
    return client.fetch(
        "playbyplayv3",
        season=season,
        use_cache=use_cache,
        game_id=gid,
    )


def fetch_gamerotation_game(
    client: NBAClient,
    game_id: str,
    *,
    season: str,
    use_cache: bool = True,
) -> FetchResult:
    """Fetch gamerotation for a single *game_id*."""
    gid = canonical_game_id(game_id)
    if gid is None:
        raise ValueError(f"Invalid game_id: {game_id!r}")
    return client.fetch(
        "gamerotation",
        season=season,
        use_cache=use_cache,
        game_id=gid,
    )


def fetch_playbyplay_batch(
    client: NBAClient,
    game_ids: list[str],
    *,
    season: str,
    use_cache: bool = True,
) -> dict[str, FetchResult]:
    """Fetch playbyplayv3 for each game in *game_ids* (one HTTP call per game)."""
    results: dict[str, FetchResult] = {}
    for game_id in game_ids:
        gid = canonical_game_id(game_id)
        if gid is None:
            continue
        results[gid] = fetch_playbyplay_game(
            client,
            gid,
            season=season,
            use_cache=use_cache,
        )
    return results


def fetch_gamerotation_batch(
    client: NBAClient,
    game_ids: list[str],
    *,
    season: str,
    use_cache: bool = True,
) -> dict[str, FetchResult]:
    """Fetch gamerotation for each game in *game_ids*."""
    results: dict[str, FetchResult] = {}
    for game_id in game_ids:
        gid = canonical_game_id(game_id)
        if gid is None:
            continue
        results[gid] = fetch_gamerotation_game(
            client,
            gid,
            season=season,
            use_cache=use_cache,
        )
    return results


def fetch_option_c_impact(
    client: NBAClient,
    *,
    season: str,
    use_cache: bool = True,
    endpoints: tuple[str, ...] = OPTION_C_IMPACT_ENDPOINTS,
    max_games: int | None = None,
) -> dict[str, Any]:
    """
    Fetch Option C impact-tier data.

    Returns a dict with keys:
      - ``leaguegamefinder``: FetchResult
      - ``game_ids``: list[str]
      - ``playbyplayv3``: dict[game_id, FetchResult]
      - ``gamerotation``: dict[game_id, FetchResult]
    """
    if "leaguegamefinder" not in endpoints:
        raise ValueError("Option C impact fetch requires leaguegamefinder in endpoints")

    finder = fetch_league_game_finder(client, season=season, use_cache=use_cache)
    game_ids = league_game_ids(finder, max_games=max_games)

    out: dict[str, Any] = {
        "leaguegamefinder": finder,
        "game_ids": game_ids,
    }

    ep_set = {e.lower() for e in endpoints}
    if "playbyplayv3" in ep_set:
        out["playbyplayv3"] = fetch_playbyplay_batch(
            client,
            game_ids,
            season=season,
            use_cache=use_cache,
        )
    if "gamerotation" in ep_set:
        out["gamerotation"] = fetch_gamerotation_batch(
            client,
            game_ids,
            season=season,
            use_cache=use_cache,
        )
    return out
