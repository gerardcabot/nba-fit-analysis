"""Convert raw PBP + rotation pulls into possession rows (pbpstats or fallback)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from nba_fit.config.settings import (
    PBP_COL_ACTION_NUMBER,
    PBP_COL_ACTION_TYPE,
    PBP_COL_CLOCK,
    PBP_COL_IS_FIELD_GOAL,
    PBP_COL_PERIOD,
    PBP_COL_SHOT_RESULT,
    PBP_COL_SUB_TYPE,
    PBP_COL_TEAM_ID,
    POSSESSION_SOURCE_EVENTS_ROTATION,
    POSSESSION_SOURCE_PBPSTATS,
    REGULATION_PERIOD_SECONDS,
    ROTATION_API_DECISECONDS_PER_SECOND,
    ROTATION_COL_IN_TIME,
    ROTATION_COL_OUT_TIME,
    ROTATION_COL_PERSON_ID,
    ROTATION_COL_TEAM_ID,
    OT_PERIOD_SECONDS,
)
from nba_fit.data.fetchers.pbp import gamerotation_stints, playbyplay_frame
from nba_fit.data.client import FetchResult
from nba_fit.normalize.ids import canonical_game_id, canonical_player_id, canonical_team_id

_CLOCK_RE = re.compile(r"PT(?:(\d+)M)?([\d.]+)S", re.IGNORECASE)

# playbyplayv3 actionType tokens that typically close a possession (subset; dev foundation)
_PBP_TERMINAL_ACTION_SUBSTRINGS: tuple[str, ...] = (
    "turnover",
    "steal",
    "violation",
)


def _clock_remaining_seconds(clock: str) -> float | None:
    if not clock or not isinstance(clock, str):
        return None
    match = _CLOCK_RE.match(clock.strip())
    if not match:
        return None
    minutes = int(match.group(1) or 0)
    seconds = float(match.group(2))
    return minutes * 60 + seconds


def _period_length_seconds(period: int) -> int:
    if period <= 4:
        return REGULATION_PERIOD_SECONDS
    return OT_PERIOD_SECONDS


def _game_elapsed_deciseconds(period: int, clock_remaining_sec: float) -> float:
    """Map period clock to cumulative game time in rotation API units (deciseconds)."""
    period_idx = max(period - 1, 0)
    elapsed_in_period = _period_length_seconds(period) - clock_remaining_sec
    prior_periods_sec = sum(
        _period_length_seconds(p) for p in range(1, period)
    )
    total_sec = prior_periods_sec + max(elapsed_in_period, 0.0)
    return total_sec * ROTATION_API_DECISECONDS_PER_SECOND


def _lineup_id_from_player_ids(player_ids: list[int]) -> str | None:
    ids = sorted(str(pid) for pid in player_ids if pid)
    if len(ids) < 5:
        return None
    return "-".join(ids[:5])


def _active_players_at_deciseconds(
    rotation: pd.DataFrame,
    *,
    team_id: int,
    game_ds: float,
) -> list[int]:
    if rotation.empty:
        return []
    team_col = ROTATION_COL_TEAM_ID if ROTATION_COL_TEAM_ID in rotation.columns else "TEAM_ID"
    in_col = ROTATION_COL_IN_TIME if ROTATION_COL_IN_TIME in rotation.columns else "IN_TIME_REAL"
    out_col = ROTATION_COL_OUT_TIME if ROTATION_COL_OUT_TIME in rotation.columns else "OUT_TIME_REAL"
    person_col = ROTATION_COL_PERSON_ID if ROTATION_COL_PERSON_ID in rotation.columns else "PERSON_ID"

    mask = (
        rotation[team_col].map(canonical_team_id) == team_id
    ) & (rotation[in_col] <= game_ds) & (rotation[out_col] > game_ds)
    players: list[int] = []
    for raw in rotation.loc[mask, person_col]:
        pid = canonical_player_id(raw)
        if pid is not None:
            players.append(pid)
    return sorted(set(players))


def _is_miss(row: pd.Series) -> bool:
    if int(row.get(PBP_COL_IS_FIELD_GOAL, 0) or 0) != 1:
        return False
    return str(row.get(PBP_COL_SHOT_RESULT, "")).lower() != "made"


def _is_possession_terminal(row: pd.Series, *, previous_was_miss: bool) -> bool:
    action = str(row.get(PBP_COL_ACTION_TYPE, "")).lower()
    sub = str(row.get(PBP_COL_SUB_TYPE, "")).lower()

    if any(token in action for token in _PBP_TERMINAL_ACTION_SUBSTRINGS):
        return True
    if action == "period" and sub == "end":
        return True
    if int(row.get(PBP_COL_IS_FIELD_GOAL, 0) or 0) == 1:
        if str(row.get(PBP_COL_SHOT_RESULT, "")).lower() == "made":
            return True
    if "rebound" in action and previous_was_miss:
        return True
    return False


def _infer_offense_team_id(events: pd.DataFrame) -> int | None:
    for raw in events[PBP_COL_TEAM_ID]:
        tid = canonical_team_id(raw)
        if tid is not None:
            return tid
    return None


def _summarize_event_possession(
    events: pd.DataFrame,
    *,
    game_id: str,
    season: str,
    possession_number: int,
    rotation: pd.DataFrame,
) -> dict[str, Any]:
    first = events.iloc[0]
    last = events.iloc[-1]
    period = int(first.get(PBP_COL_PERIOD, 0) or 0)
    start_clock = str(first.get(PBP_COL_CLOCK, ""))
    end_clock = str(last.get(PBP_COL_CLOCK, ""))
    remaining = _clock_remaining_seconds(start_clock)
    lineup_id: str | None = None
    opponent_lineup_id: str | None = None
    has_lineup = False

    offense_team_id = _infer_offense_team_id(events)
    if remaining is not None and offense_team_id is not None:
        game_ds = _game_elapsed_deciseconds(period, remaining)
        offense_players = _active_players_at_deciseconds(
            rotation,
            team_id=offense_team_id,
            game_ds=game_ds,
        )
        lineup_id = _lineup_id_from_player_ids(offense_players)
        has_lineup = lineup_id is not None

        team_ids = {
            canonical_team_id(t)
            for t in rotation.get(ROTATION_COL_TEAM_ID, rotation.get("TEAM_ID", pd.Series())).tolist()
        }
        team_ids.discard(None)
        for tid in team_ids:
            if tid == offense_team_id:
                continue
            opp_players = _active_players_at_deciseconds(
                rotation,
                team_id=tid,
                game_ds=game_ds,
            )
            opponent_lineup_id = _lineup_id_from_player_ids(opp_players)
            break

    return {
        "game_id": game_id,
        "season": season,
        "possession_number": possession_number,
        "period": period,
        "start_clock": start_clock,
        "end_clock": end_clock,
        "offense_team_id": offense_team_id,
        "possession_start_type": None,
        "start_score_margin": None,
        "lineup_id": lineup_id,
        "opponent_lineup_id": opponent_lineup_id,
        "has_lineup_on_floor": has_lineup,
        "event_count": len(events),
        "source": POSSESSION_SOURCE_EVENTS_ROTATION,
    }


def possessions_from_events_and_rotation(
    pbp: pd.DataFrame,
    rotation: pd.DataFrame,
    *,
    game_id: str,
    season: str,
) -> pd.DataFrame:
    """
    Build possession rows from playbyplayv3 events + gamerotation stints.

    Lineup-on-floor is attached when rotation stints resolve five players.
    """
    if pbp.empty:
        return pd.DataFrame()

    work = pbp.copy()
    sort_col = PBP_COL_ACTION_NUMBER if PBP_COL_ACTION_NUMBER in work.columns else None
    if sort_col:
        work = work.sort_values(sort_col)
    else:
        work = work.reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    buffer: list[pd.Series] = []
    previous_was_miss = False
    possession_number = 0

    for _, row in work.iterrows():
        buffer.append(row)
        if _is_possession_terminal(row, previous_was_miss=previous_was_miss):
            possession_number += 1
            chunk = pd.DataFrame(buffer)
            rows.append(
                _summarize_event_possession(
                    chunk,
                    game_id=game_id,
                    season=season,
                    possession_number=possession_number,
                    rotation=rotation,
                )
            )
            buffer = []
        previous_was_miss = _is_miss(row)

    if buffer:
        possession_number += 1
        chunk = pd.DataFrame(buffer)
        rows.append(
            _summarize_event_possession(
                chunk,
                game_id=game_id,
                season=season,
                possession_number=possession_number,
                rotation=rotation,
            )
        )

    return pd.DataFrame(rows)


def _row_from_pbpstats_possession(
    possession: Any,
    *,
    game_id: str,
    season: str,
    possession_number: int,
) -> dict[str, Any]:
    lineup_id: str | None = None
    opponent_lineup_id: str | None = None
    stats = getattr(possession, "possession_stats", None) or []
    for stat in stats:
        if stat.get("lineup_id"):
            lineup_id = str(stat["lineup_id"])
        if stat.get("opponent_lineup_id"):
            opponent_lineup_id = str(stat["opponent_lineup_id"])

    return {
        "game_id": game_id,
        "season": season,
        "possession_number": possession_number,
        "period": int(getattr(possession, "period", 0) or 0),
        "start_clock": str(getattr(possession, "start_time", "")),
        "end_clock": str(getattr(possession, "end_time", "")),
        "offense_team_id": canonical_team_id(getattr(possession, "offense_team_id", None)),
        "possession_start_type": str(getattr(possession, "possession_start_type", "") or "") or None,
        "start_score_margin": getattr(possession, "start_score_margin", None),
        "lineup_id": lineup_id,
        "opponent_lineup_id": opponent_lineup_id,
        "has_lineup_on_floor": bool(lineup_id),
        "event_count": len(getattr(possession, "events", []) or []),
        "source": POSSESSION_SOURCE_PBPSTATS,
    }


def possessions_from_pbpstats_game(
    game_id: str,
    *,
    season: str,
    cache_dir: Path,
) -> pd.DataFrame | None:
    """
    Try pbpstats possession parsing for *game_id*.

    Returns ``None`` when pbpstats is unavailable or the stats API response
    format is incompatible (common with playbyplayv3-era payloads).
    """
    gid = canonical_game_id(game_id)
    if gid is None:
        return None
    try:
        from pbpstats.client import Client
    except ImportError:
        return None

    settings = {
        "dir": str(cache_dir),
        "Possessions": {"source": "web", "data_provider": "stats_nba"},
    }
    try:
        client = Client(settings)
        game = client.Game(gid)
        items = game.possessions.items
    except Exception:  # noqa: BLE001 — API / parsing failures fall back to events
        return None

    rows = [
        _row_from_pbpstats_possession(
            possession,
            game_id=gid,
            season=season,
            possession_number=index,
        )
        for index, possession in enumerate(items, start=1)
    ]
    return pd.DataFrame(rows)


def build_possession_rows_for_game(
    *,
    game_id: str,
    season: str,
    pbp_result: FetchResult,
    rotation_result: FetchResult | None,
    pbpstats_cache_dir: Path,
) -> pd.DataFrame:
    """
    Possession rows for one game: prefer pbpstats, else events + rotation fallback.
    """
    gid = canonical_game_id(game_id) or game_id
    pbpstats_df = possessions_from_pbpstats_game(
        gid,
        season=season,
        cache_dir=pbpstats_cache_dir,
    )
    if pbpstats_df is not None and not pbpstats_df.empty:
        return pbpstats_df

    pbp = playbyplay_frame(pbp_result)
    rotation = (
        gamerotation_stints(rotation_result)
        if rotation_result is not None
        else pd.DataFrame()
    )
    return possessions_from_events_and_rotation(
        pbp,
        rotation,
        game_id=gid,
        season=season,
    )


def build_possession_rows(
    *,
    season: str,
    game_ids: list[str],
    pbp_by_game: dict[str, FetchResult],
    rotation_by_game: dict[str, FetchResult],
    pbpstats_cache_dir: Path,
) -> pd.DataFrame:
    """Concatenate possession rows for all games in *game_ids*."""
    chunks: list[pd.DataFrame] = []
    for raw_gid in game_ids:
        gid = canonical_game_id(raw_gid)
        if gid is None:
            continue
        pbp_result = pbp_by_game.get(gid)
        if pbp_result is None:
            continue
        rotation_result = rotation_by_game.get(gid)
        chunk = build_possession_rows_for_game(
            game_id=gid,
            season=season,
            pbp_result=pbp_result,
            rotation_result=rotation_result,
            pbpstats_cache_dir=pbpstats_cache_dir,
        )
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)
