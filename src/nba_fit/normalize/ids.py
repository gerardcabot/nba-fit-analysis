"""Canonical entity ID helpers for joins across nba_api tables."""

from __future__ import annotations

import re
from typing import Any

# NBA stats API uses 10-digit zero-padded strings for game IDs in many endpoints.
_GAME_ID_RE = re.compile(r"^0?\d{10}$")
_PLAYER_ID_RE = re.compile(r"^\d{1,10}$")
_TEAM_ID_RE = re.compile(r"^1\d{9}$")  # NBA team IDs are 16106127xx


def canonical_player_id(value: Any) -> int | None:
    """Normalize PLAYER_ID / person_id to int."""
    if value is None or (isinstance(value, float) and str(value) == "nan"):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    return pid


def canonical_team_id(value: Any) -> int | None:
    """Normalize TEAM_ID to int (1610612747 format)."""
    if value is None or (isinstance(value, float) and str(value) == "nan"):
        return None
    try:
        tid = int(value)
    except (TypeError, ValueError):
        return None
    if tid <= 0:
        return None
    return tid


def canonical_game_id(value: Any) -> str | None:
    """
    Normalize GAME_ID to 10-digit string (e.g. 0022501198).

    Regular-season NBA game IDs start with 002; playoffs 004; preseason 001.
    """
    if value is None or (isinstance(value, float) and str(value) == "nan"):
        return None
    text = str(value).strip()
    if not text:
        return None
    # Accept bare numeric IDs
    if text.isdigit():
        text = text.zfill(10)
    if not _GAME_ID_RE.match(text):
        return None
    return text.zfill(10) if len(text) < 10 else text


def is_valid_player_id(value: Any) -> bool:
    pid = canonical_player_id(value)
    if pid is None:
        return False
    return bool(_PLAYER_ID_RE.match(str(pid)))


def is_valid_team_id(value: Any) -> bool:
    tid = canonical_team_id(value)
    if tid is None:
        return False
    return bool(_TEAM_ID_RE.match(str(tid)))


def is_valid_game_id(value: Any) -> bool:
    return canonical_game_id(value) is not None
