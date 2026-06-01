"""Resolve validation season when SOTA / interim data is on 2024-25."""

from __future__ import annotations

import os

from nba_fit.config.settings import DEFAULT_SEASON, get_settings
from nba_fit.normalize.players import interim_players_path

from visual_tests._constants import SOTA_VALIDATION_SEASON, VALIDATION_SEASON


def resolve_sota_season(explicit: str | None = None) -> str:
    """Prefer explicit season, then env, then 2024-25 when interim exists."""
    if explicit:
        return explicit
    env = os.environ.get("NBA_FIT_VALIDATION_SEASON", "").strip()
    if env:
        return env
    for candidate in (SOTA_VALIDATION_SEASON, VALIDATION_SEASON):
        if interim_players_path(candidate).exists():
            return candidate
    settings = get_settings()
    return settings.default_season or DEFAULT_SEASON
