"""Fit card JSON/dict for a single player–team pair."""

from __future__ import annotations

from typing import Any

import pandas as pd

from nba_fit.features.season_context import SeasonFitContext
from nba_fit.scoring.constants import SUBMETRIC_NAMES
from nba_fit.scoring.fit_index import FitIndexTable, get_pair_row


def build_fit_card(
    player_id: int,
    team_id: int,
    table: FitIndexTable,
    context: SeasonFitContext,
) -> dict[str, Any]:
    """Structured fit card with submetrics and comps placeholder."""
    row = get_pair_row(table, player_id, team_id)
    team = context.teams.get(team_id)
    team_label = team.display_name if team else str(team_id)

    submetrics: dict[str, float] = {}
    overall: float | None = None
    raw: float | None = None
    if row is not None:
        overall = float(row["overall_fit_percentile"])
        raw = float(row["raw_fit_score"])
        for name in SUBMETRIC_NAMES:
            submetrics[name] = float(row[name])

    return {
        "player_id": player_id,
        "team_id": team_id,
        "team": team_label,
        "season": table.season,
        "overall_fit_percentile": overall,
        "raw_fit_score": raw,
        "submetrics": submetrics,
        "comps": [],
        "comps_note": "Historical player comps (nearest neighbors) — Phase 2+",
        "data_source": context.source,
    }


def fit_card_to_json(card: dict[str, Any], *, indent: int = 2) -> str:
    """Serialize fit card to JSON string."""
    import json

    return json.dumps(card, indent=indent, default=str)
