"""Fit card JSON/dict for a single player–team pair."""

from __future__ import annotations

from typing import Any

from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.role_context import RoleFitContext
from nba_fit.scoring.constants import SUBMETRIC_NAMES
from nba_fit.scoring.fit_index import FitIndexTable, get_pair_row
from nba_fit.scoring.role_fit import team_need_fit


def build_fit_card(
    player_id: int,
    team_id: int,
    table: FitIndexTable,
    context: SeasonFitContext,
    *,
    role_context: RoleFitContext | None = None,
) -> dict[str, Any]:
    """Structured fit card with submetrics, archetype label, role fit, and NN comps."""
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

    archetype_label: str | None = None
    archetype_id: int | None = None
    role_fit: float | None = None
    comps: list[dict[str, Any]] = []
    comps_note = "Nearest-neighbor comps in role embedding space (Option B)"

    rc = role_context
    if rc is None:
        try:
            rc = RoleFitContext.from_season(table.season, persist=False)
        except (FileNotFoundError, ValueError):
            rc = None

    if rc is not None:
        archetype_label = rc.archetypes.label_for(player_id)
        archetype_id = rc.archetypes.cluster_for(player_id)
        need = rc.team_needs.get(team_id)
        player = context.players.get(player_id)
        if need is not None and player is not None:
            role_fit = team_need_fit(
                player,
                need,
                embeddings=rc.embeddings,
                archetypes=rc.archetypes,
            )
        comps = rc.nearest_comps(player_id)
    else:
        comps_note = (
            "No role model for this season — run: "
            f"python -m nba_fit train-roles --season {table.season}"
        )

    return {
        "player_id": player_id,
        "team_id": team_id,
        "team": team_label,
        "season": table.season,
        "overall_fit_percentile": overall,
        "raw_fit_score": raw,
        "submetrics": submetrics,
        "archetype": archetype_label,
        "archetype_id": archetype_id,
        "role_fit": role_fit,
        "comps": comps,
        "comps_note": comps_note,
        "data_source": context.source,
    }


def fit_card_to_json(card: dict[str, Any], *, indent: int = 2) -> str:
    """Serialize fit card to JSON string."""
    import json

    return json.dumps(card, indent=indent, default=str)
