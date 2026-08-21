"""Fit card JSON/dict for a single player–team pair."""

from __future__ import annotations

from typing import Any

from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.impact_context import ImpactFitContext
from nba_fit.models.role_context import RoleFitContext
from nba_fit.scoring.constants import ENSEMBLE_DERIVED_NAMES, SUBMETRIC_NAMES
from nba_fit.scoring.ensemble import component_contributions, uncertainty_band
from nba_fit.scoring.fit_index import FitIndexTable, get_pair_row
from nba_fit.scoring.lineup_sim import run_lineup_sim
from nba_fit.scoring.role_fit import team_need_fit


def build_fit_card(
    player_id: int,
    team_id: int,
    table: FitIndexTable,
    context: SeasonFitContext,
    *,
    role_context: RoleFitContext | None = None,
    impact_context: ImpactFitContext | None = None,
    include_lineup_synergy: bool = True,
    lineup_synthetic: bool = False,
) -> dict[str, Any]:
    """Structured fit card with submetrics, role fit, lineup synergy, and NN comps."""
    row = get_pair_row(table, player_id, team_id)
    team = context.teams.get(team_id)
    team_label = team.display_name if team else str(team_id)

    submetrics: dict[str, float] = {}
    ensemble: dict[str, float] = {}
    contributions: dict[str, float] = {}
    overall: float | None = None
    raw: float | None = None
    uncertainty_low: float | None = None
    uncertainty_high: float | None = None
    components_degraded: list[str] = []
    if row is not None:
        overall = float(row["overall_fit_percentile"])
        raw = float(row["raw_fit_score"])
        for name in SUBMETRIC_NAMES:
            submetrics[name] = float(row[name])
        for name in ENSEMBLE_DERIVED_NAMES:
            col = f"ensemble_{name}"
            if col in row.index:
                ensemble[name] = float(row[col])
        contrib_cols = [c for c in row.index if str(c).startswith("contrib_")]
        for col in contrib_cols:
            contributions[str(col).replace("contrib_", "")] = float(row[col])
        if "fit_uncertainty_low" in row.index:
            uncertainty_low = float(row["fit_uncertainty_low"])
            uncertainty_high = float(row["fit_uncertainty_high"])
        elif ensemble:
            low, high = uncertainty_band(overall, ensemble)
            uncertainty_low, uncertainty_high = low, high
            contributions = contributions or component_contributions(ensemble)
        if "components_degraded" in row.index and isinstance(
            row["components_degraded"], list
        ):
            components_degraded = list(row["components_degraded"])

    archetype_label: str | None = None
    archetype_id: int | None = None
    industry_role: str | None = None
    soft_role_display: str | None = None
    industry_role_probs: dict[str, float] | None = None
    team_need_fit_value: float | None = submetrics.get("team_need_fit")
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
        industry_role = rc.archetypes.industry_role_for(player_id)
        industry_role_probs = rc.archetypes.industry_probs_for(player_id)
        soft_role_display = rc.archetypes.soft_role_display_for(player_id)
        need = rc.team_needs.get(team_id)
        player = context.players.get(player_id)
        if need is not None and player is not None:
            team_need_fit_value = team_need_fit(
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

    lineup_synergy: dict[str, Any] | None = None
    projected_net_rating_delta: float | None = None
    if include_lineup_synergy:
        sim = run_lineup_sim(
            player_id,
            team_id,
            table.season,
            prefer_interim=context.source != "synthetic",
            prefer_api=context.source == "api",
            synthetic=lineup_synthetic or context.source == "synthetic",
            impact_context=impact_context,
        )
        lineup_synergy = sim.lineup_synergy_block()
        projected_net_rating_delta = sim.projected_net_rating_delta
        if projected_net_rating_delta is not None and lineup_synergy is not None:
            lineup_synergy["headline"] = {
                "projected_net_rating_delta": projected_net_rating_delta,
                "units": "pts/100 poss",
            }

    fallbacks = sorted(set(components_degraded))

    card: dict[str, Any] = {
        "player_id": player_id,
        "team_id": team_id,
        "team": team_label,
        "season": table.season,
        "overall_fit_percentile": overall,
        "raw_fit_score": raw,
        "fit_uncertainty": {
            "low_percentile": uncertainty_low,
            "high_percentile": uncertainty_high,
        },
        "ensemble": ensemble,
        "ensemble_contributions": contributions,
        "submetrics": submetrics,
        "components_degraded": fallbacks,
        "fallbacks": fallbacks,
        "archetype": archetype_label,
        "industry_role": industry_role,
        "soft_role_display": soft_role_display,
        "industry_role_probs": industry_role_probs,
        "archetype_id": archetype_id,
        "team_need_fit": team_need_fit_value,
        "role_fit": team_need_fit_value,
        "comps": comps,
        "comps_note": comps_note,
        "data_source": context.source,
    }
    if include_lineup_synergy:
        card["lineup_synergy"] = lineup_synergy
        card["projected_net_rating_delta"] = projected_net_rating_delta
    return card


def fit_card_to_json(card: dict[str, Any], *, indent: int = 2) -> str:
    """Serialize fit card to JSON string."""
    import json

    return json.dumps(card, indent=indent, default=str)
