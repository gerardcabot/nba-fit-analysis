"""Typed contract for player–team fit card payloads (Option D)."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class FitCardUncertainty(TypedDict, total=False):
    """Percentile uncertainty band on overall fit."""

    low_percentile: float | None
    high_percentile: float | None


class FitCardComp(TypedDict, total=False):
    """Nearest-neighbor player comp in role embedding space."""

    player_id: int
    player: str
    archetype: str
    industry_role: NotRequired[str]
    distance: float


class FitCard(TypedDict, total=False):
    """Structured fit card returned by ``build_fit_card`` and the CLI/dashboard."""

    player_id: int
    team_id: int
    team: str
    season: str
    overall_fit_percentile: float | None
    raw_fit_score: float | None
    fit_uncertainty: FitCardUncertainty
    ensemble: dict[str, float]
    ensemble_contributions: dict[str, float]
    submetrics: dict[str, float]
    archetype: str | None
    industry_role: str | None
    archetype_id: int | None
    role_fit: float | None
    comps: list[FitCardComp]
    comps_note: str
    data_source: str
    lineup_synergy: dict[str, Any] | None
    projected_net_rating_delta: float | None


def fit_card_required_keys() -> tuple[str, ...]:
    """Keys that every fit card should expose for dashboard/CLI parity."""
    return (
        "player_id",
        "team_id",
        "team",
        "season",
        "overall_fit_percentile",
        "raw_fit_score",
        "fit_uncertainty",
        "ensemble",
        "ensemble_contributions",
        "submetrics",
        "archetype",
        "industry_role",
        "archetype_id",
        "role_fit",
        "comps",
        "comps_note",
        "data_source",
    )
