"""Combine submetrics, Option D ensemble, and calibrate overall_fit_percentile."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.impact_context import ImpactFitContext
from nba_fit.models.role_context import RoleFitContext
from nba_fit.scoring.constants import ENSEMBLE_DERIVED_NAMES, SUBMETRIC_NAMES
from nba_fit.scoring.ensemble import (
    calibrated_ensemble,
    component_contributions,
    extract_ensemble_components,
    uncertainty_band,
)
from nba_fit.scoring.submetrics import compute_all_submetrics


def raw_scores_to_percentiles(raw_scores: np.ndarray) -> np.ndarray:
    """
    Map raw weighted scores to percentiles in [0, 100] across all pairs.

    Uses average-rank percentile (equivalent to scipy.stats.rankdata / 100).
    """
    n = len(raw_scores)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([50.0])
    order = np.argsort(raw_scores, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    return 100.0 * (ranks - 0.5) / n


@dataclass
class FitIndexTable:
    """All player–team pairs for a season with submetrics, ensemble, and percentiles."""

    season: str
    pairs: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def is_calibrated(self) -> bool:
        return (
            not self.pairs.empty
            and "overall_fit_percentile" in self.pairs.columns
            and self.pairs["overall_fit_percentile"].notna().all()
            and "fit_uncertainty_low" in self.pairs.columns
            and "fit_uncertainty_high" in self.pairs.columns
        )


def build_fit_index_table(
    context: SeasonFitContext,
    *,
    role_context: RoleFitContext | None = None,
    impact_context: ImpactFitContext | None = None,
) -> FitIndexTable:
    """Score every player×team pair with Option D ensemble and calibrate percentiles."""
    if role_context is None:
        try:
            role_context = RoleFitContext.from_synthetic(context)
        except ValueError:
            role_context = None

    if impact_context is None and role_context is not None:
        try:
            impact_context = ImpactFitContext.from_synthetic(role_context)
        except ValueError:
            impact_context = None

    rows: list[dict[str, float | int | str]] = []
    for player in context.players.values():
        for team in context.teams.values():
            team_need = (
                role_context.team_needs.get(team.team_id) if role_context else None
            )
            sub = compute_all_submetrics(
                player,
                team,
                team_need=team_need,
                embeddings=role_context.embeddings if role_context else None,
                archetypes=role_context.archetypes if role_context else None,
                impact_context=impact_context,
            )
            if "team_need_fit" not in sub:
                sub["team_need_fit"] = 0.5
            if "lineup_impact_fit" not in sub:
                sub["lineup_impact_fit"] = 0.5

            extracted = extract_ensemble_components(sub, player=player)
            raw, components = calibrated_ensemble(
                extracted["profile_fit"],
                extracted["role_fit"],
                extracted["team_need_fit"],
                extracted["projected_impact"],
                extracted["replacement_upgrade"],
                extracted["risk_penalty"],
            )
            contribs = component_contributions(components)

            row: dict[str, float | int | str] = {
                "player_id": player.player_id,
                "team_id": team.team_id,
                "season": context.season,
                "raw_fit_score": raw,
            }
            for name in SUBMETRIC_NAMES:
                row[name] = sub[name]
            for name in ENSEMBLE_DERIVED_NAMES:
                row[f"ensemble_{name}"] = components[name]
            for name, value in contribs.items():
                row[f"contrib_{name}"] = value
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        raw_arr = df["raw_fit_score"].to_numpy(dtype=float)
        df["overall_fit_percentile"] = raw_scores_to_percentiles(raw_arr)

        lows: list[float] = []
        highs: list[float] = []
        for _, r in df.iterrows():
            comp = {name: float(r[f"ensemble_{name}"]) for name in ENSEMBLE_DERIVED_NAMES}
            pct = float(r["overall_fit_percentile"])
            low, high = uncertainty_band(pct, comp)
            lows.append(low)
            highs.append(high)
        df["fit_uncertainty_low"] = lows
        df["fit_uncertainty_high"] = highs

    return FitIndexTable(season=context.season, pairs=df)


def get_pair_row(
    table: FitIndexTable,
    player_id: int,
    team_id: int,
) -> pd.Series | None:
    """Lookup one calibrated pair row."""
    if table.pairs.empty:
        return None
    mask = (table.pairs["player_id"] == player_id) & (table.pairs["team_id"] == team_id)
    hits = table.pairs.loc[mask]
    if hits.empty:
        return None
    return hits.iloc[0]


def percentile_for_pair(
    table: FitIndexTable,
    player_id: int,
    team_id: int,
) -> float | None:
    row = get_pair_row(table, player_id, team_id)
    if row is None:
        return None
    return float(row["overall_fit_percentile"])
