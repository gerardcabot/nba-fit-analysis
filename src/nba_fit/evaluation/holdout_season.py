"""Held-out season smoke: train features on season N, score pairs on season N+1."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.impact_context import ImpactFitContext
from nba_fit.models.role_context import RoleFitContext
from nba_fit.scoring.fit_index import FitIndexTable, build_fit_index_table


def next_season(season: str) -> str:
    """Advance NBA season label (e.g. ``2024-25`` → ``2025-26``)."""
    start_year = int(season.split("-")[0])
    return f"{start_year + 1}-{str(start_year + 2)[-2:]}"


@dataclass
class HoldoutSeasonResult:
    """Summary of train-N / predict-N+1 holdout smoke."""

    train_season: str
    predict_season: str
    train_pairs: int = 0
    predict_pairs: int = 0
    overlap_players: int = 0
    mean_predict_percentile: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "train_season": self.train_season,
            "predict_season": self.predict_season,
            "train_pairs": self.train_pairs,
            "predict_pairs": self.predict_pairs,
            "overlap_players": self.overlap_players,
            "mean_predict_percentile": self.mean_predict_percentile,
            "note": self.note,
        }


def run_holdout_season_smoke(
    train_season: str,
    *,
    synthetic: bool = True,
) -> HoldoutSeasonResult:
    """
    Build fit tables on season N (train) and N+1 (predict) using synthetic data.

    Uses train-season role/impact contexts when scoring the predict season so the
  pipeline exercises cross-season artifact wiring without live API calls.
    """
    predict_season = next_season(train_season)
    train_ctx = SeasonFitContext.from_synthetic(train_season, n_players=40)
    predict_ctx = SeasonFitContext.from_synthetic(predict_season, n_players=40)

    role_ctx = RoleFitContext.from_synthetic(train_ctx)
    impact_ctx = ImpactFitContext.from_synthetic(role_ctx)

    train_table = build_fit_index_table(
        train_ctx, role_context=role_ctx, impact_context=impact_ctx
    )
    predict_table = build_fit_index_table(
        predict_ctx, role_context=role_ctx, impact_context=impact_ctx
    )

    train_players = set(train_ctx.players.keys())
    predict_players = set(predict_ctx.players.keys())
    overlap = train_players & predict_players

    mean_pct: float | None = None
    if not predict_table.pairs.empty:
        mean_pct = float(predict_table.pairs["overall_fit_percentile"].mean())

    return HoldoutSeasonResult(
        train_season=train_season,
        predict_season=predict_season,
        train_pairs=len(train_table.pairs),
        predict_pairs=len(predict_table.pairs),
        overlap_players=len(overlap),
        mean_predict_percentile=mean_pct,
        note=(
            "Synthetic holdout smoke — train-season role/impact artifacts applied "
            "to predict-season vectors."
        ),
    )


def holdout_rank_stability(
    train_table: FitIndexTable,
    predict_table: FitIndexTable,
    *,
    top_n: int = 5,
) -> float | None:
    """
    Mean Spearman-like rank correlation for overlapping players' top teams.

    Returns ``None`` when insufficient overlap.
    """
    if train_table.pairs.empty or predict_table.pairs.empty:
        return None

    train_players = set(train_table.pairs["player_id"].unique())
    predict_players = set(predict_table.pairs["player_id"].unique())
    shared = train_players & predict_players
    if not shared:
        return None

    corrs: list[float] = []
    for pid in shared:
        t_sub = (
            train_table.pairs.loc[train_table.pairs["player_id"] == pid]
            .nlargest(top_n, "overall_fit_percentile")["team_id"]
            .tolist()
        )
        p_sub = (
            predict_table.pairs.loc[predict_table.pairs["player_id"] == pid]
            .nlargest(top_n, "overall_fit_percentile")["team_id"]
            .tolist()
        )
        if not t_sub or not p_sub:
            continue
        overlap = len(set(t_sub) & set(p_sub))
        corrs.append(overlap / top_n)

    if not corrs:
        return None
    return float(np.mean(corrs))
