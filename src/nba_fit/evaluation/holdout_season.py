"""Held-out season evaluation: train features on season N, score pairs on season N+1."""

from __future__ import annotations

from dataclasses import dataclass

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
    """Summary of train-N / predict-N+1 holdout."""

    train_season: str
    predict_season: str
    train_pairs: int = 0
    predict_pairs: int = 0
    overlap_players: int = 0
    mean_predict_percentile: float | None = None
    rank_stability: float | None = None
    data_source: str = "synthetic"
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "train_season": self.train_season,
            "predict_season": self.predict_season,
            "train_pairs": self.train_pairs,
            "predict_pairs": self.predict_pairs,
            "overlap_players": self.overlap_players,
            "mean_predict_percentile": self.mean_predict_percentile,
            "rank_stability": self.rank_stability,
            "data_source": self.data_source,
            "note": self.note,
        }


def _load_role_context(season: str) -> RoleFitContext | None:
    try:
        return RoleFitContext.from_season(
            season,
            prefer_interim=True,
            prefer_api=False,
            synthetic=False,
            persist=False,
        )
    except (FileNotFoundError, ValueError):
        return None


def _load_impact_context(season: str) -> ImpactFitContext | None:
    try:
        return ImpactFitContext.from_season(
            season,
            prefer_interim=True,
            synthetic=False,
            persist=False,
        )
    except (FileNotFoundError, ValueError):
        return None


def _build_context(season: str, *, prefer_interim: bool) -> SeasonFitContext:
    if prefer_interim:
        try:
            return SeasonFitContext.from_interim(season)
        except FileNotFoundError:
            pass
    return SeasonFitContext.build(season, prefer_interim=False, prefer_api=False)


def run_holdout_season(
    train_season: str,
    *,
    prefer_interim: bool = True,
    synthetic_fallback: bool = True,
) -> HoldoutSeasonResult:
    """
    Build fit tables on season N (train) and N+1 (predict).

    Uses train-season role/impact artifacts when scoring the predict season.
    Falls back to synthetic smoke when interim partitions are unavailable.
    """
    predict_season = next_season(train_season)
    if prefer_interim:
        try:
            train_ctx = _build_context(train_season, prefer_interim=True)
            predict_ctx = _build_context(predict_season, prefer_interim=True)
            if train_ctx.source != "interim" or predict_ctx.source != "interim":
                raise FileNotFoundError("interim partitions missing for holdout seasons")

            role_ctx = _load_role_context(train_season)
            impact_ctx = _load_impact_context(train_season)
            if role_ctx is None:
                role_ctx = RoleFitContext.from_synthetic(train_ctx)
            if impact_ctx is None and role_ctx is not None:
                impact_ctx = ImpactFitContext.from_synthetic(role_ctx)

            train_table = build_fit_index_table(
                train_ctx, role_context=role_ctx, impact_context=impact_ctx
            )
            predict_table = build_fit_index_table(
                predict_ctx, role_context=role_ctx, impact_context=impact_ctx
            )
            return _summarize_holdout(
                train_season,
                predict_season,
                train_table,
                predict_table,
                data_source="interim",
                note=(
                    "Interim holdout — train-season role/impact artifacts applied "
                    "to predict-season vectors."
                ),
            )
        except FileNotFoundError:
            pass

    if synthetic_fallback:
        return run_holdout_season_smoke(train_season, synthetic=True)
    raise FileNotFoundError(
        f"No interim data for holdout train={train_season} predict={predict_season}"
    )


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

    return _summarize_holdout(
        train_season,
        predict_season,
        train_table,
        predict_table,
        data_source="synthetic",
        note=(
            "Synthetic holdout smoke — train-season role/impact artifacts applied "
            "to predict-season vectors."
        ),
    )


def _summarize_holdout(
    train_season: str,
    predict_season: str,
    train_table: FitIndexTable,
    predict_table: FitIndexTable,
    *,
    data_source: str,
    note: str,
) -> HoldoutSeasonResult:
    train_players = set(train_table.pairs["player_id"].unique()) if not train_table.pairs.empty else set()
    predict_players = (
        set(predict_table.pairs["player_id"].unique()) if not predict_table.pairs.empty else set()
    )
    overlap = train_players & predict_players

    mean_pct: float | None = None
    if not predict_table.pairs.empty:
        mean_pct = float(predict_table.pairs["overall_fit_percentile"].mean())

    stability = holdout_rank_stability(train_table, predict_table)

    return HoldoutSeasonResult(
        train_season=train_season,
        predict_season=predict_season,
        train_pairs=len(train_table.pairs),
        predict_pairs=len(predict_table.pairs),
        overlap_players=len(overlap),
        mean_predict_percentile=mean_pct,
        rank_stability=stability,
        data_source=data_source,
        note=note,
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
