"""Replacement-player benchmark: rank actual signing vs same-position pool."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_fit.features.constants import COL_PLAYER_ID, COL_ROLE_BUCKET
from nba_fit.features.player_vector import build_player_features, infer_role_bucket
from nba_fit.features.season_context import SeasonFitContext
from nba_fit.normalize.players import load_players_table
from nba_fit.scoring.fit_index import FitIndexTable


@dataclass(frozen=True)
class ReplacementBenchmarkRow:
    """One movement: percentile rank of signing within position pool at destination."""

    player_id: int
    to_team_id: int
    role_bucket: str
    pool_size: int
    signing_percentile: float
    pool_rank: int
    pool_rank_pct: float
    beat_median: bool


@dataclass
class ReplacementBenchmarkResult:
    """Aggregate replacement-player benchmark on movement rows."""

    season: str
    n_movements: int
    mean_pool_rank_pct: float | None
    median_beat_rate: float | None
    rows: list[ReplacementBenchmarkRow]

    def to_dict(self) -> dict[str, object]:
        return {
            "season": self.season,
            "n_movements": self.n_movements,
            "mean_pool_rank_pct": self.mean_pool_rank_pct,
            "median_beat_rate": self.median_beat_rate,
            "movements": [
                {
                    "player_id": r.player_id,
                    "to_team_id": r.to_team_id,
                    "role_bucket": r.role_bucket,
                    "pool_size": r.pool_size,
                    "signing_percentile": r.signing_percentile,
                    "pool_rank": r.pool_rank,
                    "pool_rank_pct": r.pool_rank_pct,
                    "beat_median": r.beat_median,
                }
                for r in self.rows
            ],
        }


def player_role_buckets(context: SeasonFitContext) -> dict[int, str]:
    """Map player_id → G/W/B role bucket for position-pool filtering."""
    if context.source == "interim":
        try:
            raw = load_players_table(context.season)
            feats = build_player_features(raw)
            if COL_ROLE_BUCKET in feats.columns:
                return {
                    int(pid): str(bucket)
                    for pid, bucket in zip(
                        feats[COL_PLAYER_ID], feats[COL_ROLE_BUCKET], strict=False
                    )
                }
        except FileNotFoundError:
            pass

    # Synthetic / API fallback: assign buckets from player index order.
    buckets = infer_role_bucket(
        pd.DataFrame({COL_PLAYER_ID: list(context.players.keys())})
    )
    return {
        int(pid): str(buckets.iloc[i])
        for i, pid in enumerate(context.players.keys())
    }


def rank_signing_vs_position_pool(
    movements: pd.DataFrame,
    context: SeasonFitContext,
    fit_table: FitIndexTable,
    *,
    role_buckets: dict[int, str] | None = None,
) -> ReplacementBenchmarkResult:
    """
    For each movement, rank the signing's fit percentile within same-position
    free-agent pool candidates at the destination team.

    Pool = all players sharing the signing's role bucket (G/W/B) in the season
    context, scored against the destination team.
    """
    buckets = role_buckets or player_role_buckets(context)
    rows: list[ReplacementBenchmarkRow] = []

    if movements.empty or fit_table.pairs.empty:
        return ReplacementBenchmarkResult(
            season=context.season,
            n_movements=0,
            mean_pool_rank_pct=None,
            median_beat_rate=None,
            rows=rows,
        )

    pairs = fit_table.pairs

    for _, move in movements.iterrows():
        player_id = int(move["player_id"])
        to_team = int(move["to_team_id"])
        bucket = buckets.get(player_id)
        if bucket is None:
            continue

        signing_hit = pairs[
            (pairs["player_id"] == player_id) & (pairs["team_id"] == to_team)
        ]
        if signing_hit.empty:
            continue
        signing_pct = float(signing_hit.iloc[0]["overall_fit_percentile"])

        pool_ids = [pid for pid, b in buckets.items() if b == bucket]
        pool = pairs[
            (pairs["player_id"].isin(pool_ids)) & (pairs["team_id"] == to_team)
        ]
        if pool.empty:
            continue

        pool_scores = pool["overall_fit_percentile"].to_numpy(dtype=float)
        pool_rank = int(1 + (pool_scores > signing_pct).sum())
        pool_rank_pct = float(pool_rank / len(pool_scores))
        median = float(np.median(pool_scores))

        rows.append(
            ReplacementBenchmarkRow(
                player_id=player_id,
                to_team_id=to_team,
                role_bucket=bucket,
                pool_size=len(pool_scores),
                signing_percentile=signing_pct,
                pool_rank=pool_rank,
                pool_rank_pct=pool_rank_pct,
                beat_median=signing_pct >= median,
            )
        )

    mean_rank_pct = float(np.mean([r.pool_rank_pct for r in rows])) if rows else None
    beat_rate = float(np.mean([r.beat_median for r in rows])) if rows else None

    return ReplacementBenchmarkResult(
        season=context.season,
        n_movements=len(rows),
        mean_pool_rank_pct=mean_rank_pct,
        median_beat_rate=beat_rate,
        rows=rows,
    )
