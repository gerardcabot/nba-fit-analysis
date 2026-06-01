"""Benchmark possession/lineup RAPM against NBA estimated net rating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nba_fit.evaluation.calibration_metrics import spearman_rank_corr
from nba_fit.features.constants import COL_PLAYER_ID
from nba_fit.models.rapm import RapmArtifacts, load_rapm

MIN_BENCHMARK_PLAYERS = 100


@dataclass(frozen=True)
class RapmBenchmarkResult:
    """Spearman correlation between ``net_rapm`` and ``E_NET_RATING``."""

    season: str
    n_overlap: int
    spearman_rho: float | None
    passed: bool
    e_net_column: str
    degenerate_rapm: bool
    net_rapm_std: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "n_overlap": self.n_overlap,
            "spearman_net_rapm_vs_e_net_rating": self.spearman_rho,
            "passed_min_players": self.passed,
            "min_players_required": MIN_BENCHMARK_PLAYERS,
            "e_net_column": self.e_net_column,
            "degenerate_rapm": self.degenerate_rapm,
            "net_rapm_std": self.net_rapm_std,
            "note": self.note,
        }


def _e_net_rating_column(players: pd.DataFrame) -> str | None:
    for col in ("E_NET_RATING", "NET_RATING", "PLUS_MINUS"):
        if col in players.columns and players[col].notna().any():
            return col
    return None


def benchmark_rapm_vs_estimated_net(
    season: str,
    *,
    rapm: RapmArtifacts | None = None,
    players: pd.DataFrame | None = None,
    min_players: int = MIN_BENCHMARK_PLAYERS,
) -> RapmBenchmarkResult:
    """
    Spearman rank correlation of ``net_rapm`` vs NBA ``E_NET_RATING`` on overlapping players.

    Loads artifacts and interim players when not supplied.
    """
    if rapm is None:
        rapm = load_rapm(season)

    net_std = float(rapm.metadata.get("net_rapm_std", np.std(rapm.net_rapm)))
    degenerate = bool(rapm.degenerate or net_std == 0.0)

    if players is None:
        from nba_fit.normalize.players import load_players_table

        players = load_players_table(season)

    e_col = _e_net_rating_column(players)
    if e_col is None:
        return RapmBenchmarkResult(
            season=season,
            n_overlap=0,
            spearman_rho=None,
            passed=False,
            e_net_column="",
            degenerate_rapm=degenerate,
            net_rapm_std=net_std,
            note="No E_NET_RATING / NET_RATING column in players table",
        )

    pid_col = COL_PLAYER_ID if COL_PLAYER_ID in players.columns else "player_id"
    if pid_col not in players.columns and "PLAYER_ID" in players.columns:
        pid_col = "PLAYER_ID"

    est = players[[pid_col, e_col]].dropna(subset=[e_col]).copy()
    est[pid_col] = est[pid_col].astype(int)
    est = est.rename(columns={pid_col: "player_id", e_col: "e_net_rating"})
    est = est.groupby("player_id", as_index=False)["e_net_rating"].mean()

    rapm_df = pd.DataFrame(
        {
            "player_id": rapm.player_ids.astype(int),
            "net_rapm": rapm.net_rapm.astype(float),
        }
    )
    merged = rapm_df.merge(est, on="player_id", how="inner")
    n = len(merged)

    if n < 2 or degenerate:
        rho = None
    else:
        rho = spearman_rank_corr(
            merged["e_net_rating"].to_numpy(dtype=float),
            merged["net_rapm"].to_numpy(dtype=float),
        )

    return RapmBenchmarkResult(
        season=season,
        n_overlap=n,
        spearman_rho=rho,
        passed=n >= min_players and rho is not None and not degenerate,
        e_net_column=e_col,
        degenerate_rapm=degenerate,
        net_rapm_std=net_std,
        note="" if n >= min_players else f"overlap {n} < {min_players}",
    )
