#!/usr/bin/env python3
"""Heatmap: one player vs top-N team destinations by fit percentile."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nba_fit.features.season_context import DEMO_PLAYER_ID, SeasonFitContext
from nba_fit.scoring.fit_index import build_fit_index_table
from nba_fit.scoring.ranker import FitRanker
from visual_tests._plot_utils import apply_plot_style, save_figure

TOP_TEAMS = 10


def main() -> int:
    context = SeasonFitContext.from_synthetic("2025-26", n_players=80)
    table = build_fit_index_table(context)
    ranker = FitRanker(context=context, table=table)
    dest = ranker.rank_destinations_for_player(DEMO_PLAYER_ID, top_n=TOP_TEAMS)

    if dest.empty:
        print("No destination rankings to plot.", file=sys.stderr)
        return 1

    submetric_cols = [
        c
        for c in dest.columns
        if c
        not in (
            "rank",
            "team_id",
            "team",
            "overall_fit_percentile",
            "raw_fit_score",
            "player_id",
            "season",
        )
    ]
    matrix = dest[submetric_cols].to_numpy(dtype=float)
    teams = dest["team"].tolist()

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        xticklabels=submetric_cols,
        yticklabels=teams,
        ax=ax,
        cbar_kws={"label": "Submetric score (0–1)"},
    )
    ax.set_title(f"Player {DEMO_PLAYER_ID} — submetrics vs top {TOP_TEAMS} destinations")
    ax.set_xlabel("Submetric")
    ax.set_ylabel("Team")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()

    path = save_figure(fig, "05_fit_score_heatmap", subdir="fit")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
