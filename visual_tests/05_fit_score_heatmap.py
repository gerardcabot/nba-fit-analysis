#!/usr/bin/env python3
"""Heatmap: one player vs top-N team destinations by fit percentile (live interim)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import matplotlib.pyplot as plt
import seaborn as sns

from nba_fit.features.season_context import DEMO_PLAYER_ID, SeasonFitContext
from nba_fit.scoring.fit_index import build_fit_index_table
from nba_fit.scoring.ranker import FitRanker
from visual_tests._plot_utils import apply_plot_style, save_figure

DEFAULT_SEASON = "2024-25"
TOP_TEAMS = 10


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        help="Season label for interim tables (default: 2024-25)",
    )
    parser.add_argument(
        "--player-id",
        type=int,
        default=DEMO_PLAYER_ID,
        help="NBA player id for destination heatmap (default: 2544)",
    )
    parser.add_argument(
        "--top-teams",
        type=int,
        default=TOP_TEAMS,
        help="Number of top destination teams to plot",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    season = args.season
    player_id = args.player_id
    top_teams = args.top_teams

    context = SeasonFitContext.build(
        season,
        prefer_interim=True,
        prefer_api=False,
    )
    if context.source != "interim":
        print(
            f"Expected interim context for {season}; got source={context.source!r}.",
            file=sys.stderr,
        )
        return 1

    table = build_fit_index_table(context)
    ranker = FitRanker(context=context, table=table)
    dest = ranker.rank_destinations_for_player(player_id, top_n=top_teams)

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
    ax.set_title(
        f"Player {player_id} — submetrics vs top {top_teams} destinations "
        f"({season} interim)"
    )
    ax.set_xlabel("Submetric")
    ax.set_ylabel("Team")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()

    path = save_figure(fig, "05_fit_score_heatmap", subdir="fit")
    print(f"Wrote {path}")
    for _, row in dest.iterrows():
        pct = float(row["overall_fit_percentile"])
        print(f"  #{int(row['rank'])} {row['team']}: {pct:.2f} percentile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
