#!/usr/bin/env python3
"""Stacked bar of Option D ensemble component contributions for sample pairs."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

from nba_fit.features.season_context import DEMO_PLAYER_ID, DEMO_TEAM_ID, SeasonFitContext
from nba_fit.scoring.constants import ENSEMBLE_COMPONENT_NAMES
from nba_fit.scoring.ensemble import component_contributions, extract_ensemble_components
from nba_fit.scoring.fit_index import build_fit_index_table
from nba_fit.scoring.submetrics import compute_all_submetrics
from visual_tests._plot_utils import apply_plot_style, save_figure


def _sample_pairs(context: SeasonFitContext) -> list[tuple[int, int, str]]:
    """Demo player vs a few teams plus one extra cross-team pair."""
    team_ids = list(context.teams.keys())[:4]
    pairs: list[tuple[int, int, str]] = []
    for tid in team_ids:
        label = context.teams[tid].display_name or str(tid)
        pairs.append((DEMO_PLAYER_ID, tid, label[:12]))
    if len(team_ids) >= 2:
        other_pid = next(
            pid for pid in context.players if pid != DEMO_PLAYER_ID
        )
        pairs.append(
            (other_pid, DEMO_TEAM_ID, f"P{other_pid}→LAL"[:12])
        )
    return pairs


def main() -> int:
    context = SeasonFitContext.from_synthetic("2025-26", n_players=40)
    table = build_fit_index_table(context)
    samples = _sample_pairs(context)

    labels: list[str] = []
    contrib_matrix: list[list[float]] = []

    for player_id, team_id, label in samples:
        row = table.pairs.loc[
            (table.pairs["player_id"] == player_id)
            & (table.pairs["team_id"] == team_id)
        ]
        if row.empty:
            player = context.players[player_id]
            team = context.teams[team_id]
            sub = compute_all_submetrics(player, team)
            comps = extract_ensemble_components(sub, player=player)
            contribs = component_contributions(comps)
        else:
            contribs = {
                name: float(row.iloc[0].get(f"contrib_{name}", 0.0))
                for name in ENSEMBLE_COMPONENT_NAMES
            }
        labels.append(label)
        contrib_matrix.append([contribs[n] for n in ENSEMBLE_COMPONENT_NAMES])

    if not labels:
        print("No sample pairs to plot", file=sys.stderr)
        return 1

    data = np.array(contrib_matrix)
    apply_plot_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]
    for i, name in enumerate(ENSEMBLE_COMPONENT_NAMES):
        ax.bar(
            x,
            data[:, i],
            bottom=bottom,
            label=name.replace("_", " "),
            color=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
        )
        bottom += data[:, i]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Weighted contribution to raw ensemble")
    ax.set_title("Option D ensemble component contributions (sample player–team pairs)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    path = save_figure(fig, "13_ensemble_weights", subdir="option_d")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
