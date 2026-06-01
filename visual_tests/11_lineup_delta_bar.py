#!/usr/bin/env python3
"""Bar chart of top lineup projected net-rating deltas (Option C demo player)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from visual_tests._constants import VALIDATION_SEASON
from nba_fit.features.season_context import DEMO_PLAYER_ID, DEMO_TEAM_ID
from nba_fit.scoring.lineup_sim import run_lineup_sim
from visual_tests._plot_utils import apply_plot_style, save_figure


def main() -> int:
    try:
        result = run_lineup_sim(
            DEMO_PLAYER_ID,
            DEMO_TEAM_ID,
            VALIDATION_SEASON,
            top_n=5,
            prefer_interim=True,
            prefer_api=False,
            synthetic=False,
        )
    except Exception:  # noqa: BLE001
        result = run_lineup_sim(
            DEMO_PLAYER_ID,
            DEMO_TEAM_ID,
            top_n=5,
            synthetic=True,
        )
    if not result.top_lineups:
        print("No lineup projections to plot", file=sys.stderr)
        return 1

    labels = [
        f"#{i + 1}"
        for i in range(len(result.top_lineups))
    ]
    deltas = [u.projected_net_rating_delta for u in result.top_lineups]

    apply_plot_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
    ax.bar(labels, deltas, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Top projected five-man unit")
    ax.set_ylabel("Projected net rating delta (pts / 100 poss)")
    agg = result.projected_net_rating_delta
    agg_txt = f"{agg:+.2f}" if agg is not None else "n/a"
    ax.set_title(
        f"Lineup delta bar — player {DEMO_PLAYER_ID} → team {DEMO_TEAM_ID}\n"
        f"Minutes-weighted aggregate: {agg_txt}"
    )
    fig.tight_layout()

    path = save_figure(fig, "11_lineup_delta_bar", subdir="option_c")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
