#!/usr/bin/env python3
"""Histogram of player RAPM (net) estimates for a season."""

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

from nba_fit.config.settings import DEFAULT_SEASON
from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.impact_context import ImpactFitContext
from nba_fit.models.rapm import load_rapm
from nba_fit.models.role_context import RoleFitContext
from visual_tests._plot_utils import apply_plot_style, save_figure


def main() -> int:
    season = DEFAULT_SEASON
    source = "trained"
    try:
        rapm = load_rapm(season)
    except (FileNotFoundError, OSError):
        context = SeasonFitContext.from_synthetic(season, n_players=80)
        role = RoleFitContext.from_synthetic(context)
        impact = ImpactFitContext.from_synthetic(role)
        rapm = impact.rapm
        source = "synthetic"

    net = rapm.net_rapm
    low = rapm.low_sample_flag

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    stable = net[~low]
    if len(stable):
        ax.hist(stable, bins=20, alpha=0.75, color="#2a6f97", label="stable sample")
    if low.any():
        ax.hist(net[low], bins=12, alpha=0.45, color="#e76f51", label="low sample")
    ax.axvline(0.0, color="0.3", linestyle="--", linewidth=1)
    ax.set_xlabel("Net RAPM (ORAPM + DRAPM, pts/100)")
    ax.set_ylabel("Player count")
    ax.set_title(f"Player RAPM distribution — {season} ({source})")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "10_rapm_distribution")
    print(f"Saved histogram for {len(net)} players (mean={net.mean():.2f}, std={net.std():.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
