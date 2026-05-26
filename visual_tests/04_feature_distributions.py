#!/usr/bin/env python3
"""Visual test: distributions of interpretable feature groups (synthetic leaguedash-shaped data)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nba_fit.features.player_vector import PLAYER_FEATURE_GROUPS, build_player_features
from nba_fit.features.scaling import scale_player_features
from visual_tests._plot_utils import new_figure, save_figure

OUTPUT_STEM = "04_feature_distributions"
RNG_SEED = 42
N_PLAYERS = 120


def _synthetic_league_df(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a league-shaped table mimicking merged leaguedash columns."""
    usg = rng.uniform(0.12, 0.34, size=n)
    min_pg = rng.uniform(8, 36, size=n)
    gp = rng.integers(15, 82, size=n)
    fga = np.clip(min_pg * usg * 2.2 + rng.normal(0, 2, size=n), 3, 28)
    fg3a = fga * rng.uniform(0.15, 0.55, size=n)
    total_zone = fga * 1.05
    rim = total_zone * rng.uniform(0.25, 0.55, size=n)

    return pd.DataFrame(
        {
            "PLAYER_ID": np.arange(1, n + 1),
            "TEAM_ID": rng.integers(1, 31, size=n),
            "PLAYER_POSITION": rng.choice(
                ["Guard", "Forward", "Center", "G-F", "F-C"], size=n
            ),
            "MIN": min_pg,
            "GP": gp,
            "FGA": fga,
            "FG3A": fg3a,
            "PTS": fga * rng.uniform(0.9, 1.2, size=n),
            "AST": rng.uniform(0.5, 9, size=n),
            "TOV": rng.uniform(0.5, 4.5, size=n),
            "USG_PCT": usg,
            "PACE": rng.normal(100, 3, size=n),
            "TS_PCT": rng.uniform(0.48, 0.66, size=n),
            "EFG_PCT": rng.uniform(0.42, 0.62, size=n),
            "OFF_RATING": rng.normal(112, 8, size=n),
            "DEF_RATING": rng.normal(112, 8, size=n),
            "NET_RATING": rng.normal(0, 6, size=n),
            "AST_PCT": rng.uniform(0.05, 0.40, size=n),
            "AST_RATIO": rng.uniform(0.8, 2.5, size=n),
            "OREB_PCT": rng.uniform(0.01, 0.14, size=n),
            "DREB_PCT": rng.uniform(0.08, 0.32, size=n),
            "REB_PCT": rng.uniform(0.06, 0.22, size=n),
            "STL": rng.uniform(0.2, 2.2, size=n),
            "BLK": rng.uniform(0.0, 2.5, size=n),
            "OREB": rng.uniform(0.2, 3.0, size=n),
            "DREB": rng.uniform(1.0, 8.0, size=n),
            "REB": rng.uniform(1.5, 10.0, size=n),
            "PLAYER_HEIGHT_INCHES": rng.normal(78, 3.5, size=n),
            "PLAYER_WEIGHT": rng.normal(215, 25, size=n),
            "AGE": rng.integers(19, 38, size=n),
            "Restricted Area | FGA": rim * 0.55,
            "In The Paint (Non-RA) | FGA": rim * 0.45,
            "Mid-Range | FGA": total_zone * rng.uniform(0.05, 0.18, size=n),
            "Left Corner 3 | FGA": fg3a * 0.08,
            "Right Corner 3 | FGA": fg3a * 0.08,
            "Above the Break 3 | FGA": fg3a * 0.84,
        }
    )


def _pick_example_column(features: pd.DataFrame, group: str) -> str | None:
    prefix = f"pf_{group}__"
    cols = [c for c in features.columns if c.startswith(f"z_{prefix}")]
    if not cols:
        cols = [c for c in features.columns if c.startswith(prefix)]
    return cols[0] if cols else None


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)
    raw = _synthetic_league_df(N_PLAYERS, rng)
    features = build_player_features(raw)
    scaled = scale_player_features(features)

    fig, axes = new_figure(width_in=14, height_in=7)
    if not isinstance(axes, np.ndarray):
        axes_list = [axes]
    else:
        axes_list = list(np.atleast_1d(axes).flatten())

    # new_figure returns single Axes; create a grid for eight groups
    plt.close(fig)
    fig, axes_grid = plt.subplots(2, 4, figsize=(14, 7))
    axes_flat = axes_grid.flatten()

    for ax, group in zip(axes_flat, PLAYER_FEATURE_GROUPS, strict=True):
        col = _pick_example_column(scaled, group)
        if col is None:
            ax.set_visible(False)
            continue
        series = scaled[col].dropna()
        ax.hist(series, bins=20, color="#4c72b0", edgecolor="black", alpha=0.85)
        ax.set_title(group.replace("_", " ").title())
        ax.set_xlabel(col.replace("z_pf_", "").replace("__", " / "))

    fig.suptitle("Player feature group distributions (robust z-scored by role bucket)")
    fig.tight_layout()
    path = save_figure(fig, OUTPUT_STEM, subdir="features")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
