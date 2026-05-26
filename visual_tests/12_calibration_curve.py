#!/usr/bin/env python3
"""Reliability diagram: calibrated fit percentiles vs post-move outcomes."""

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

from nba_fit.evaluation.movement_backtest import run_movement_backtest
from nba_fit.features.season_context import SeasonFitContext
from nba_fit.data.fetchers.transactions import synthetic_movements
from visual_tests._plot_utils import apply_plot_style, save_figure


def _reliability_bins(
    predicted: np.ndarray,
    observed: np.ndarray,
    n_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean observed outcome per predicted percentile bin."""
    order = np.argsort(predicted)
    pred = predicted[order]
    obs = observed[order]
    edges = np.linspace(0, len(pred), n_bins + 1, dtype=int)
    bin_pred: list[float] = []
    bin_obs: list[float] = []
    counts: list[int] = []
    for i in range(n_bins):
        sl = slice(edges[i], edges[i + 1])
        if sl.start >= sl.stop:
            continue
        bin_pred.append(float(np.mean(pred[sl])))
        bin_obs.append(float(np.mean(obs[sl])))
        counts.append(sl.stop - sl.start)
    return np.array(bin_pred), np.array(bin_obs), np.array(counts)


def main() -> int:
    season = "2025-26"
    context = SeasonFitContext.from_synthetic(season, n_players=60)
    player_ids = tuple(list(context.players.keys())[:8])
    movements = synthetic_movements(season, player_ids=player_ids, n_moves=12)
    result = run_movement_backtest(context, movements, calibrate=True)
    cal = result.calibration_frame()

    if cal.empty or len(cal) < 3:
        print("Not enough movement rows for reliability plot")
        return 1

    pred = cal["calibrated_fit_percentile"].to_numpy(dtype=float)
    obs = cal["post_move_outcome"].to_numpy(dtype=float) * 100.0

    bin_pred, bin_obs, counts = _reliability_bins(pred, obs, n_bins=5)

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 100], [0, 100], "--", color="0.45", linewidth=1, label="perfect calibration")
    ax.scatter(bin_pred, bin_obs, s=counts * 8 + 40, color="#2a6f97", zorder=3, label="binned mean")
    ax.plot(bin_pred, bin_obs, "-", color="#e76f51", linewidth=2, alpha=0.85)
    ax.set_xlabel("Predicted fit percentile (calibrated)")
    ax.set_ylabel("Mean post-move outcome (×100)")
    ax.set_title(f"Movement backtest reliability — {season} (synthetic)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(100, bin_obs.max() * 1.1))
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "12_calibration_curve")
    print(
        f"Saved reliability diagram ({result.n_movements} movements, "
        f"{len(bin_pred)} bins)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
