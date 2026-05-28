#!/usr/bin/env python3
"""Reliability diagram from Option D movement backtest rows (validation CSV)."""

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
import numpy as np
import pandas as pd

from visual_tests._plot_utils import apply_plot_style, save_figure
from visual_tests._validation_artifacts import resolve_backtest_csv


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


def _to_percent_scale(values: np.ndarray) -> np.ndarray:
    """Map unit-interval scores to 0–100 when values look like probabilities."""
    if values.size == 0:
        return values
    if float(np.nanmax(values)) <= 1.5:
        return values * 100.0
    return values


def _load_backtest_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing backtest CSV at {path}. "
            "Run scripts/run_option_d_validation.py first."
        )
    df = pd.read_csv(path)
    for col in ("calibrated_fit_percentile", "post_move_outcome"):
        if col not in df.columns:
            raise KeyError(f"backtest CSV missing column {col!r}")
    return df


def main(backtest_path: Path | None = None) -> int:
    path = backtest_path or resolve_backtest_csv()
    df = _load_backtest_frame(path)

    pred = _to_percent_scale(
        df["calibrated_fit_percentile"].to_numpy(dtype=float)
    )
    obs = _to_percent_scale(df["post_move_outcome"].to_numpy(dtype=float))

    if len(pred) < 3:
        print("Not enough movement rows for reliability plot", file=sys.stderr)
        return 1

    season = str(df["season"].iloc[0]) if "season" in df.columns else "unknown"
    n_moves = len(df)

    bin_pred, bin_obs, counts = _reliability_bins(pred, obs, n_bins=min(5, len(pred)))

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 100], [0, 100], "--", color="0.45", linewidth=1, label="perfect calibration")
    ax.scatter(bin_pred, bin_obs, s=counts * 8 + 40, color="#2a6f97", zorder=3, label="binned mean")
    ax.plot(bin_pred, bin_obs, "-", color="#e76f51", linewidth=2, alpha=0.85)
    ax.set_xlabel("Predicted fit (calibrated, ×100)")
    ax.set_ylabel("Mean post-move outcome (×100)")
    ax.set_title(
        f"Movement backtest reliability — {season} "
        f"({n_moves} moves, {path.name})"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(100, float(bin_obs.max()) * 1.1) if len(bin_obs) else 100)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "12_calibration_curve")
    print(f"Saved reliability diagram from {path} ({n_moves} movements, {len(bin_pred)} bins)")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backtest-csv",
        type=Path,
        default=None,
        help="Path to backtest_results.csv (default: validation artifact)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(main(args.backtest_csv))
