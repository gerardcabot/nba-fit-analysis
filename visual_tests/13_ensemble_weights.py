#!/usr/bin/env python3
"""Stacked bar of Option D ensemble contributions from a validation fit card."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from nba_fit.scoring.constants import ENSEMBLE_COMPONENT_NAMES
from visual_tests._plot_utils import apply_plot_style, save_figure
from visual_tests._validation_artifacts import resolve_fit_card_json


def _load_fit_card(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing fit card at {path}. "
            "Run scripts/run_option_d_validation.py first."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main(fit_card_path: Path | None = None) -> int:
    path = fit_card_path or resolve_fit_card_json()
    card = _load_fit_card(path)

    contributions = card.get("ensemble_contributions") or {}
    missing = [n for n in ENSEMBLE_COMPONENT_NAMES if n not in contributions]
    if missing:
        print(f"Fit card missing ensemble_contributions keys: {missing}", file=sys.stderr)
        return 1

    player_id = card.get("player_id", "?")
    team = card.get("team") or card.get("team_id", "?")
    season = card.get("season", "unknown")
    overall = card.get("overall_fit_percentile")
    overall_txt = f"{overall:.2f}" if overall is not None else "n/a"

    values = [float(contributions[name]) for name in ENSEMBLE_COMPONENT_NAMES]
    label = f"{player_id} @ {team}"

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.array([0.0])
    bottom = 0.0
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974"]
    for i, name in enumerate(ENSEMBLE_COMPONENT_NAMES):
        val = values[i]
        ax.bar(
            x,
            val,
            bottom=bottom,
            label=name.replace("_", " "),
            color=colors[i % len(colors)],
            width=0.55,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
        )
        bottom += val

    ax.set_xticks(x)
    ax.set_xticklabels([label], rotation=0, ha="center")
    ax.set_ylabel("Weighted contribution to raw ensemble")
    ax.set_title(
        f"Option D ensemble — {season} interim "
        f"(overall {overall_txt} pct, {path.name})"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    out_path = save_figure(fig, "13_ensemble_weights")
    print(f"Wrote {out_path} ({label}, overall={overall_txt})")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-card",
        type=Path,
        default=None,
        help="Path to fit_card JSON (default: validation artifact)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(main(args.fit_card))
