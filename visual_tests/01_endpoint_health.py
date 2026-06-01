#!/usr/bin/env python3
"""Visual test: bar chart of nba_api endpoint probe OK / EMPTY / FAIL counts."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from nba_fit.config.settings import get_settings
from nba_fit.data.registry import ProbeRegistry
from visual_tests._plot_utils import (
    BAR_ALPHA,
    BAR_EDGE_WIDTH,
    COLOR_EMPTY,
    COLOR_FAIL,
    COLOR_SUCCESS,
    new_figure,
    save_figure,
)

OUTCOME_OK = "OK"
OUTCOME_EMPTY = "EMPTY"
OUTCOME_FAIL = "FAIL"
OUTCOMES = (OUTCOME_OK, OUTCOME_EMPTY, OUTCOME_FAIL)

CHART_TITLE = "nba_api Endpoint Probe Health"
X_AXIS_LABEL = "Probe outcome"
Y_AXIS_LABEL = "Number of unique endpoints"
OUTPUT_STEM = "01_endpoint_health"


def main() -> int:
    settings = get_settings()
    registry = ProbeRegistry.load(settings=settings)
    summary = registry.summary()
    meta = registry.meta
    total_endpoints = len(registry.endpoints)
    season = meta.get("season", "unknown season")

    fig, ax = new_figure()
    categories = list(OUTCOMES)
    counts = [summary[c] for c in categories]
    colors = [COLOR_SUCCESS, COLOR_EMPTY, COLOR_FAIL]

    bars = ax.bar(
        categories,
        counts,
        color=colors,
        edgecolor="black",
        linewidth=BAR_EDGE_WIDTH,
        alpha=BAR_ALPHA,
    )
    ax.bar_label(bars, labels=[str(c) for c in counts], padding=4)
    ax.set_title(f"{CHART_TITLE} ({season}, n={total_endpoints} unique endpoints)")
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(Y_AXIS_LABEL)
    ax.set_ylim(bottom=0)

    out_path = save_figure(fig, OUTPUT_STEM)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
