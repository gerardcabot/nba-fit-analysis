#!/usr/bin/env python3
"""Visual test: bar chart of nba_api endpoint probe success vs failure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from visual_tests._constants import PROBE_RESULTS_JSON, REPO_ROOT
from visual_tests._plot_utils import (
    BAR_ALPHA,
    BAR_EDGE_WIDTH,
    COLOR_FAIL,
    COLOR_SUCCESS,
    new_figure,
    save_figure,
)

JSON_KEY_META = "meta"
JSON_KEY_RESULTS = "results"
JSON_KEY_SUCCESS = "success"
JSON_KEY_SEASON = "season"
JSON_KEY_TOTAL = "total"

OUTCOME_SUCCESS = "Success"
OUTCOME_FAILED = "Failed"

CHART_TITLE = "nba_api Endpoint Probe Health"
X_AXIS_LABEL = "Probe outcome"
Y_AXIS_LABEL = "Number of endpoints"
OUTPUT_STEM = "01_endpoint_health"


def _load_probe_payload(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Run: python {REPO_ROOT / 'probe_all_nba_endpoints.py'}"
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _count_outcomes(results: list[dict]) -> tuple[int, int]:
    success_count = sum(1 for row in results if row.get(JSON_KEY_SUCCESS))
    fail_count = len(results) - success_count
    return success_count, fail_count


def main() -> int:
    payload = _load_probe_payload(PROBE_RESULTS_JSON)
    meta = payload.get(JSON_KEY_META, {})
    results = payload.get(JSON_KEY_RESULTS, [])
    success_count, fail_count = _count_outcomes(results)
    total_endpoints = meta.get(JSON_KEY_TOTAL, len(results))
    season = meta.get(JSON_KEY_SEASON, "unknown season")

    fig, ax = new_figure()
    categories = [OUTCOME_SUCCESS, OUTCOME_FAILED]
    counts = [success_count, fail_count]
    colors = [COLOR_SUCCESS, COLOR_FAIL]

    bars = ax.bar(
        categories,
        counts,
        color=colors,
        edgecolor="black",
        linewidth=BAR_EDGE_WIDTH,
        alpha=BAR_ALPHA,
    )
    ax.bar_label(bars, labels=[str(c) for c in counts], padding=4)
    ax.set_title(f"{CHART_TITLE} ({season}, n={total_endpoints})")
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
