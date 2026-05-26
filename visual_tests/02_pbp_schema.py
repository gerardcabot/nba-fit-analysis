#!/usr/bin/env python3
"""Visual test: PlayByPlayV3 actionType distribution for one probed game."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from visual_tests._constants import (
    DEFAULT_PROBE_GAME_ID,
    NBA_API_PACING_DELAY_SEC,
    NBA_API_REQUEST_TIMEOUT_SEC,
    PBP_ACTION_TYPE_COLUMN,
    PBP_DATASET_INDEX,
    PROBE_RESULTS_JSON,
    TOP_ACTION_TYPES_TO_LABEL,
)
from visual_tests._plot_utils import BAR_ALPHA, COLOR_NEUTRAL, new_figure, save_figure

JSON_KEY_META = "meta"
JSON_KEY_GAME_ID = "game_id"

CHART_TITLE = "PlayByPlayV3 Event Type Distribution"
X_AXIS_LABEL = "Event count"
Y_AXIS_LABEL = "actionType (PlayByPlayV3)"
OUTPUT_STEM = "02_pbp_action_type_distribution"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def _probe_game_id() -> str:
    if PROBE_RESULTS_JSON.is_file():
        with PROBE_RESULTS_JSON.open(encoding="utf-8") as handle:
            meta = json.load(handle).get(JSON_KEY_META, {})
        return str(meta.get(JSON_KEY_GAME_ID, DEFAULT_PROBE_GAME_ID))
    return DEFAULT_PROBE_GAME_ID


def _patch_nba_api_headers() -> None:
    from nba_api.stats.library.http import NBAStatsHTTP

    NBAStatsHTTP.headers = {**NBAStatsHTTP.headers, **BROWSER_HEADERS}


def _fetch_playbyplay_v3(game_id: str):
    from nba_api.stats.endpoints import playbyplayv3

    _patch_nba_api_headers()
    time.sleep(NBA_API_PACING_DELAY_SEC)
    endpoint = playbyplayv3.PlayByPlayV3(
        game_id=game_id,
        timeout=NBA_API_REQUEST_TIMEOUT_SEC,
    )
    return endpoint.get_data_frames()[PBP_DATASET_INDEX]


def main() -> int:
    try:
        import nba_api  # noqa: F401
    except ImportError:
        print("SKIP: nba_api not installed — install requirements.txt to run this test.")
        return 0

    game_id = _probe_game_id()
    df = _fetch_playbyplay_v3(game_id)

    if PBP_ACTION_TYPE_COLUMN not in df.columns:
        print(
            f"FAIL: column {PBP_ACTION_TYPE_COLUMN!r} missing; got {list(df.columns)}",
            file=sys.stderr,
        )
        return 1

    counts = df[PBP_ACTION_TYPE_COLUMN].value_counts()
    if TOP_ACTION_TYPES_TO_LABEL and len(counts) > TOP_ACTION_TYPES_TO_LABEL:
        counts = counts.head(TOP_ACTION_TYPES_TO_LABEL)

    fig, ax = new_figure(width_in=10.0, height_in=max(6.0, 0.35 * len(counts)))
    y_pos = range(len(counts))
    ax.barh(
        y_pos,
        counts.values,
        color=COLOR_NEUTRAL,
        alpha=BAR_ALPHA,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(counts.index.astype(str))
    ax.invert_yaxis()
    ax.set_title(f"{CHART_TITLE} (game_id={game_id}, n={len(df)} events)")
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel(Y_AXIS_LABEL)

    out_path = save_figure(fig, OUTPUT_STEM)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
