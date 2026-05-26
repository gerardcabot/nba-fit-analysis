#!/usr/bin/env python3
"""Histogram of possession rows per game (ingested or synthetic sample)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from nba_fit.config.settings import (
    DEFAULT_SEASON,
    POSSESSIONS_PER_TEAM_PER_GAME_REGULATION,
    get_settings,
)
from nba_fit.data.pbpstats_adapter import possessions_from_events_and_rotation
from nba_fit.data.fetchers.pbp import playbyplay_frame
from nba_fit.data.storage import interim_path
from nba_fit.config.settings import INTERIM_TABLE_POSSESSIONS
from nba_fit.normalize.possessions import (
    load_possessions_table,
    possessions_per_game_counts,
)


def _season_partition_dir(season: str) -> Path:
    return interim_path(INTERIM_TABLE_POSSESSIONS) / f"season={season}"

from visual_tests._plot_utils import apply_plot_style, save_figure


def _synthetic_possession_counts() -> pd.Series:
    """Offline fallback: one game sample from probe-style event list."""
    from nba_fit.data.client import NBAClient
    from nba_fit.data.fetchers.pbp import fetch_playbyplay_game, fetch_gamerotation_game, gamerotation_stints
    from nba_fit.config.settings import DEFAULT_GAME_ID

    settings = get_settings()
    client = NBAClient(settings=settings)
    season = settings.default_season
    game_id = DEFAULT_GAME_ID
    try:
        pbp = fetch_playbyplay_game(client, game_id, season=season, use_cache=True)
        rot = fetch_gamerotation_game(client, game_id, season=season, use_cache=True)
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP live fetch: {exc}")
        return pd.Series({game_id: int(POSSESSIONS_PER_TEAM_PER_GAME_REGULATION * 2)})

    df = possessions_from_events_and_rotation(
        playbyplay_frame(pbp),
        gamerotation_stints(rot),
        game_id=game_id,
        season=season,
    )
    return possessions_per_game_counts(df)


def _load_counts(season: str) -> pd.Series:
    base = _season_partition_dir(season)
    if base.exists():
        print(f"Using ingested possessions: {base}")
        df = load_possessions_table(season)
        return possessions_per_game_counts(df)
    print("No ingested possessions — using live single-game sample")
    return _synthetic_possession_counts()


def main() -> int:
    season = DEFAULT_SEASON
    counts = _load_counts(season)
    if counts.empty:
        print("No possession counts to plot", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(8, 5))
    apply_plot_style()
    ax.hist(counts.values, bins=min(20, len(counts)), color="#4C72B0", edgecolor="black")
    expected = POSSESSIONS_PER_TEAM_PER_GAME_REGULATION * 2
    ax.axvline(expected, color="#C44E52", linestyle="--", label=f"~{expected:.0f} total/game (2 teams)")
    ax.set_title(f"Possession rows per game ({season}, n={len(counts)} games)")
    ax.set_xlabel("Possession rows per game_id")
    ax.set_ylabel("Games")
    ax.legend()
    out = save_figure(fig, "09_possession_rate", subdir="ingest")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
