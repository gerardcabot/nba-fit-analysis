#!/usr/bin/env python3
"""Histograms of PTS, USG%, and TS% from ingested or live league-dash data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from nba_fit.config.settings import (
    DEFAULT_SEASON,
    PLAYER_STAT_EST_USG_PCT,
    PLAYER_STAT_PTS,
    PLAYER_STAT_TS_PCT,
    PLAYER_STAT_USG_PCT,
    get_settings,
)
from nba_fit.data.client import NBAClient
from nba_fit.data.fetchers.league_dash import fetch_league_dash, primary_frame
from nba_fit.normalize.players import interim_players_path, load_players_table

from visual_tests._plot_utils import save_figure

PRIMARY_SEASON = "2024-25"


def _resolve_season(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("PRIMARY_SEASON", "").strip()
    if env:
        return env
    if interim_players_path(PRIMARY_SEASON).exists():
        return PRIMARY_SEASON
    settings = get_settings()
    return settings.default_season or DEFAULT_SEASON


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        default=None,
        help="Season for interim/live league-dash (default: PRIMARY_SEASON env, "
        "else 2024-25 if interim exists, else project default)",
    )
    return parser.parse_args(argv)


def _usage_column(df: pd.DataFrame) -> str:
    if PLAYER_STAT_USG_PCT in df.columns:
        return PLAYER_STAT_USG_PCT
    if PLAYER_STAT_EST_USG_PCT in df.columns:
        return PLAYER_STAT_EST_USG_PCT
    raise KeyError(f"Neither {PLAYER_STAT_USG_PCT} nor {PLAYER_STAT_EST_USG_PCT} in table")


def _load_player_frame(season: str) -> pd.DataFrame:
    path = interim_players_path(season)
    if path.exists():
        print(f"Using ingested interim table: {path}")
        return load_players_table(season)

    print("Interim table missing; live-fetching leaguedashplayerstats + biostats...")
    client = NBAClient()
    stats = primary_frame(fetch_league_dash(client, "leaguedashplayerstats", season=season))
    bio = primary_frame(fetch_league_dash(client, "leaguedashplayerbiostats", season=season))
    bio_cols = ["PLAYER_ID", PLAYER_STAT_USG_PCT, PLAYER_STAT_TS_PCT]
    bio_cols = [c for c in bio_cols if c in bio.columns]
    merged = stats.merge(bio[bio_cols], on="PLAYER_ID", how="left", suffixes=("", "_bio"))
    return merged


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    season = _resolve_season(args.season)
    df = _load_player_frame(season)

    usg_col = _usage_column(df)
    metrics: list[tuple[str, str]] = [
        (PLAYER_STAT_PTS, "Points per game"),
        (usg_col, "Usage rate"),
        (PLAYER_STAT_TS_PCT, "True shooting %"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, label) in zip(axes, metrics, strict=True):
        if col not in df.columns:
            ax.set_title(f"{col} (missing)")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(series, bins=30, color="steelblue", edgecolor="white", alpha=0.9)
        ax.set_xlabel(label)
        ax.set_ylabel("Players")
        ax.set_title(f"{col} (n={len(series)})")

    source = "interim" if interim_players_path(season).exists() else "live"
    fig.suptitle(f"Player stat distributions — {season} ({source})", fontsize=14)
    fig.tight_layout()
    out = save_figure(fig, "03_player_stat_distributions", subdir="ingest")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
