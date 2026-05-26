#!/usr/bin/env python3
"""Bar chart of on-court minutes from on/off summary (ingested or live sample team)."""

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
    DEFAULT_TEAM_ID,
    ONOFF_COL_COURT_STATUS,
    ONOFF_COL_VS_PLAYER_NAME,
    ONOFF_STAT_MIN,
    ONOFF_SUMMARY_DATASETS,
    get_settings,
)
from nba_fit.data.client import NBAClient
from nba_fit.data.fetchers.lineups_onoff import fetch_lineup_onoff
from nba_fit.normalize.lineups import interim_onoff_path, load_onoff_table, onoff_summary_rows

from visual_tests._plot_utils import apply_plot_style, save_figure


def _on_court_minutes(df: pd.DataFrame, *, top_n: int = 12) -> pd.DataFrame:
    if ONOFF_COL_COURT_STATUS in df.columns:
        df = df.loc[df[ONOFF_COL_COURT_STATUS] == "On"].copy()
    minutes = pd.to_numeric(df[ONOFF_STAT_MIN], errors="coerce")
    name_col = ONOFF_COL_VS_PLAYER_NAME if ONOFF_COL_VS_PLAYER_NAME in df.columns else "focal_player_name"
    if name_col not in df.columns:
        raise KeyError(f"No player name column in on/off table (tried {name_col!r})")
    out = df.assign(minutes=minutes).dropna(subset=["minutes", name_col])
    out = out.sort_values("minutes", ascending=False).head(top_n)
    return out[[name_col, "minutes"]].rename(columns={name_col: "player"})


def _load_onoff_frame(season: str, team_id: int) -> pd.DataFrame:
    path = interim_onoff_path(season)
    if path.exists():
        print(f"Using ingested on/off table: {path}")
        df = load_onoff_table(season)
        if "team_id" in df.columns:
            df = df.loc[df["team_id"] == team_id]
        return onoff_summary_rows(df)

    print(f"Interim on/off missing; live-fetching teamplayeronoffsummary team_id={team_id}...")
    client = NBAClient()
    result = fetch_lineup_onoff(
        client,
        "teamplayeronoffsummary",
        season=season,
        team_id=team_id,
        use_cache=True,
    )
    ds_name = ONOFF_SUMMARY_DATASETS[0]
    if ds_name not in result.frames:
        ds_name = next(iter(result.frames))
    return result.frames[ds_name]


def main() -> int:
    settings = get_settings()
    season = settings.default_season or DEFAULT_SEASON
    team_id = DEFAULT_TEAM_ID

    df = _load_onoff_frame(season, team_id)
    sample = _on_court_minutes(df, top_n=12)
    if sample.empty:
        print("No on-court minutes rows to plot.", file=sys.stderr)
        return 1

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        sample["player"][::-1],
        sample["minutes"][::-1],
        color="steelblue",
        edgecolor="white",
        alpha=0.9,
    )
    ax.set_xlabel("On-court minutes (season)")
    ax.set_ylabel("Teammate (VS player)")
    ax.set_title(f"On-court minutes sample — team {team_id}, {season}")
    fig.tight_layout()
    out = save_figure(fig, "06_onoff_minutes", subdir="ingest")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
