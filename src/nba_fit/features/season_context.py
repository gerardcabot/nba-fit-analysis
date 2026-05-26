"""Build season-level scaled player/team vectors for fit scoring."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nba_fit.config.settings import get_settings
from nba_fit.data.client import NBAClient
from nba_fit.features.constants import COL_PLAYER_ID, COL_TEAM_ID
from nba_fit.features.player_vector import build_player_features
from nba_fit.features.scaling import scale_player_features, scale_team_features
from nba_fit.features.team_vector import aggregate_roster_composition, build_team_features
from nba_fit.features.vectors import PlayerVector, TeamVector
from nba_fit.normalize.players import load_players_table
from nba_fit.normalize.teams import load_teams_table

DEMO_PLAYER_ID = 2544
DEMO_TEAM_ID = 1610612747

_TEAM_ABBREV: dict[int, str] = {
    1610612737: "ATL",
    1610612738: "BOS",
    1610612739: "BKN",
    1610612740: "NOP",
    1610612741: "CHI",
    1610612742: "CLE",
    1610612743: "DEN",
    1610612744: "GSW",
    1610612745: "HOU",
    1610612746: "LAC",
    1610612747: "LAL",
    1610612748: "MIA",
    1610612749: "MIL",
    1610612750: "MIN",
    1610612751: "DAL",
    1610612752: "NYK",
    1610612753: "ORL",
    1610612754: "IND",
    1610612755: "PHI",
    1610612756: "PHX",
    1610612757: "POR",
    1610612758: "SAC",
    1610612759: "SAS",
    1610612760: "OKC",
    1610612761: "TOR",
    1610612762: "UTA",
    1610612763: "MEM",
    1610612764: "WAS",
    1610612765: "DET",
    1610612766: "CHA",
}


@dataclass
class SeasonFitContext:
    season: str
    players: dict[int, PlayerVector] = field(default_factory=dict)
    teams: dict[int, TeamVector] = field(default_factory=dict)
    source: str = "synthetic"

    @classmethod
    def build(
        cls,
        season: str | None = None,
        *,
        prefer_interim: bool = True,
        prefer_api: bool = True,
        client: NBAClient | None = None,
    ) -> SeasonFitContext:
        season = season or get_settings().default_season
        if prefer_interim:
            try:
                return cls.from_interim(season)
            except FileNotFoundError:
                pass
        if prefer_api:
            try:
                return cls.from_api(season, client=client)
            except Exception:  # noqa: BLE001
                pass
        return cls.from_synthetic(season)

    @classmethod
    def from_interim(cls, season: str) -> SeasonFitContext:
        players_raw = load_players_table(season)
        teams_raw = load_teams_table(season)
        return cls._from_frames(players_raw, teams_raw, season, source="interim")

    @classmethod
    def from_api(cls, season: str, client: NBAClient | None = None) -> SeasonFitContext:
        client = client or NBAClient()
        p_result = client.fetch("leaguedashplayerstats", season=season)
        t_result = client.fetch("leaguedashteamstats", season=season)
        players_raw = next(iter(p_result.frames.values()))
        teams_raw = next(iter(t_result.frames.values()))
        return cls._from_frames(players_raw, teams_raw, season, source="api")

    @classmethod
    def from_synthetic(cls, season: str, n_players: int = 120) -> SeasonFitContext:
        from nba_fit.features._synthetic import synthetic_player_df, synthetic_team_df

        base = synthetic_player_df(n_rows=2)
        rows = []
        rng = np.random.default_rng(42)
        for i in range(n_players):
            row = base.iloc[i % len(base)].copy()
            row[COL_PLAYER_ID] = DEMO_PLAYER_ID if i == 0 else int(rng.integers(1_000_000, 9_999_999))
            row[COL_TEAM_ID] = 1610612737 + (i % 30)
            rows.append(row)
        players_raw = pd.DataFrame(rows)
        teams_raw = synthetic_team_df(list(_TEAM_ABBREV.keys()))
        return cls._from_frames(players_raw, teams_raw, season, source="synthetic")

    @classmethod
    def _from_frames(
        cls,
        players_raw: pd.DataFrame,
        teams_raw: pd.DataFrame,
        season: str,
        *,
        source: str,
    ) -> SeasonFitContext:
        player_feats = build_player_features(players_raw)
        player_scaled = scale_player_features(player_feats)
        roster = aggregate_roster_composition(players_raw)
        team_feats = build_team_features(teams_raw, roster_agg=roster)
        team_scaled = scale_team_features(team_feats)

        z_player_cols = sorted(c for c in player_scaled.columns if c.startswith("z_"))
        z_team_cols = sorted(c for c in team_scaled.columns if c.startswith("z_"))

        players: dict[int, PlayerVector] = {}
        for _, row in player_scaled.iterrows():
            pid = int(row[COL_PLAYER_ID])
            vals = row[z_player_cols].to_numpy(dtype=float)
            vals = np.nan_to_num(vals, nan=0.0)
            players[pid] = PlayerVector(
                player_id=pid,
                season=season,
                values=vals,
                feature_names=tuple(z_player_cols),
            )

        teams: dict[int, TeamVector] = {}
        for _, row in team_scaled.iterrows():
            tid = int(row[COL_TEAM_ID])
            vals = row[z_team_cols].to_numpy(dtype=float)
            vals = np.nan_to_num(vals, nan=0.0)
            name = _TEAM_ABBREV.get(tid, str(tid))
            if "TEAM_NAME" in teams_raw.columns:
                match = teams_raw.loc[teams_raw[COL_TEAM_ID] == tid, "TEAM_NAME"]
                if not match.empty:
                    name = str(match.iloc[0])
            teams[tid] = TeamVector(
                team_id=tid,
                season=season,
                values=vals,
                feature_names=tuple(z_team_cols),
                display_name=name,
            )

        return cls(season=season, players=players, teams=teams, source=source)
