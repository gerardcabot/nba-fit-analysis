"""Build season-level scaled player/team vectors for fit scoring."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from nba_fit.config.settings import get_settings
from nba_fit.data.client import NBAClient
from nba_fit.features.constants import COL_PLAYER_ID, COL_TEAM_ID
from nba_fit.features.player_vector import build_player_features
from nba_fit.features.scaling import scale_player_features, scale_team_features
from nba_fit.features.team_vector import aggregate_roster_composition, build_team_features
from nba_fit.features.vectors import PlayerVector, TeamVector
from nba_fit.features.supplemental_join import enrich_player_frame, enrich_team_frame
from nba_fit.normalize.players import load_players_table
from nba_fit.normalize.teams import load_teams_table

DEMO_PLAYER_ID = 2544
DEMO_TEAM_ID = 1610612747

# Game-log date column candidates for temporal ``as_of`` filtering.
GAMELOG_DATE_COLUMNS: tuple[str, ...] = ("GAME_DATE", "game_date", "GAME_DATE_EST")


def filter_gamelogs_as_of(
    gamelogs: pd.DataFrame,
    move_date: date,
    *,
    date_col: str | None = None,
) -> pd.DataFrame:
    """
    Keep only game-log rows strictly before *move_date* (trade deadline / FA move).

    Intended for leakage-safe backtests: features and rates should reflect
    pre-move performance only. When *date_col* is omitted, the first matching
    column from :data:`GAMELOG_DATE_COLUMNS` is used.

    Raises ``KeyError`` if no date column is found.
    """
    if gamelogs.empty:
        return gamelogs.copy()

    col = date_col
    if col is None:
        for candidate in GAMELOG_DATE_COLUMNS:
            if candidate in gamelogs.columns:
                col = candidate
                break
    if col is None or col not in gamelogs.columns:
        raise KeyError(
            f"gamelogs missing date column; tried {GAMELOG_DATE_COLUMNS!r}, "
            f"got {list(gamelogs.columns)}"
        )

    parsed = pd.to_datetime(gamelogs[col], errors="coerce")
    cutoff = pd.Timestamp(move_date)
    mask = parsed < cutoff
    return gamelogs.loc[mask].copy()


def _gamelog_column(logs: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in logs.columns:
            return col
    upper = {c.upper(): c for c in logs.columns}
    for col in candidates:
        if col.upper() in upper:
            return upper[col.upper()]
    return None


def _parse_gamelog_minutes(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            return float(parts[0]) + float(parts[1]) / 60.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def aggregate_gamelogs_to_player_stats(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse game logs to one leaguedash-shaped row per player (latest team).

    Used by :meth:`SeasonFitContext.as_of` for leakage-safe pre-move profiles.
    """
    if logs.empty:
        return pd.DataFrame()

    pid_col = _gamelog_column(logs, ("PLAYER_ID", "player_id"))
    team_col = _gamelog_column(logs, ("TEAM_ID", "team_id"))
    date_col = _gamelog_column(logs, GAMELOG_DATE_COLUMNS)
    if pid_col is None or team_col is None:
        return pd.DataFrame()

    work = logs.copy()
    if date_col is not None:
        work["_sort_dt"] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=["_sort_dt"]).sort_values([pid_col, "_sort_dt"])
    else:
        work = work.sort_values(pid_col)

    min_col = _gamelog_column(logs, ("MIN", "min", "minutes"))
    stat_cols = {
        "PTS": _gamelog_column(logs, ("PTS", "pts")),
        "AST": _gamelog_column(logs, ("AST", "ast")),
        "TOV": _gamelog_column(logs, ("TOV", "tov")),
        "USG_PCT": _gamelog_column(logs, ("USG_PCT", "usg_pct")),
        "TS_PCT": _gamelog_column(logs, ("TS_PCT", "ts_pct")),
        "FGA": _gamelog_column(logs, ("FGA", "fga")),
        "FG3A": _gamelog_column(logs, ("FG3A", "fg3a")),
    }

    rows: list[dict[str, Any]] = []
    for player_id, grp in work.groupby(pid_col, sort=False):
        last = grp.iloc[-1]
        gp = len(grp)
        if min_col is not None:
            mins = grp[min_col].map(_parse_gamelog_minutes)
            total_min = float(mins.sum())
            min_pg = total_min / gp if gp else 0.0
        else:
            total_min = 0.0
            min_pg = 0.0

        row: dict[str, Any] = {
            COL_PLAYER_ID: int(player_id),
            COL_TEAM_ID: int(last[team_col]),
            "GP": gp,
            "MIN": min_pg,
        }
        for out_col, src in stat_cols.items():
            if src is None or src not in grp.columns:
                continue
            vals = pd.to_numeric(grp[src], errors="coerce").dropna()
            if vals.empty:
                continue
            if out_col in ("USG_PCT", "TS_PCT"):
                row[out_col] = float(vals.mean())
            else:
                row[out_col] = float(vals.sum() / gp) if gp else float(vals.mean())
        rows.append(row)

    return pd.DataFrame(rows)


def merge_as_of_player_stats(
    players_raw: pd.DataFrame, as_of_stats: pd.DataFrame
) -> pd.DataFrame:
    """Overlay game-log aggregates onto a leaguedash-wide player table."""
    if as_of_stats.empty:
        return players_raw.copy()
    if players_raw.empty:
        return as_of_stats.copy()

    base = players_raw.copy()
    overlay_cols = [c for c in as_of_stats.columns if c != COL_PLAYER_ID]
    indexed = base.set_index(COL_PLAYER_ID)
    upd = as_of_stats.set_index(COL_PLAYER_ID)
    for col in overlay_cols:
        if col in upd.columns:
            indexed.loc[upd.index, col] = upd[col]
    if COL_TEAM_ID in upd.columns:
        indexed.loc[upd.index, COL_TEAM_ID] = upd[COL_TEAM_ID]
    return indexed.reset_index()


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
    as_of_date: date | None = None
    gamelogs: pd.DataFrame | None = field(default=None, repr=False)
    _players_raw: pd.DataFrame | None = field(default=None, repr=False)
    _teams_raw: pd.DataFrame | None = field(default=None, repr=False)
    _effective_players_raw: pd.DataFrame | None = field(default=None, repr=False)

    def with_gamelogs(self, gamelogs: pd.DataFrame) -> SeasonFitContext:
        """Attach game logs for temporal :meth:`as_of` re-materialization."""
        return replace(self, gamelogs=gamelogs)

    def as_of(self, move_date: date | None) -> SeasonFitContext:
        """
        Return a context scoped to stats available strictly before *move_date*.

        When game logs and interim player frames are present, player vectors are
        rebuilt from :func:`filter_gamelogs_as_of` + :func:`aggregate_gamelogs_to_player_stats`.
        Team vectors are unchanged (league-wide baselines). Otherwise sets
        ``as_of_date`` only (metadata for callers).

        When *move_date* is ``None``, clears the temporal cutoff (full season).
        """
        if move_date is None:
            if self._players_raw is not None and self._teams_raw is not None:
                rebuilt = self._from_frames(
                    self._players_raw,
                    self._teams_raw,
                    self.season,
                    source=self.source,
                    gamelogs=self.gamelogs,
                    stored_players_raw=self._players_raw,
                    stored_teams_raw=self._teams_raw,
                    as_of_date=None,
                )
                return rebuilt
            return replace(self, as_of_date=None)

        if self.gamelogs is None or self.gamelogs.empty:
            return replace(self, as_of_date=move_date)
        if self._players_raw is None or self._teams_raw is None:
            return replace(self, as_of_date=move_date)

        filtered = filter_gamelogs_as_of(self.gamelogs, move_date)
        partial = aggregate_gamelogs_to_player_stats(filtered)
        merged_players = merge_as_of_player_stats(self._players_raw, partial)
        return self._from_frames(
            merged_players,
            self._teams_raw,
            self.season,
            source=f"{self.source}_as_of",
            gamelogs=self.gamelogs,
            stored_players_raw=self._players_raw,
            stored_teams_raw=self._teams_raw,
            effective_players_raw=merged_players,
            as_of_date=move_date,
        )

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
        players_raw = enrich_player_frame(load_players_table(season), season)
        teams_raw = enrich_team_frame(load_teams_table(season), season)
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
        gamelogs: pd.DataFrame | None = None,
        stored_players_raw: pd.DataFrame | None = None,
        stored_teams_raw: pd.DataFrame | None = None,
        effective_players_raw: pd.DataFrame | None = None,
        as_of_date: date | None = None,
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

        return cls(
            season=season,
            players=players,
            teams=teams,
            source=source,
            as_of_date=as_of_date,
            gamelogs=gamelogs,
            _players_raw=(
                stored_players_raw.copy()
                if stored_players_raw is not None
                else players_raw.copy()
            ),
            _teams_raw=(
                stored_teams_raw.copy()
                if stored_teams_raw is not None
                else teams_raw.copy()
            ),
            _effective_players_raw=(
                effective_players_raw.copy()
                if effective_players_raw is not None
                else None
            ),
        )
