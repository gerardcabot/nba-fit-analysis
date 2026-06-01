"""Fit/load Option B artifacts and team need profiles for a season."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nba_fit.features.constants import COL_PLAYER_ID, COL_TEAM_ID
from nba_fit.features.player_vector import build_player_features
from nba_fit.features.scaling import scale_player_features
from nba_fit.features.season_context import SeasonFitContext
from nba_fit.features.team_need import TeamNeedProfile, build_team_need_profiles
from nba_fit.models.archetypes import (
    ArchetypeArtifacts,
    fit_archetypes,
    load_archetypes,
    save_archetypes,
)
from nba_fit.models.constants import ROLE_EMBEDDING_MIN_PLAYERS
from nba_fit.models.player_comps import (
    fit_player_comps_index,
    load_player_comps_index,
    nearest_player_comps,
    save_player_comps_index,
)
from nba_fit.models.role_embeddings import (
    RoleEmbeddingArtifacts,
    fit_role_embeddings,
    load_role_embeddings,
    save_role_embeddings,
)
from nba_fit.normalize.players import load_players_table


@dataclass
class RoleFitContext:
    """Season-level Option B state for team_need_fit scoring."""

    season: str
    embeddings: RoleEmbeddingArtifacts
    archetypes: ArchetypeArtifacts
    team_needs: dict[int, TeamNeedProfile] = field(default_factory=dict)
    player_team_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    comps_index: object | None = None
    player_names: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_frames(
        cls,
        players_raw: pd.DataFrame,
        context: SeasonFitContext,
        *,
        persist: bool = False,
    ) -> RoleFitContext:
        """Fit embeddings + archetypes and derive team need profiles."""
        player_feats = build_player_features(players_raw)
        player_scaled = scale_player_features(player_feats)
        season = context.season

        if len(player_scaled) < ROLE_EMBEDDING_MIN_PLAYERS:
            raise ValueError(
                f"Need at least {ROLE_EMBEDDING_MIN_PLAYERS} players for RoleFitContext"
            )

        embeddings = fit_role_embeddings(player_scaled, season=season)
        archetypes = fit_archetypes(embeddings, player_scaled)

        comps_index = fit_player_comps_index(embeddings)
        player_names = _player_display_names(players_raw)

        if persist:
            save_role_embeddings(embeddings)
            save_archetypes(archetypes)
            save_player_comps_index(comps_index, season)

        team_col = COL_TEAM_ID
        if team_col not in players_raw.columns:
            raise ValueError("players_raw must include TEAM_ID for team need profiles")

        player_team_map = players_raw[[COL_PLAYER_ID, team_col]].drop_duplicates()
        team_needs = build_team_need_profiles(
            archetypes,
            player_team_map,
            context.teams,
            season=season,
        )

        return cls(
            season=season,
            embeddings=embeddings,
            archetypes=archetypes,
            team_needs=team_needs,
            player_team_map=player_team_map,
            comps_index=comps_index,
            player_names=player_names,
        )

    @classmethod
    def from_synthetic(cls, context: SeasonFitContext) -> RoleFitContext:
        from nba_fit.features._synthetic import synthetic_player_df

        n = max(len(context.players), ROLE_EMBEDDING_MIN_PLAYERS + 1)
        players_raw = synthetic_player_df(n_rows=2)
        rows = []
        pids = list(context.players.keys())
        for i, pid in enumerate(pids):
            row = players_raw.iloc[i % len(players_raw)].copy()
            row[COL_PLAYER_ID] = pid
            row[COL_TEAM_ID] = 1610612737 + (i % 30)
            rows.append(row)
        while len(rows) < n:
            i = len(rows)
            row = players_raw.iloc[i % len(players_raw)].copy()
            row[COL_PLAYER_ID] = 1_000_000 + i
            row[COL_TEAM_ID] = 1610612737 + (i % 30)
            rows.append(row)
        return cls.from_frames(pd.DataFrame(rows), context)

    @classmethod
    def load(cls, season: str, context: SeasonFitContext, player_team_map: pd.DataFrame) -> RoleFitContext:
        embeddings = load_role_embeddings(season)
        archetypes = load_archetypes(season)
        comps_index = load_player_comps_index(season)
        team_needs = build_team_need_profiles(
            archetypes,
            player_team_map,
            context.teams,
            season=season,
        )
        names = _player_display_names_from_map(player_team_map)
        return cls(
            season=season,
            embeddings=embeddings,
            archetypes=archetypes,
            team_needs=team_needs,
            player_team_map=player_team_map,
            comps_index=comps_index,
            player_names=names,
        )

    @classmethod
    def from_season(
        cls,
        season: str | None = None,
        *,
        prefer_interim: bool = True,
        prefer_api: bool = True,
        synthetic: bool = False,
        persist: bool = False,
    ) -> RoleFitContext:
        """Build or load Option B artifacts for a season."""
        context = SeasonFitContext.build(
            season=season,
            prefer_interim=prefer_interim and not synthetic,
            prefer_api=prefer_api and not synthetic,
        )
        season = context.season
        if synthetic:
            return cls.from_synthetic(context)

        try:
            players_raw = load_players_table(season)
            return cls.from_frames(players_raw, context, persist=persist)
        except FileNotFoundError:
            try:
                return cls.load(season, context, _player_team_map_from_context(context))
            except FileNotFoundError:
                return cls.from_synthetic(context)

    def nearest_comps(self, player_id: int, *, k: int | None = None) -> list[dict[str, object]]:
        if self.comps_index is None:
            return []
        return nearest_player_comps(
            self.embeddings,
            self.comps_index,
            player_id,
            k=k,
            archetypes=self.archetypes,
            player_names=self.player_names,
        )


def _player_team_map_from_context(context: SeasonFitContext) -> pd.DataFrame:
    rows = []
    for i, pid in enumerate(sorted(context.players.keys())):
        rows.append({COL_PLAYER_ID: pid, COL_TEAM_ID: 1610612737 + (i % 30)})
    return pd.DataFrame(rows)


def _player_display_names(players_raw: pd.DataFrame) -> dict[int, str]:
    pid_col = COL_PLAYER_ID if COL_PLAYER_ID in players_raw.columns else "PLAYER_ID"
    name_col = "PLAYER_NAME" if "PLAYER_NAME" in players_raw.columns else None
    names: dict[int, str] = {}
    if name_col:
        for _, row in players_raw.iterrows():
            names[int(row[pid_col])] = str(row[name_col])
    return names


def _player_display_names_from_map(player_team_map: pd.DataFrame) -> dict[int, str]:
    pid_col = COL_PLAYER_ID if COL_PLAYER_ID in player_team_map.columns else "PLAYER_ID"
    return {int(row[pid_col]): f"Player {int(row[pid_col])}" for _, row in player_team_map.iterrows()}


def train_roles_for_season(
    season: str | None = None,
    *,
    prefer_interim: bool = True,
    prefer_api: bool = True,
    synthetic: bool = False,
) -> RoleFitContext:
    """Train and persist Option B role embeddings, archetypes, team need, and comps."""
    return RoleFitContext.from_season(
        season,
        prefer_interim=prefer_interim,
        prefer_api=prefer_api,
        synthetic=synthetic,
        persist=True,
    )
