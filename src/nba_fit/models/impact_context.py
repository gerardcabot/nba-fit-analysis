"""Train/load Option C RAPM + lineup model artifacts for a season."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nba_fit.config.settings import PLAYER_STAT_MIN
from nba_fit.features.constants import COL_PLAYER_ID, COL_TEAM_ID
from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.lineup_model import (
    LineupModelArtifacts,
    fit_lineup_model,
    load_lineup_model,
    save_lineup_model,
    synthetic_lineup_table,
)
from nba_fit.models.rapm import (
    RapmArtifacts,
    check_degenerate_rapm,
    fit_rapm_from_lineup_table,
    fit_rapm_from_possessions,
    fit_rapm_from_stints,
    load_rapm,
    save_rapm,
    stints_from_possessions,
    synthetic_stint_matrix,
)
from nba_fit.models.role_context import RoleFitContext
from nba_fit.models.role_embeddings import RoleEmbeddingArtifacts
from nba_fit.normalize.lineups import load_lineup_units_table
from nba_fit.normalize.players import load_players_table
from nba_fit.normalize.players import rotation_minutes_from_players
from nba_fit.normalize.possessions import has_possessions_partition, load_possessions_table


@dataclass
class ImpactFitContext:
    """Season-level Option C state for lineup impact scoring."""

    season: str
    rapm: RapmArtifacts
    lineup_model: LineupModelArtifacts
    embeddings: RoleEmbeddingArtifacts
    lineup_stints: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_team_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    rotation_minutes: dict[int, float] = field(default_factory=dict)
    rapm_source: str = "lineup_units"
    degenerate_rapm: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_possessions_table(
        cls,
        possessions: pd.DataFrame,
        lineups: pd.DataFrame,
        embeddings: RoleEmbeddingArtifacts,
        *,
        season: str,
        players_raw: pd.DataFrame | None = None,
        persist: bool = False,
    ) -> ImpactFitContext:
        """Train RAPM from possession-native stint matrix; lineup model from lineups."""
        rapm = fit_rapm_from_possessions(
            possessions,
            season=season,
            tune_hyperparameters=True,
        )
        lineup_model = fit_lineup_model(lineups, embeddings, season=season)
        if persist:
            save_rapm(rapm)
            save_lineup_model(lineup_model)

        stints = stints_from_possessions(possessions)
        player_team_map, rotation_minutes = _roster_context(players_raw)

        return cls(
            season=season,
            rapm=rapm,
            lineup_model=lineup_model,
            embeddings=embeddings,
            lineup_stints=stints,
            player_team_map=player_team_map,
            rotation_minutes=rotation_minutes,
            rapm_source=rapm.rapm_source,
            degenerate_rapm=rapm.degenerate,
            metadata=dict(rapm.metadata),
        )

    @classmethod
    def from_lineup_table(
        cls,
        lineups: pd.DataFrame,
        embeddings: RoleEmbeddingArtifacts,
        *,
        season: str,
        players_raw: pd.DataFrame | None = None,
        persist: bool = False,
    ) -> ImpactFitContext:
        rapm = fit_rapm_from_lineup_table(
            lineups,
            season=season,
            tune_hyperparameters=True,
        )
        lineup_model = fit_lineup_model(lineups, embeddings, season=season)
        if persist:
            save_rapm(rapm)
            save_lineup_model(lineup_model)

        stints = _lineup_stints_frame(lineups)
        player_team_map, rotation_minutes = _roster_context(players_raw)

        return cls(
            season=season,
            rapm=rapm,
            lineup_model=lineup_model,
            embeddings=embeddings,
            lineup_stints=stints,
            player_team_map=player_team_map,
            rotation_minutes=rotation_minutes,
            rapm_source=rapm.rapm_source,
            degenerate_rapm=rapm.degenerate,
            metadata=dict(rapm.metadata),
        )

    @classmethod
    def from_synthetic(
        cls,
        role_context: RoleFitContext,
        *,
        persist: bool = False,
    ) -> ImpactFitContext:
        """Deterministic synthetic lineups + stints for offline tests."""
        season = role_context.season
        pids = [int(p) for p in role_context.embeddings.player_ids[:20]]
        lineups = synthetic_lineup_table(pids, n_lineups=60)
        lineups["team_id"] = 1610612747

        stint_players, off_t, def_t, weights = synthetic_stint_matrix(
            n_players=len(pids),
            n_stints=100,
        )
        stint_players = [
            [pids[(int(p) - 1) % len(pids)] for p in stint] for stint in stint_players
        ]

        rapm = fit_rapm_from_stints(
            stint_players,
            off_t,
            def_t,
            season=season,
            stint_weights=weights,
        )
        rapm = check_degenerate_rapm(
            RapmArtifacts(
                season=rapm.season,
                player_ids=rapm.player_ids,
                orapm=rapm.orapm,
                drapm=rapm.drapm,
                net_rapm=rapm.net_rapm,
                stint_possessions=rapm.stint_possessions,
                low_sample_flag=rapm.low_sample_flag,
                ridge_alpha=rapm.ridge_alpha,
                recency_half_life_games=rapm.recency_half_life_games,
                rapm_source="synthetic",
            )
        )
        lineup_model = fit_lineup_model(lineups, role_context.embeddings, season=season)
        if persist:
            save_rapm(rapm)
            save_lineup_model(lineup_model)

        stints = pd.DataFrame(
            {
                "player_ids": stint_players,
                "team_id": 1610612747,
                "net_rating": off_t - def_t,
                "off_rating": off_t,
                "def_rating": def_t,
                "minutes": weights,
                "stint_weight": weights,
            }
        )
        player_team_map = role_context.player_team_map
        rotation_minutes = {int(p): 100.0 + i for i, p in enumerate(pids)}

        return cls(
            season=season,
            rapm=rapm,
            lineup_model=lineup_model,
            embeddings=role_context.embeddings,
            lineup_stints=stints,
            player_team_map=player_team_map,
            rotation_minutes=rotation_minutes,
            rapm_source=rapm.rapm_source,
            degenerate_rapm=rapm.degenerate,
            metadata=dict(rapm.metadata),
        )

    @classmethod
    def load(
        cls,
        season: str,
        role_context: RoleFitContext,
        *,
        lineups: pd.DataFrame | None = None,
        players_raw: pd.DataFrame | None = None,
    ) -> ImpactFitContext:
        rapm = load_rapm(season)
        lineup_model = load_lineup_model(season)
        if lineups is None:
            lineups = load_lineup_units_table(season)
        stints = _lineup_stints_frame(lineups)
        player_team_map, rotation_minutes = _roster_context(players_raw or role_context.player_team_map)
        return cls(
            season=season,
            rapm=rapm,
            lineup_model=lineup_model,
            embeddings=role_context.embeddings,
            lineup_stints=stints,
            player_team_map=player_team_map,
            rotation_minutes=rotation_minutes,
            rapm_source=rapm.rapm_source,
            degenerate_rapm=rapm.degenerate,
            metadata=dict(rapm.metadata),
        )

    @classmethod
    def from_season(
        cls,
        season: str | None = None,
        *,
        prefer_interim: bool = True,
        synthetic: bool = False,
        persist: bool = False,
    ) -> ImpactFitContext:
        role_context = RoleFitContext.from_season(
            season,
            prefer_interim=prefer_interim,
            synthetic=synthetic,
            persist=False,
        )
        season = role_context.season
        if synthetic:
            return cls.from_synthetic(role_context, persist=persist)

        try:
            from nba_fit.models.rapm import rapm_dir

            if not persist and (rapm_dir(season) / "player_rapm.parquet").is_file():
                lineups = load_lineup_units_table(season)
                players_raw = load_players_table(season)
                return cls.load(season, role_context, lineups=lineups, players_raw=players_raw)

            lineups = load_lineup_units_table(season)
            players_raw = load_players_table(season)
            if has_possessions_partition(season):
                possessions = load_possessions_table(season)
                return cls.from_possessions_table(
                    possessions,
                    lineups,
                    role_context.embeddings,
                    season=season,
                    players_raw=players_raw,
                    persist=persist,
                )
            return cls.from_lineup_table(
                lineups,
                role_context.embeddings,
                season=season,
                players_raw=players_raw,
                persist=persist,
            )
        except (FileNotFoundError, ValueError):
            try:
                return cls.load(season, role_context)
            except FileNotFoundError:
                return cls.from_synthetic(role_context, persist=persist)


def _lineup_stints_frame(lineups: pd.DataFrame) -> pd.DataFrame:
    from nba_fit.models.rapm import stints_from_lineup_units

    return stints_from_lineup_units(lineups)


def _roster_context(
    players_raw: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[int, float]]:
    if players_raw is None or players_raw.empty:
        return pd.DataFrame(), {}
    pid_col = COL_PLAYER_ID if COL_PLAYER_ID in players_raw.columns else "PLAYER_ID"
    team_col = COL_TEAM_ID if COL_TEAM_ID in players_raw.columns else "TEAM_ID"
    player_team_map = players_raw[[pid_col, team_col]].drop_duplicates()
    player_team_map = player_team_map.rename(columns={pid_col: COL_PLAYER_ID, team_col: COL_TEAM_ID})
    if PLAYER_STAT_MIN in players_raw.columns or "MIN" in players_raw.columns:
        minutes = rotation_minutes_from_players(players_raw)
    else:
        minutes = {}
    return player_team_map, minutes


def train_impact_for_season(
    season: str | None = None,
    *,
    prefer_interim: bool = True,
    synthetic: bool = False,
) -> ImpactFitContext:
    """Train and persist Option C RAPM and lineup net-rating models."""
    return ImpactFitContext.from_season(
        season,
        prefer_interim=prefer_interim,
        synthetic=synthetic,
        persist=True,
    )
