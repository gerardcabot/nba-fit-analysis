"""Load persisted Option B role artifacts for visual tests."""

from __future__ import annotations

from nba_fit.features.constants import COL_PLAYER_ID, COL_TEAM_ID
from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.role_context import RoleFitContext
from nba_fit.normalize.players import load_players_table


def load_trained_role_context(season: str) -> RoleFitContext:
    """Load embeddings/archetypes saved by ``train-roles`` and rebuild team needs."""
    context = SeasonFitContext.build(
        season,
        prefer_interim=True,
        prefer_api=False,
    )
    players_raw = load_players_table(season)
    player_team_map = players_raw[[COL_PLAYER_ID, COL_TEAM_ID]].drop_duplicates()
    return RoleFitContext.load(season, context, player_team_map)
