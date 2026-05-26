"""Fit models, embeddings, archetypes, and lineup impact (Option B+ / C)."""

from nba_fit.models.archetypes import ArchetypeArtifacts, fit_archetypes
from nba_fit.models.constants import (
    LINEUP_IMPACT_WEIGHT,
    LINEUP_SIM_TOP_N,
    NEAREST_NEIGHBOR_K,
    ROLE_FIT_WEIGHT,
)
from nba_fit.models.impact_context import ImpactFitContext, train_impact_for_season
from nba_fit.models.lineup_impact import LineupUnitProjection, aggregate_projected_delta
from nba_fit.models.player_comps import nearest_player_comps
from nba_fit.models.rapm import RapmArtifacts, fit_rapm_from_stints
from nba_fit.models.role_context import RoleFitContext, train_roles_for_season
from nba_fit.models.role_embeddings import RoleEmbeddingArtifacts, fit_role_embeddings

__all__ = [
    "ArchetypeArtifacts",
    "ImpactFitContext",
    "LINEUP_IMPACT_WEIGHT",
    "LINEUP_SIM_TOP_N",
    "LineupUnitProjection",
    "NEAREST_NEIGHBOR_K",
    "RapmArtifacts",
    "RoleEmbeddingArtifacts",
    "RoleFitContext",
    "ROLE_FIT_WEIGHT",
    "aggregate_projected_delta",
    "fit_archetypes",
    "fit_rapm_from_stints",
    "fit_role_embeddings",
    "nearest_player_comps",
    "train_impact_for_season",
    "train_roles_for_season",
]
