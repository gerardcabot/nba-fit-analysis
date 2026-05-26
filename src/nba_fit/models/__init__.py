"""Fit models, embeddings, and archetypes (Option B+)."""

from nba_fit.models.archetypes import ArchetypeArtifacts, fit_archetypes
from nba_fit.models.constants import NEAREST_NEIGHBOR_K, ROLE_FIT_WEIGHT
from nba_fit.models.player_comps import nearest_player_comps
from nba_fit.models.role_context import RoleFitContext, train_roles_for_season
from nba_fit.models.role_embeddings import RoleEmbeddingArtifacts, fit_role_embeddings

__all__ = [
    "ArchetypeArtifacts",
    "NEAREST_NEIGHBOR_K",
    "RoleEmbeddingArtifacts",
    "RoleFitContext",
    "ROLE_FIT_WEIGHT",
    "fit_archetypes",
    "fit_role_embeddings",
    "nearest_player_comps",
    "train_roles_for_season",
]
