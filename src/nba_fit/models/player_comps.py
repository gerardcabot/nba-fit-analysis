"""Nearest-neighbor player comps in role embedding space."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

from nba_fit.models.constants import NEAREST_NEIGHBOR_K
from nba_fit.models.role_embeddings import RoleEmbeddingArtifacts, role_embedding_dir

if TYPE_CHECKING:
    from nba_fit.models.archetypes import ArchetypeArtifacts


def fit_player_comps_index(artifacts: RoleEmbeddingArtifacts) -> NearestNeighbors:
    """Fit sklearn NearestNeighbors on season role embeddings."""
    n_neighbors = min(NEAREST_NEIGHBOR_K + 1, len(artifacts.player_ids))
    nn = NearestNeighbors(n_neighbors=max(1, n_neighbors), metric="euclidean")
    nn.fit(artifacts.embeddings)
    return nn


def save_player_comps_index(nn: NearestNeighbors, season: str, *, root: Path | None = None) -> Path:
    out_path = role_embedding_dir(season, root=root) / "comps_nn.joblib"
    joblib.dump(nn, out_path)
    return out_path


def load_player_comps_index(season: str, *, root: Path | None = None) -> NearestNeighbors:
    path = role_embedding_dir(season, root=root) / "comps_nn.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Player comps index not found: {path}")
    return joblib.load(path)


def nearest_player_comps(
    artifacts: RoleEmbeddingArtifacts,
    nn: NearestNeighbors,
    player_id: int,
    *,
    k: int | None = None,
    archetypes: ArchetypeArtifacts | None = None,
    player_names: dict[int, str] | None = None,
) -> list[dict[str, object]]:
    """Return k nearest neighbors in embedding space (excludes self)."""
    k = k if k is not None else NEAREST_NEIGHBOR_K
    hits = np.where(artifacts.player_ids == player_id)[0]
    if len(hits) == 0:
        return []
    idx = int(hits[0])
    n_query = min(k + 1, len(artifacts.player_ids))
    distances, indices = nn.kneighbors(artifacts.embeddings[idx : idx + 1], n_neighbors=n_query)
    comps: list[dict[str, object]] = []
    for dist, j in zip(distances[0], indices[0], strict=True):
        pid = int(artifacts.player_ids[int(j)])
        if pid == player_id:
            continue
        entry: dict[str, object] = {
            "player_id": pid,
            "player": (player_names or {}).get(pid, str(pid)),
            "distance": float(dist),
        }
        if archetypes is not None:
            entry["archetype"] = archetypes.label_for(pid)
        comps.append(entry)
        if len(comps) >= k:
            break
    return comps
