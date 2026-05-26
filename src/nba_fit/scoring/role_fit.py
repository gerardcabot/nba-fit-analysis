"""Option B team_need_fit submetric (role embedding + archetype need alignment)."""

from __future__ import annotations

import numpy as np

from nba_fit.features.team_need import TeamNeedProfile
from nba_fit.features.vectors import PlayerVector
from nba_fit.models.archetypes import ArchetypeArtifacts
from nba_fit.models.role_embeddings import RoleEmbeddingArtifacts
from nba_fit.scoring._similarity import cosine_similarity as _cosine_similarity
from nba_fit.scoring._similarity import role_fit_sigmoid as _sigmoid


def _player_archetype_indicator(
    player_id: int,
    archetypes: ArchetypeArtifacts,
    archetype_order: tuple[str, ...],
) -> np.ndarray:
    label = archetypes.label_for(player_id)
    if label is None:
        return np.zeros(len(archetype_order), dtype=float)
    vec = np.zeros(len(archetype_order), dtype=float)
    if label in archetype_order:
        vec[archetype_order.index(label)] = 1.0
    return vec


def team_need_fit(
    player: PlayerVector,
    team_need: TeamNeedProfile,
    *,
    embeddings: RoleEmbeddingArtifacts | None = None,
    archetypes: ArchetypeArtifacts | None = None,
) -> float:
    """Score how well a player fills a team's archetype gaps (0–1).

  Combines (a) archetype-gap complementarity and (b) embedding cosine vs a
  roster-implied need direction when embeddings are available.
    """
    n_arch = team_need.n_archetypes
    gap_block = team_need.values[:n_arch] if n_arch else np.array([], dtype=float)
    weak_block = team_need.values[n_arch:] if n_arch < len(team_need.values) else np.array([], dtype=float)

    arch_score = 0.5
    if archetypes is not None and n_arch > 0:
        indicator = _player_archetype_indicator(
            player.player_id, archetypes, team_need.archetype_labels
        )
        if np.linalg.norm(indicator) > 0 and np.linalg.norm(gap_block) > 0:
            raw = float(np.dot(indicator, gap_block))
            arch_score = _sigmoid(raw)

    embed_score = 0.5
    if embeddings is not None:
        emb = embeddings.embedding_for(player.player_id)
        if emb is not None and len(team_need.values) > 0:
            need_dir = np.zeros_like(emb)
            if len(gap_block) > 0:
                need_dir[: min(len(gap_block), len(emb))] = gap_block[: len(emb)]
            offset = min(len(gap_block), len(emb))
            if len(weak_block) > 0 and offset < len(emb):
                tail = min(len(weak_block), len(emb) - offset)
                need_dir[offset : offset + tail] = weak_block[:tail]
            if np.linalg.norm(need_dir) > 1e-9:
                cos = _cosine_similarity(emb, need_dir)
                embed_score = (cos + 1.0) / 2.0

    if archetypes is None and embeddings is None:
        return 0.5

    if archetypes is not None and embeddings is not None:
        return float(0.6 * arch_score + 0.4 * embed_score)
    if archetypes is not None:
        return arch_score
    return embed_score
