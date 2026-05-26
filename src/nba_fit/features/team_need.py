"""Team need vectors from roster archetype gaps and lineup weakness proxies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_fit.features.constants import COL_PLAYER_ID, COL_TEAM_ID
from nba_fit.features.team_vector import FEATURE_GROUP_WEAKNESSES
from nba_fit.features.vectors import TeamVector
from nba_fit.models.archetypes import ArchetypeArtifacts
from nba_fit.models.constants import (
    ARCHETYPE_LEAGUE_SHARE_FLOOR,
    ARCHETYPE_NOISE_LABEL,
    TEAM_NEED_ARCHETYPE_BLOCK_WEIGHT,
    TEAM_NEED_WEAKNESS_BLOCK_WEIGHT,
    VECTOR_NORM_EPSILON,
)


@dataclass(frozen=True)
class TeamNeedProfile:
    """Archetype-gap block + weakness proxy block for one team."""

    team_id: int
    season: str
    values: np.ndarray
    archetype_labels: tuple[str, ...]
    weakness_feature_names: tuple[str, ...]

    @property
    def n_archetypes(self) -> int:
        return len(self.archetype_labels)


def _league_archetype_shares(
    archetypes: ArchetypeArtifacts,
) -> dict[str, float]:
    labels = [str(l) for l in archetypes.archetype_labels if str(l) != ARCHETYPE_NOISE_LABEL]
    n = len(labels)
    if n == 0:
        return {}
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return {label: count / n for label, count in counts.items()}


def _roster_archetype_shares(
    roster_labels: list[str],
) -> dict[str, float]:
    if not roster_labels:
        return {}
    n = len(roster_labels)
    counts: dict[str, int] = {}
    for label in roster_labels:
        if label == ARCHETYPE_NOISE_LABEL:
            continue
        counts[label] = counts.get(label, 0) + 1
    return {label: count / n for label, count in counts.items()}


def archetype_gap_vector(
    league_shares: dict[str, float],
    roster_shares: dict[str, float],
    archetype_order: tuple[str, ...],
) -> np.ndarray:
    """Positive gaps mean the team under-indexes that archetype vs league."""
    gaps = []
    for label in archetype_order:
        league_share = league_shares.get(label, 0.0)
        if league_share < ARCHETYPE_LEAGUE_SHARE_FLOOR:
            gaps.append(0.0)
            continue
        roster_share = roster_shares.get(label, 0.0)
        gaps.append(max(0.0, league_share - roster_share))
    arr = np.asarray(gaps, dtype=float)
    norm = np.linalg.norm(arr)
    if norm > VECTOR_NORM_EPSILON:
        arr = arr / norm
    return arr


def weakness_proxy_vector(team: TeamVector) -> tuple[np.ndarray, tuple[str, ...]]:
    """Scaled team weakness features (higher = larger team need)."""
    prefix = f"z_tf_{FEATURE_GROUP_WEAKNESSES}__"
    names = tuple(
        n.removeprefix(prefix) for n in team.feature_names if n.startswith(prefix)
    )
    raw = team.select_groups(FEATURE_GROUP_WEAKNESSES)
    if len(raw) == 0:
        return np.array([], dtype=float), ()
    # Weakness columns are oriented so higher z => worse team outcome on that axis.
    positive_need = np.maximum(raw, 0.0)
    norm = np.linalg.norm(positive_need)
    if norm > VECTOR_NORM_EPSILON:
        positive_need = positive_need / norm
    return positive_need.astype(float), names


def build_team_need_profiles(
    archetypes: ArchetypeArtifacts,
    player_team_map: pd.DataFrame,
    teams: dict[int, TeamVector],
    *,
    season: str,
) -> dict[int, TeamNeedProfile]:
    """Build one need vector per team from roster archetype gaps + weaknesses."""
    league_shares = _league_archetype_shares(archetypes)
    archetype_order = tuple(sorted(league_shares.keys()))
    if not archetype_order:
        archetype_order = tuple(sorted(set(str(l) for l in archetypes.archetype_labels)))

    label_by_player = {
        int(pid): str(lab)
        for pid, lab in zip(archetypes.player_ids, archetypes.archetype_labels, strict=False)
    }

    team_col = COL_TEAM_ID
    if team_col not in player_team_map.columns:
        raise ValueError("player_team_map must include TEAM_ID")

    profiles: dict[int, TeamNeedProfile] = {}
    for team_id, team in teams.items():
        roster_pids = player_team_map.loc[
            player_team_map[team_col] == team_id, COL_PLAYER_ID
        ].astype(int)
        roster_labels = [label_by_player[pid] for pid in roster_pids if pid in label_by_player]
        roster_shares = _roster_archetype_shares(roster_labels)
        gap_block = archetype_gap_vector(league_shares, roster_shares, archetype_order)
        weak_block, weak_names = weakness_proxy_vector(team)

        if len(weak_block) == 0 and len(gap_block) == 0:
            combined = np.array([0.0], dtype=float)
        elif len(weak_block) == 0:
            combined = gap_block * TEAM_NEED_ARCHETYPE_BLOCK_WEIGHT
        elif len(gap_block) == 0:
            combined = weak_block * TEAM_NEED_WEAKNESS_BLOCK_WEIGHT
        else:
            combined = np.concatenate(
                [
                    gap_block * TEAM_NEED_ARCHETYPE_BLOCK_WEIGHT,
                    weak_block * TEAM_NEED_WEAKNESS_BLOCK_WEIGHT,
                ]
            )

        profiles[team_id] = TeamNeedProfile(
            team_id=team_id,
            season=season,
            values=combined,
            archetype_labels=archetype_order,
            weakness_feature_names=weak_names,
        )
    return profiles
