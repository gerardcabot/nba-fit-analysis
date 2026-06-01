"""Option A interpretable submetrics (0–1 scale per player–team pair)."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from nba_fit.features.team_need import TeamNeedProfile
from nba_fit.features.vectors import PlayerVector, TeamVector
from nba_fit.models.archetypes import ArchetypeArtifacts
from nba_fit.models.role_embeddings import RoleEmbeddingArtifacts
from nba_fit.models.constants import LINEUP_IMPACT_NEUTRAL_SCORE, ROLE_FIT_NEUTRAL_SCORE
from nba_fit.scoring.constants import (
    PLAYER_DEF_GROUPS,
    PLAYER_OFF_GROUPS,
    PLAYER_ROLE_GROUPS,
    PLAYER_SHOT_GROUPS,
    PLAYER_SPACING_GROUPS,
    PLAYER_USAGE_GROUPS,
    REPLACEMENT_TEAM_GAP_WEIGHT,
    SIGMOID_SCALE,
    TEAM_DEF_NEED_GROUPS,
    TEAM_OFF_NEED_GROUPS,
    TEAM_ROLE_NEED_GROUPS,
    TEAM_SHOT_NEED_GROUPS,
    TEAM_USAGE_GROUPS,
    USAGE_CREATOR_SURPLUS_WEIGHT,
)
from nba_fit.scoring._similarity import cosine_similarity as _cosine_similarity
from nba_fit.scoring._similarity import sigmoid as _sigmoid


def _complementarity_score(
    player_strengths: np.ndarray,
    team_needs: np.ndarray,
    *,
    invert_player: tuple[int, ...] = (),
) -> float:
    """Complementarity: high when player strengths align with team needs."""
    if len(player_strengths) == 0 or len(team_needs) == 0:
        return 0.5
    p = player_strengths.copy()
    for idx in invert_player:
        if 0 <= idx < len(p):
            p[idx] = -p[idx]
    n = min(len(p), len(team_needs))
    raw = float(np.mean(p[:n] * team_needs[:n]))
    return _sigmoid(raw * SIGMOID_SCALE)


def offensive_fit(player: PlayerVector, team: TeamVector) -> float:
    return _complementarity_score(
        player.select_groups(*PLAYER_OFF_GROUPS),
        team.select_groups(*TEAM_OFF_NEED_GROUPS),
    )


def defensive_fit(player: PlayerVector, team: TeamVector) -> float:
    return _complementarity_score(
        player.select_groups(*PLAYER_DEF_GROUPS),
        team.select_groups(*TEAM_DEF_NEED_GROUPS),
    )


def role_alignment(player: PlayerVector, team: TeamVector) -> float:
    cos = _cosine_similarity(
        player.select_groups(*PLAYER_ROLE_GROUPS),
        team.select_groups(*TEAM_ROLE_NEED_GROUPS),
    )
    return (cos + 1.0) / 2.0


def usage_compatibility(player: PlayerVector, team: TeamVector) -> float:
    p = player.select_groups(*PLAYER_USAGE_GROUPS)
    t = team.select_groups(*TEAM_USAGE_GROUPS)
    if len(p) == 0 or len(t) == 0:
        return 0.5
    player_usage = float(p[0])
    headroom = float(t[0]) if len(t) > 0 else 0.0
    creator_surplus = float(t[1]) if len(t) > 1 else 0.0
    gap = headroom - player_usage + USAGE_CREATOR_SURPLUS_WEIGHT * creator_surplus
    return _sigmoid(gap * SIGMOID_SCALE)


def shot_profile_fit(player: PlayerVector, team: TeamVector) -> float:
    return _complementarity_score(
        player.select_groups(*PLAYER_SHOT_GROUPS),
        team.select_groups(*TEAM_SHOT_NEED_GROUPS),
    )


def spacing_gravity_fit(player: PlayerVector, team: TeamVector) -> float:
    p = player.select_groups(*PLAYER_SPACING_GROUPS)
    t_shot = team.select_groups(*TEAM_SHOT_NEED_GROUPS)
    if len(p) == 0:
        return 0.5
    if len(t_shot) == 0:
        return _complementarity_score(p, np.array([0.5], dtype=float))
    return _complementarity_score(p, t_shot)


def replacement_upgrade(player: PlayerVector, team: TeamVector) -> float:
    off = player.select_groups(*PLAYER_OFF_GROUPS)
    defn = player.select_groups(*PLAYER_DEF_GROUPS)
    if len(off) == 0 and len(defn) == 0:
        return 0.5
    player_level = float(np.mean(off)) if len(off) else 0.0
    player_level += float(np.mean(defn)) if len(defn) else 0.0
    team_gap = team.select_groups(*TEAM_DEF_NEED_GROUPS, *TEAM_OFF_NEED_GROUPS)
    replacement = float(np.mean(team_gap)) if len(team_gap) else 0.0
    return _sigmoid(
        (player_level - REPLACEMENT_TEAM_GAP_WEIGHT * replacement) * SIGMOID_SCALE
    )


def compute_all_submetrics(
    player: PlayerVector,
    team: TeamVector,
    *,
    team_need: TeamNeedProfile | None = None,
    embeddings: RoleEmbeddingArtifacts | None = None,
    archetypes: ArchetypeArtifacts | None = None,
    impact_context: object | None = None,
    degraded: list[str] | None = None,
) -> dict[str, float]:
    out = {
        "offensive_fit": offensive_fit(player, team),
        "defensive_fit": defensive_fit(player, team),
        "role_alignment": role_alignment(player, team),
        "usage_compatibility": usage_compatibility(player, team),
        "shot_profile_fit": shot_profile_fit(player, team),
        "spacing_gravity_fit": spacing_gravity_fit(player, team),
        "replacement_upgrade": replacement_upgrade(player, team),
    }
    from nba_fit.scoring.role_fit import team_need_fit

    if team_need is not None:
        out["team_need_fit"] = team_need_fit(
            player,
            team_need,
            embeddings=embeddings,
            archetypes=archetypes,
        )
    else:
        out["team_need_fit"] = ROLE_FIT_NEUTRAL_SCORE
        if degraded is not None:
            degraded.append("team_need_fit")

    from nba_fit.scoring.lineup_fit import lineup_impact_for_pair

    impact_ctx = impact_context
    if impact_ctx is not None:
        fit_score, _delta = lineup_impact_for_pair(
            player,
            team.team_id,
            rapm=impact_ctx.rapm,
            lineup_stints=impact_ctx.lineup_stints,
            lineup_model=impact_ctx.lineup_model,
            embeddings=impact_ctx.embeddings,
            player_team_map=impact_ctx.player_team_map,
            rotation_minutes=impact_ctx.rotation_minutes,
        )
        out["lineup_impact_fit"] = fit_score
    else:
        out["lineup_impact_fit"] = LINEUP_IMPACT_NEUTRAL_SCORE
        if degraded is not None:
            degraded.append("lineup_impact_fit")
    return out


def weighted_raw_score(submetrics: Mapping[str, float], weights: Mapping[str, float]) -> float:
    total = 0.0
    weight_sum = 0.0
    for name, weight in weights.items():
        if name in submetrics:
            total += weight * submetrics[name]
            weight_sum += weight
    if weight_sum <= 0:
        return 0.0
    return total / weight_sum
