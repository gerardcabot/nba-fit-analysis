"""Option C lineup impact: projected net rating delta and fit submetric."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_fit.features.constants import COL_PLAYER_ID, COL_TEAM_ID
from nba_fit.features.vectors import PlayerVector
from nba_fit.normalize.players import rotation_minutes_from_players
from nba_fit.models.constants import LINEUP_IMPACT_NEUTRAL_SCORE, LINEUP_IMPACT_SIGMOID_SCALE
from nba_fit.models.lineup_model import (
    LineupModelArtifacts,
    predict_lineup_net_rating,
    predict_with_player_swap,
)
from nba_fit.models.rapm import RapmArtifacts, stints_from_lineup_units
from nba_fit.scoring._similarity import sigmoid as _sigmoid


def _rapm_index(rapm: RapmArtifacts) -> dict[int, int]:
    return {int(pid): i for i, pid in enumerate(rapm.player_ids)}


def _net_rapm_for(player_id: int, rapm: RapmArtifacts) -> float:
    idx = _rapm_index(rapm).get(int(player_id))
    if idx is None:
        return 0.0
    return float(rapm.net_rapm[idx])


def bottom_rotation_player(
    team_id: int,
    player_team_map: pd.DataFrame,
    rotation_minutes: dict[int, float],
    *,
    exclude_player_ids: set[int] | None = None,
) -> int | None:
    """Lowest-minute roster player on *team_id* (proxy for displaced rotation slot)."""
    exclude = exclude_player_ids or set()
    team_col = COL_TEAM_ID if COL_TEAM_ID in player_team_map.columns else "TEAM_ID"
    pid_col = COL_PLAYER_ID if COL_PLAYER_ID in player_team_map.columns else "PLAYER_ID"
    roster = player_team_map.loc[player_team_map[team_col] == team_id, pid_col].astype(int)
    candidates = [int(pid) for pid in roster if int(pid) not in exclude]
    if not candidates:
        return None
    return min(candidates, key=lambda pid: rotation_minutes.get(pid, 0.0))


def projected_net_rating_delta(
    player: PlayerVector,
    team_id: int,
    *,
    rapm: RapmArtifacts,
    lineup_stints: pd.DataFrame,
    lineup_model: LineupModelArtifacts | None = None,
    embeddings: object | None = None,
    player_team_map: pd.DataFrame | None = None,
    rotation_minutes: dict[int, float] | None = None,
) -> float:
    """
    Minute-weighted simulation of replacing the team's bottom-rotation player.

    Returns expected change in team net rating (points per 100 possessions).
  Positive values favor adding the candidate to the destination roster.
    """
    candidate_id = int(player.player_id)
    replaced_id: int | None = None
    if player_team_map is not None and rotation_minutes is not None:
        replaced_id = bottom_rotation_player(
            team_id,
            player_team_map,
            rotation_minutes,
            exclude_player_ids={candidate_id},
        )

    if replaced_id is None and rapm.player_ids.size > 0:
        team_col = "team_id"
        if not lineup_stints.empty and team_col in lineup_stints.columns:
            team_players: set[int] = set()
            for players in lineup_stints.loc[lineup_stints[team_col] == team_id, "player_ids"]:
                team_players.update(int(p) for p in players)
            pool = [p for p in team_players if p != candidate_id]
            if pool:
                replaced_id = min(pool, key=lambda pid: _net_rapm_for(pid, rapm))

    rapm_delta = _net_rapm_for(candidate_id, rapm) - (
        _net_rapm_for(replaced_id, rapm) if replaced_id is not None else 0.0
    )

    if lineup_stints.empty or replaced_id is None:
        return rapm_delta

    stints = stints_from_lineup_units(lineup_stints) if "player_ids" not in lineup_stints.columns else lineup_stints
    total_weight = 0.0
    weighted_delta = 0.0

    for _, stint in stints.iterrows():
        if int(stint.get("team_id", 0)) != int(team_id):
            continue
        players = [int(p) for p in stint["player_ids"]]
        minutes = float(stint.get("minutes", stint.get("stint_weight", 1.0)))
        if minutes <= 0:
            continue
        if replaced_id not in players:
            continue

        if lineup_model is not None and embeddings is not None:
            from nba_fit.models.role_embeddings import RoleEmbeddingArtifacts

            assert isinstance(embeddings, RoleEmbeddingArtifacts)
            before = predict_lineup_net_rating(
                players,
                team_id,
                embeddings,
                lineup_model,
            )
            after = predict_with_player_swap(
                players,
                team_id,
                out_player_id=replaced_id,
                in_player_id=candidate_id,
                embeddings=embeddings,
                artifacts=lineup_model,
            )
            if before is not None and after is not None:
                delta = after - before
            else:
                delta = rapm_delta
        else:
            delta = rapm_delta

        weighted_delta += minutes * delta
        total_weight += minutes

    if total_weight > 0:
        return weighted_delta / total_weight
    return rapm_delta


def lineup_impact_fit(projected_delta: float) -> float:
    """Map projected net rating delta (pts/100) to a 0–1 fit submetric."""
    return _sigmoid(projected_delta * LINEUP_IMPACT_SIGMOID_SCALE)


def lineup_impact_for_pair(
    player: PlayerVector,
    team_id: int,
    *,
    rapm: RapmArtifacts | None = None,
    lineup_stints: pd.DataFrame | None = None,
    lineup_model: LineupModelArtifacts | None = None,
    embeddings: object | None = None,
    player_team_map: pd.DataFrame | None = None,
    rotation_minutes: dict[int, float] | None = None,
) -> tuple[float, float]:
    """Return ``(lineup_impact_fit, projected_net_rating_delta)``."""
    if rapm is None or lineup_stints is None:
        return LINEUP_IMPACT_NEUTRAL_SCORE, 0.0
    delta = projected_net_rating_delta(
        player,
        team_id,
        rapm=rapm,
        lineup_stints=lineup_stints,
        lineup_model=lineup_model,
        embeddings=embeddings,
        player_team_map=player_team_map,
        rotation_minutes=rotation_minutes,
    )
    return lineup_impact_fit(delta), delta
