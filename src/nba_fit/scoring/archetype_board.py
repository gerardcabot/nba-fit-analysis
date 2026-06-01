"""Archetype target board: rank players by team need × role fit."""

from __future__ import annotations

import pandas as pd

from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.role_context import RoleFitContext
from nba_fit.scoring.role_fit import team_need_fit


def rank_archetype_board(
    team_id: int,
    role_context: RoleFitContext,
    season_context: SeasonFitContext,
    *,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Rank players by team_need_fit (archetype gaps × embedding alignment)."""
    need = role_context.team_needs.get(team_id)
    if need is None:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for pid in role_context.embeddings.player_ids:
        player = season_context.players.get(int(pid))
        if player is None:
            continue
        rows.append(
            {
                "player_id": int(pid),
                "player": role_context.player_names.get(int(pid), str(pid)),
                "archetype": role_context.archetypes.label_for(int(pid)) or "unknown",
                "role_fit": team_need_fit(
                    player,
                    need,
                    embeddings=role_context.embeddings,
                    archetypes=role_context.archetypes,
                ),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("role_fit", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    out = df[["rank", "player_id", "player", "archetype", "role_fit"]]
    if top_n is not None:
        out = out.head(top_n)
    return out.reset_index(drop=True)


def archetype_board_for_team(
    team_id: int,
    season: str | None = None,
    *,
    top_n: int | None = None,
    prefer_interim: bool = True,
    prefer_api: bool = True,
    synthetic: bool = False,
    force_retrain: bool = False,
) -> pd.DataFrame:
    season_context = SeasonFitContext.build(
        season=season,
        prefer_interim=prefer_interim and not synthetic,
        prefer_api=prefer_api and not synthetic,
    )
    if force_retrain:
        role_context = RoleFitContext.from_season(
            season_context.season,
            prefer_interim=prefer_interim,
            prefer_api=prefer_api,
            synthetic=synthetic,
            persist=True,
        )
    else:
        role_context = RoleFitContext.from_season(
            season_context.season,
            prefer_interim=prefer_interim,
            prefer_api=prefer_api,
            synthetic=synthetic,
            persist=False,
        )
    return rank_archetype_board(team_id, role_context, season_context, top_n=top_n)
