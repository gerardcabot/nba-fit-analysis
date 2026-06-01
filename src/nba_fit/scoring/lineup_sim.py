"""Lineup simulation product API (Option C)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.constants import LINEUP_SIM_TOP_N
from nba_fit.models.impact_context import ImpactFitContext
from nba_fit.models.lineup_impact import LineupUnitProjection, aggregate_projected_delta
from nba_fit.models.lineup_model import predict_with_player_swap
from nba_fit.scoring.lineup_fit import bottom_rotation_player, projected_net_rating_delta


@dataclass
class LineupSimResult:
    """Top projected five-man units for a player on a destination team."""

    player_id: int
    team_id: int
    season: str
    projected_net_rating_delta: float | None
    top_lineups: list[LineupUnitProjection] = field(default_factory=list)
    data_source: str = "synthetic"
    note: str = ""

    def to_table(self) -> pd.DataFrame:
        """Tabular view for CLI printing."""
        rows: list[dict[str, Any]] = []
        for rank, unit in enumerate(self.top_lineups, start=1):
            rows.append(
                {
                    "rank": rank,
                    "lineup": unit.lineup_label,
                    "baseline_net_rating": round(unit.baseline_net_rating, 2),
                    "projected_net_rating_delta": round(
                        unit.projected_net_rating_delta, 2
                    ),
                    "minutes": round(unit.minutes, 1),
                    "replaced_player_id": unit.replaced_player_id,
                }
            )
        return pd.DataFrame(rows)

    def lineup_synergy_block(self) -> dict[str, Any]:
        """Fit-card ``lineup_synergy`` payload."""
        return {
            "projected_net_rating_delta": self.projected_net_rating_delta,
            "top_lineups": [
                {
                    "lineup_key": u.lineup_key,
                    "lineup_label": u.lineup_label,
                    "player_ids": list(u.player_ids),
                    "baseline_net_rating": u.baseline_net_rating,
                    "projected_net_rating_delta": u.projected_net_rating_delta,
                    "minutes": u.minutes,
                    "replaced_player_id": u.replaced_player_id,
                }
                for u in self.top_lineups
            ],
            "data_source": self.data_source,
            "note": self.note,
        }


def _stint_label(players: list[int]) -> str:
    return " - ".join(str(p) for p in players)


def _per_stint_delta(
    players: list[int],
    team_id: int,
    candidate_id: int,
    replaced_id: int,
    *,
    lineup_ctx: ImpactFitContext,
) -> float:
    rapm = lineup_ctx.rapm
    idx = {int(pid): i for i, pid in enumerate(rapm.player_ids)}

    def _net(pid: int) -> float:
        hit = idx.get(pid)
        return float(rapm.net_rapm[hit]) if hit is not None else 0.0

    rapm_delta = _net(candidate_id) - _net(replaced_id)

    if lineup_ctx.lineup_model is not None and lineup_ctx.embeddings is not None:
        before = predict_with_player_swap(
            players,
            team_id,
            out_player_id=replaced_id,
            in_player_id=replaced_id,
            embeddings=lineup_ctx.embeddings,
            artifacts=lineup_ctx.lineup_model,
        )
        after = predict_with_player_swap(
            players,
            team_id,
            out_player_id=replaced_id,
            in_player_id=candidate_id,
            embeddings=lineup_ctx.embeddings,
            artifacts=lineup_ctx.lineup_model,
        )
        if before is not None and after is not None:
            return after - before
    return rapm_delta


def _top_lineup_projections(
    player: object,
    team_id: int,
    *,
    lineup_ctx: ImpactFitContext,
    top_n: int,
) -> list[LineupUnitProjection]:
    from nba_fit.features.vectors import PlayerVector

    assert isinstance(player, PlayerVector)
    candidate_id = int(player.player_id)
    rotation = lineup_ctx.rotation_minutes
    replaced_id = bottom_rotation_player(
        team_id,
        lineup_ctx.player_team_map,
        rotation,
        exclude_player_ids={candidate_id},
    )
    if replaced_id is None:
        return []

    projections: list[LineupUnitProjection] = []
    stints = lineup_ctx.lineup_stints
    if stints.empty:
        return projections

    for _, stint in stints.iterrows():
        if int(stint.get("team_id", 0)) != int(team_id):
            continue
        players = [int(p) for p in stint["player_ids"]]
        if replaced_id not in players:
            continue
        minutes = float(stint.get("minutes", stint.get("stint_weight", 1.0)))
        baseline = float(stint.get("net_rating", 0.0))
        if pd.isna(baseline):
            baseline = 0.0
        delta = _per_stint_delta(
            players,
            team_id,
            candidate_id,
            replaced_id,
            lineup_ctx=lineup_ctx,
        )
        swapped = [candidate_id if p == replaced_id else p for p in players]
        projections.append(
            LineupUnitProjection(
                lineup_key="-".join(str(p) for p in sorted(swapped)),
                lineup_label=_stint_label(swapped),
                player_ids=tuple(swapped),
                baseline_net_rating=baseline,
                projected_net_rating_delta=delta,
                minutes=minutes,
                replaced_player_id=replaced_id,
            )
        )

    projections.sort(key=lambda p: p.projected_net_rating_delta, reverse=True)
    return projections[:top_n]


def run_lineup_sim(
    player_id: int,
    team_id: int,
    season: str | None = None,
    *,
    top_n: int = LINEUP_SIM_TOP_N,
    prefer_interim: bool = True,
    prefer_api: bool = True,
    synthetic: bool = False,
    impact_context: ImpactFitContext | None = None,
) -> LineupSimResult:
    """
    Project top five-man units and net-rating deltas for *player_id* on *team_id*.

    Uses RAPM + optional lineup net-rating model when artifacts or interim tables
    exist; falls back to :meth:`ImpactFitContext.from_synthetic` for offline demos.
    """
    context = SeasonFitContext.build(
        season,
        prefer_interim=prefer_interim and not synthetic,
        prefer_api=prefer_api and not synthetic,
    )
    if synthetic:
        context = SeasonFitContext.from_synthetic(context.season, n_players=120)

    season = context.season
    player = context.players.get(player_id)
    if player is None:
        return LineupSimResult(
            player_id=player_id,
            team_id=team_id,
            season=season,
            projected_net_rating_delta=None,
            note=f"Unknown player_id={player_id}",
        )

    note = (
        "Minute-weighted lineup deltas from RAPM swap + optional embedding lineup "
        "model (train roles + ingest --tier role for live data)."
    )
    data_source = "synthetic" if synthetic else "rapm"

    lctx = impact_context
    if lctx is None:
        try:
            lctx = ImpactFitContext.from_season(
                season,
                prefer_interim=prefer_interim and not synthetic,
                synthetic=synthetic,
            )
            data_source = "synthetic" if synthetic else "lineup_units"
        except (FileNotFoundError, ValueError) as exc:
            try:
                from nba_fit.models.role_context import RoleFitContext

                role_ctx = RoleFitContext.from_synthetic(context)
                lctx = ImpactFitContext.from_synthetic(role_ctx)
                data_source = "synthetic"
                note = f"Synthetic Option C fallback ({exc})."
            except ValueError as syn_exc:
                return LineupSimResult(
                    player_id=player_id,
                    team_id=team_id,
                    season=season,
                    projected_net_rating_delta=None,
                    note=str(syn_exc),
                )

    aggregate_delta = projected_net_rating_delta(
        player,
        team_id,
        rapm=lctx.rapm,
        lineup_stints=lctx.lineup_stints,
        lineup_model=lctx.lineup_model,
        embeddings=lctx.embeddings,
        player_team_map=lctx.player_team_map,
        rotation_minutes=lctx.rotation_minutes,
    )
    top_lineups = _top_lineup_projections(
        player,
        team_id,
        lineup_ctx=lctx,
        top_n=top_n,
    )
    if not top_lineups and aggregate_delta is not None:
        replaced = bottom_rotation_player(
            team_id,
            lctx.player_team_map,
            lctx.rotation_minutes,
            exclude_player_ids={player_id},
        )
        top_lineups = [
            LineupUnitProjection(
                lineup_key=f"team_{team_id}_swap",
                lineup_label=f"rotation swap (replaces {replaced})",
                player_ids=(player_id,),
                baseline_net_rating=0.0,
                projected_net_rating_delta=aggregate_delta,
                minutes=1.0,
                replaced_player_id=replaced,
            )
        ]

    unit_agg = aggregate_projected_delta(top_lineups) if top_lineups else aggregate_delta

    return LineupSimResult(
        player_id=player_id,
        team_id=team_id,
        season=season,
        projected_net_rating_delta=unit_agg,
        top_lineups=top_lineups,
        data_source=data_source,
        note=note,
    )
