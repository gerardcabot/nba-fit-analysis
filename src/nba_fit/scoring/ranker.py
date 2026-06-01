"""Rank player destinations and team acquisition targets."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.impact_context import ImpactFitContext
from nba_fit.models.role_context import RoleFitContext
from nba_fit.scoring.fit_card import build_fit_card
from nba_fit.scoring.fit_index import FitIndexTable, build_fit_index_table


@dataclass
class FitRanker:
    """Season-level rankings from calibrated fit index."""

    context: SeasonFitContext
    table: FitIndexTable = field(default_factory=FitIndexTable)
    impact_context: ImpactFitContext | None = None

    @classmethod
    def from_season(
        cls,
        season: str | None = None,
        *,
        prefer_interim: bool = True,
        prefer_api: bool = True,
        synthetic: bool = False,
    ) -> FitRanker:
        context = SeasonFitContext.build(
            season=season,
            prefer_interim=prefer_interim and not synthetic,
            prefer_api=prefer_api and not synthetic,
        )
        role_context: RoleFitContext | None = None
        impact_context: ImpactFitContext | None = None
        if not synthetic and prefer_interim:
            try:
                role_context = RoleFitContext.from_season(
                    context.season,
                    prefer_interim=True,
                    prefer_api=prefer_api,
                    synthetic=False,
                    persist=False,
                )
            except (FileNotFoundError, ValueError):
                role_context = None
            try:
                impact_context = ImpactFitContext.from_season(
                    context.season,
                    prefer_interim=True,
                    synthetic=False,
                    persist=False,
                )
            except (FileNotFoundError, ValueError):
                impact_context = None
        table = build_fit_index_table(
            context,
            role_context=role_context,
            impact_context=impact_context,
            synthetic=synthetic,
        )
        return cls(context=context, table=table, impact_context=impact_context)

    def rank_destinations_for_player(
        self,
        player_id: int,
        *,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """Rank all teams for one player by overall_fit_percentile."""
        if self.table.pairs.empty:
            return pd.DataFrame()
        sub = self.table.pairs[self.table.pairs["player_id"] == player_id].copy()
        if sub.empty:
            return pd.DataFrame()
        sub = sub.sort_values("overall_fit_percentile", ascending=False)
        sub["rank"] = range(1, len(sub) + 1)
        team_names = {
            tid: t.display_name for tid, t in self.context.teams.items()
        }
        sub["team"] = sub["team_id"].map(lambda tid: team_names.get(int(tid), str(tid)))
        cols = ["rank", "team_id", "team", "overall_fit_percentile", "raw_fit_score"]
        cols += [c for c in sub.columns if c not in cols and c not in ("player_id", "season")]
        out = sub[cols]
        if top_n is not None:
            out = out.head(top_n)
        return out.reset_index(drop=True)

    def rank_players_for_team(
        self,
        team_id: int,
        *,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """Rank all players for one team by overall_fit_percentile."""
        if self.table.pairs.empty:
            return pd.DataFrame()
        sub = self.table.pairs[self.table.pairs["team_id"] == team_id].copy()
        if sub.empty:
            return pd.DataFrame()
        sub = sub.sort_values("overall_fit_percentile", ascending=False)
        sub["rank"] = range(1, len(sub) + 1)
        cols = ["rank", "player_id", "overall_fit_percentile", "raw_fit_score"]
        cols += [c for c in sub.columns if c not in cols and c not in ("team_id", "season")]
        out = sub[cols]
        if top_n is not None:
            out = out.head(top_n)
        return out.reset_index(drop=True)

    def fit_card(
        self,
        player_id: int,
        team_id: int,
        *,
        role_context: RoleFitContext | None = None,
    ) -> dict:
        return build_fit_card(
            player_id,
            team_id,
            self.table,
            self.context,
            role_context=role_context,
            impact_context=self.impact_context,
        )


def rank_destinations_for_player(
    player_id: int,
    season: str | None = None,
    *,
    top_n: int | None = None,
    prefer_interim: bool = True,
    prefer_api: bool = True,
) -> pd.DataFrame:
    ranker = FitRanker.from_season(
        season, prefer_interim=prefer_interim, prefer_api=prefer_api
    )
    return ranker.rank_destinations_for_player(player_id, top_n=top_n)


def rank_players_for_team(
    team_id: int,
    season: str | None = None,
    *,
    top_n: int | None = None,
    prefer_interim: bool = True,
    prefer_api: bool = True,
) -> pd.DataFrame:
    ranker = FitRanker.from_season(
        season, prefer_interim=prefer_interim, prefer_api=prefer_api
    )
    return ranker.rank_players_for_team(team_id, top_n=top_n)
