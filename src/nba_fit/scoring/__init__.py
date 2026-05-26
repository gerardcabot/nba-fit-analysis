"""Player-team scoring API (Option A interpretable fit index)."""

from nba_fit.scoring.constants import SUBMETRIC_NAMES, SUBMETRIC_WEIGHTS
from nba_fit.scoring.fit_card import build_fit_card, fit_card_to_json
from nba_fit.scoring.fit_index import FitIndexTable, build_fit_index_table
from nba_fit.scoring.ranker import (
    FitRanker,
    rank_destinations_for_player,
    rank_players_for_team,
)
from nba_fit.scoring.submetrics import compute_all_submetrics

__all__ = [
    "SUBMETRIC_NAMES",
    "SUBMETRIC_WEIGHTS",
    "FitIndexTable",
    "FitRanker",
    "build_fit_card",
    "build_fit_index_table",
    "compute_all_submetrics",
    "fit_card_to_json",
    "rank_destinations_for_player",
    "rank_players_for_team",
]
