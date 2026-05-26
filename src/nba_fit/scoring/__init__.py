"""Player-team scoring API (Option A + Option B + Option C)."""

from nba_fit.scoring.archetype_board import archetype_board_for_team, rank_archetype_board
from nba_fit.scoring.constants import SUBMETRIC_NAMES, SUBMETRIC_WEIGHTS
from nba_fit.scoring.fit_index import FitIndexTable, build_fit_index_table
from nba_fit.scoring.lineup_sim import LineupSimResult, run_lineup_sim
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
    "LineupSimResult",
    "archetype_board_for_team",
    "build_fit_card",
    "build_fit_index_table",
    "compute_all_submetrics",
    "fit_card_to_json",
    "rank_archetype_board",
    "rank_destinations_for_player",
    "rank_players_for_team",
    "run_lineup_sim",
]
