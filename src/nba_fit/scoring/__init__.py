"""Player-team scoring API (Option A + B + C + D ensemble)."""

from nba_fit.scoring.archetype_board import archetype_board_for_team, rank_archetype_board
from nba_fit.scoring.constants import (
    ENSEMBLE_COMPONENT_NAMES,
    ENSEMBLE_COMPONENT_WEIGHTS,
    SUBMETRIC_NAMES,
    SUBMETRIC_WEIGHTS,
)
from nba_fit.scoring.ensemble import calibrated_ensemble, extract_ensemble_components
from nba_fit.scoring.fit_card import build_fit_card, fit_card_to_json
from nba_fit.scoring.fit_index import FitIndexTable, build_fit_index_table
from nba_fit.scoring.lineup_sim import LineupSimResult, run_lineup_sim
from nba_fit.scoring.ranker import (
    FitRanker,
    rank_destinations_for_player,
    rank_players_for_team,
)
from nba_fit.scoring.submetrics import compute_all_submetrics
from nba_fit.scoring.uncertainty import (
    UncertaintyResult,
    bootstrap_disagreement_ci,
    fit_uncertainty,
    sample_size_penalty,
)

__all__ = [
    "ENSEMBLE_COMPONENT_NAMES",
    "ENSEMBLE_COMPONENT_WEIGHTS",
    "SUBMETRIC_NAMES",
    "SUBMETRIC_WEIGHTS",
    "FitIndexTable",
    "FitRanker",
    "LineupSimResult",
    "archetype_board_for_team",
    "build_fit_card",
    "build_fit_index_table",
    "calibrated_ensemble",
    "compute_all_submetrics",
    "extract_ensemble_components",
    "fit_card_to_json",
    "rank_archetype_board",
    "rank_destinations_for_player",
    "rank_players_for_team",
    "run_lineup_sim",
    "UncertaintyResult",
    "bootstrap_disagreement_ci",
    "fit_uncertainty",
    "sample_size_penalty",
]
