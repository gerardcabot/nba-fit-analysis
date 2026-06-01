"""Endpoint-specific fetch helpers built on :class:`NBAClient`."""

from nba_fit.data.fetchers.league_dash import (
    fetch_league_dash,
    fetch_option_a_mvp,
    league_dash_kwargs,
    primary_dataset_name,
    primary_frame,
)
from nba_fit.data.fetchers.lineups_onoff import (
    fetch_lineup_onoff,
    fetch_option_b_role,
    lineup_onoff_kwargs,
    lineup_units_frame,
)
from nba_fit.data.fetchers.pbp import (
    fetch_gamerotation_batch,
    fetch_gamerotation_game,
    fetch_league_game_finder,
    fetch_option_c_impact,
    fetch_playbyplay_batch,
    fetch_playbyplay_game,
    league_game_ids,
    playbyplay_frame,
)
from nba_fit.data.fetchers.tactical import (
    fetch_option_d_tactical,
    fetch_tactical,
    primary_dataset_name as tactical_primary_dataset_name,
    primary_frame as tactical_primary_frame,
    tactical_kwargs,
)
from nba_fit.data.fetchers.transactions import (
    MovementEvent,
    ensure_stub_template,
    fetch_movements_from_gamelogs,
    get_movements,
    load_manual_movements,
    synthetic_movements,
)

__all__ = [
    "fetch_league_dash",
    "fetch_option_a_mvp",
    "league_dash_kwargs",
    "primary_dataset_name",
    "primary_frame",
    "fetch_lineup_onoff",
    "fetch_option_b_role",
    "lineup_onoff_kwargs",
    "lineup_units_frame",
    "fetch_league_game_finder",
    "league_game_ids",
    "fetch_playbyplay_game",
    "fetch_playbyplay_batch",
    "fetch_gamerotation_game",
    "fetch_gamerotation_batch",
    "fetch_option_c_impact",
    "playbyplay_frame",
    "fetch_option_d_tactical",
    "fetch_tactical",
    "tactical_kwargs",
    "tactical_primary_dataset_name",
    "tactical_primary_frame",
    "MovementEvent",
    "ensure_stub_template",
    "fetch_movements_from_gamelogs",
    "get_movements",
    "load_manual_movements",
    "synthetic_movements",
]
