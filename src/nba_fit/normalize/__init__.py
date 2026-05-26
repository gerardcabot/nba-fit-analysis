"""Canonical schemas and ID normalization."""

from nba_fit.normalize.ids import canonical_game_id, canonical_player_id, canonical_team_id
from nba_fit.normalize.players import (
    build_players_table,
    interim_players_path,
    load_players_table,
    write_players_table,
)
from nba_fit.normalize.teams import (
    build_teams_table,
    interim_teams_path,
    load_teams_table,
    write_teams_table,
)

__all__ = [
    "canonical_player_id",
    "canonical_team_id",
    "canonical_game_id",
    "build_players_table",
    "write_players_table",
    "load_players_table",
    "interim_players_path",
    "build_teams_table",
    "write_teams_table",
    "load_teams_table",
    "interim_teams_path",
]
