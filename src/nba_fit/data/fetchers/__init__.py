"""Endpoint-specific fetch helpers built on :class:`NBAClient`."""

from nba_fit.data.fetchers.league_dash import (
    fetch_league_dash,
    fetch_option_a_mvp,
    league_dash_kwargs,
    primary_dataset_name,
    primary_frame,
)

__all__ = [
    "fetch_league_dash",
    "fetch_option_a_mvp",
    "league_dash_kwargs",
    "primary_dataset_name",
    "primary_frame",
]
