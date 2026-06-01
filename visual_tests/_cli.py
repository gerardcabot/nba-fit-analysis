"""Shared CLI flags for visual_tests scripts."""

from __future__ import annotations

import argparse

from nba_fit.config.settings import DEFAULT_SEASON
from nba_fit.features.season_context import DEMO_TEAM_ID


def add_season_team_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--season",
        default=DEFAULT_SEASON,
        help=f"NBA season label (default: {DEFAULT_SEASON})",
    )
    parser.add_argument(
        "--team-id",
        type=int,
        default=DEMO_TEAM_ID,
        help=f"NBA team ID (default: {DEMO_TEAM_ID})",
    )


def parse_season_team(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_season_team_args(parser)
    return parser.parse_args(argv)


def resolve_season_team(
    *,
    season: str | None = None,
    team_id: int | None = None,
    argv: list[str] | None = None,
) -> tuple[str, int]:
    if season is not None and team_id is not None:
        return season, team_id
    args = parse_season_team(argv)
    return season or args.season, team_id if team_id is not None else args.team_id
