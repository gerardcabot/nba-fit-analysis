"""Shared CLI flags for visual_tests scripts."""

from __future__ import annotations

import argparse

from nba_fit.features.season_context import DEMO_TEAM_ID

from visual_tests._season_resolve import resolve_sota_season


def add_season_team_args(parser: argparse.ArgumentParser) -> None:
    default_season = resolve_sota_season()
    parser.add_argument(
        "--season",
        default=default_season,
        help=(
            "NBA season label "
            f"(default: {default_season} when SOTA interim exists, else project default)"
        ),
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
    resolved_season = season or args.season or resolve_sota_season()
    return resolved_season, team_id if team_id is not None else args.team_id
