"""Registry of essential nba_api endpoints for fit modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Reliability = Literal["essential", "optional", "unreliable", "deprecated"]


@dataclass(frozen=True)
class EndpointSpec:
    """Metadata for a single stats.nba.com endpoint."""

    name: str
    category: str
    reliability: Reliability
    description: str
    # If True, bulk pulls must be chunked (e.g. playbyplayv3 per game_id)
    chunk_by_game: bool = False
    # Prefer V3 over V2 when both exist
    notes: str = ""
    default_kwargs: dict[str, Any] = field(default_factory=dict)


def _spec(
    name: str,
    category: str,
    reliability: Reliability,
    description: str,
    *,
    chunk_by_game: bool = False,
    notes: str = "",
    **kwargs: Any,
) -> EndpointSpec:
    return EndpointSpec(
        name=name,
        category=category,
        reliability=reliability,
        description=description,
        chunk_by_game=chunk_by_game,
        notes=notes,
        default_kwargs=dict(kwargs),
    )


# Essential families from project plan (docs/NBA_ENDPOINTS_DATA_REFERENCE.md)
ENDPOINT_REGISTRY: dict[str, EndpointSpec] = {
    # Identity and rosters
    "commonallplayers": _spec(
        "commonallplayers",
        "identity",
        "essential",
        "League player directory for ID joins.",
        is_only_current_season=1,
        league_id="00",
    ),
    "commonteamroster": _spec(
        "commonteamroster",
        "identity",
        "essential",
        "Team roster by season.",
    ),
    "commonplayerinfo": _spec(
        "commonplayerinfo",
        "identity",
        "essential",
        "Biographical and draft info per player.",
    ),
    "leaguedashplayerbiostats": _spec(
        "leaguedashplayerbiostats",
        "identity",
        "essential",
        "Height, weight, age, position for physical features.",
    ),
    # Player baselines
    "leaguedashplayerstats": _spec(
        "leaguedashplayerstats",
        "player_baseline",
        "essential",
        "Season player box/Advanced aggregates.",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    ),
    "playerestimatedmetrics": _spec(
        "playerestimatedmetrics",
        "player_baseline",
        "essential",
        "NBA estimated ORtg/DRtg/NetRtg and usage.",
    ),
    "playergamelogs": _spec(
        "playergamelogs",
        "player_baseline",
        "essential",
        "Game-level player logs for availability and trends.",
        season_type_all_star="Regular Season",
    ),
    "playergamelog": _spec(
        "playergamelog",
        "player_baseline",
        "essential",
        "Single-player game log.",
        season_type_all_star="Regular Season",
    ),
    "playerprofilev2": _spec(
        "playerprofilev2",
        "player_baseline",
        "optional",
        "Career/season profile tables (some overlap with dash).",
    ),
    # Team baselines
    "leaguedashteamstats": _spec(
        "leaguedashteamstats",
        "team_baseline",
        "essential",
        "Season team aggregates.",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    ),
    "teamestimatedmetrics": _spec(
        "teamestimatedmetrics",
        "team_baseline",
        "essential",
        "Team estimated ratings.",
    ),
    "teamgamelogs": _spec(
        "teamgamelogs",
        "team_baseline",
        "essential",
        "Team game logs.",
        season_type_all_star="Regular Season",
    ),
    "teamdashboardbygeneralsplits": _spec(
        "teamdashboardbygeneralsplits",
        "team_baseline",
        "essential",
        "Team split dashboards (home/road, etc.).",
    ),
    # Shot geography
    "shotchartdetail": _spec(
        "shotchartdetail",
        "shot_geography",
        "essential",
        "Shot-level coordinates for gravity/spacing.",
        context_measure_simple="FGA",
    ),
    "shotchartleaguewide": _spec(
        "shotchartleaguewide",
        "shot_geography",
        "optional",
        "League-wide shot chart sample.",
    ),
    "leaguedashplayershotlocations": _spec(
        "leaguedashplayershotlocations",
        "shot_geography",
        "essential",
        "Zone shooting frequencies by player.",
    ),
    "leaguedashteamshotlocations": _spec(
        "leaguedashteamshotlocations",
        "shot_geography",
        "essential",
        "Zone shooting by team.",
    ),
    # Shooting behavior (tracking dashboards)
    "playerdashptshots": _spec(
        "playerdashptshots",
        "shooting_behavior",
        "essential",
        "Player tracking shot splits (touch time, dribbles).",
    ),
    "leaguedashplayerptshot": _spec(
        "leaguedashplayerptshot",
        "shooting_behavior",
        "essential",
        "League player tracking shot table.",
    ),
    "leaguedashteamptshot": _spec(
        "leaguedashteamptshot",
        "shooting_behavior",
        "essential",
        "League team tracking shot table.",
    ),
    "teamdashptshots": _spec(
        "teamdashptshots",
        "shooting_behavior",
        "essential",
        "Team tracking shot splits.",
    ),
    # Passing
    "playerdashptpass": _spec(
        "playerdashptpass",
        "passing",
        "essential",
        "Player passing tracking.",
    ),
    "teamdashptpass": _spec(
        "teamdashptpass",
        "passing",
        "essential",
        "Team passing tracking.",
    ),
    # Rebounding / hustle
    "playerdashptreb": _spec(
        "playerdashptreb",
        "rebounding",
        "essential",
        "Player rebounding tracking.",
    ),
    "teamdashptreb": _spec(
        "teamdashptreb",
        "rebounding",
        "essential",
        "Team rebounding tracking.",
    ),
    "leaguehustlestatsplayer": _spec(
        "leaguehustlestatsplayer",
        "hustle",
        "essential",
        "Deflections, charges, loose balls.",
    ),
    "leaguehustlestatsteam": _spec(
        "leaguehustlestatsteam",
        "hustle",
        "essential",
        "Team hustle aggregates.",
    ),
    # Defense proxies
    "leaguedashptdefend": _spec(
        "leaguedashptdefend",
        "defense",
        "essential",
        "Player defended-shot frequencies.",
    ),
    "leaguedashptteamdefend": _spec(
        "leaguedashptteamdefend",
        "defense",
        "essential",
        "Team defended-shot table.",
    ),
    "boxscoredefensivev2": _spec(
        "boxscoredefensivev2",
        "defense",
        "optional",
        "Game-level defensive box (per game_id).",
        chunk_by_game=True,
    ),
    "boxscorematchupsv3": _spec(
        "boxscorematchupsv3",
        "defense",
        "essential",
        "Matchup minutes and outcomes.",
        chunk_by_game=True,
    ),
    # Lineups and on/off
    "leaguedashlineups": _spec(
        "leaguedashlineups",
        "lineups",
        "essential",
        "Five-man lineup performance (Advanced measure for OFF/DEF/NET ratings).",
        measure_type_detailed_defense="Advanced",
    ),
    "teamdashlineups": _spec(
        "teamdashlineups",
        "lineups",
        "essential",
        "Team-specific lineup table (Advanced measure for OFF/DEF/NET ratings).",
        measure_type_detailed_defense="Advanced",
    ),
    "teamplayeronoffsummary": _spec(
        "teamplayeronoffsummary",
        "on_off",
        "essential",
        "On/off summary for roster fit.",
    ),
    "teamplayeronoffdetails": _spec(
        "teamplayeronoffdetails",
        "on_off",
        "essential",
        "Detailed on/off splits.",
    ),
    "gamerotation": _spec(
        "gamerotation",
        "lineups",
        "essential",
        "Stint/rotation log per game.",
        chunk_by_game=True,
    ),
    # Event streams and schedules
    "playbyplayv3": _spec(
        "playbyplayv3",
        "play_by_play",
        "essential",
        "Official play-by-play (preferred over v2).",
        chunk_by_game=True,
        notes="Chunk bulk pulls by game_id; one game per request.",
    ),
    "leaguegamefinder": _spec(
        "leaguegamefinder",
        "schedule",
        "essential",
        "Game finder for game_id lists.",
    ),
    "leaguegamelog": _spec(
        "leaguegamelog",
        "schedule",
        "essential",
        "League game log index.",
        player_or_team_abbreviation="T",
    ),
    "scheduleleaguev2": _spec(
        "scheduleleaguev2",
        "schedule",
        "essential",
        "Full season schedule.",
        league_id="00",
    ),
    "boxscoresummaryv3": _spec(
        "boxscoresummaryv3",
        "box_score",
        "essential",
        "Game summary metadata.",
        chunk_by_game=True,
    ),
    # Game-level advanced box scores
    "boxscoreadvancedv3": _spec(
        "boxscoreadvancedv3",
        "box_score",
        "essential",
        "Advanced box per game.",
        chunk_by_game=True,
    ),
    "boxscorefourfactorsv3": _spec(
        "boxscorefourfactorsv3",
        "box_score",
        "essential",
        "Four factors per game.",
        chunk_by_game=True,
    ),
    "boxscoremiscv3": _spec(
        "boxscoremiscv3",
        "box_score",
        "optional",
        "Misc box stats per game.",
        chunk_by_game=True,
    ),
    "boxscorescoringv3": _spec(
        "boxscorescoringv3",
        "box_score",
        "optional",
        "Scoring breakdown per game.",
        chunk_by_game=True,
    ),
    "boxscoreusagev3": _spec(
        "boxscoreusagev3",
        "box_score",
        "essential",
        "Usage box per game.",
        chunk_by_game=True,
    ),
    "boxscoreplayertrackv3": _spec(
        "boxscoreplayertrackv3",
        "box_score",
        "essential",
        "Player tracking box per game.",
        chunk_by_game=True,
    ),
    # Unreliable / deprecated (documented from local probe)
    "synergyplaytypes": _spec(
        "synergyplaytypes",
        "play_type",
        "unreliable",
        "Synergy play-type labels — empty in 2025-26 probe.",
        notes="Do not depend on Synergy; proxy play types from usage/shot/passing.",
        player_or_team_abbreviation="P",
    ),
    "playbyplayv2": _spec(
        "playbyplayv2",
        "play_by_play",
        "deprecated",
        "Legacy PBP; use playbyplayv3.",
        chunk_by_game=True,
    ),
    "playbyplay": _spec(
        "playbyplay",
        "play_by_play",
        "deprecated",
        "Legacy PBP v1.",
        chunk_by_game=True,
    ),
    # Optional / experimental
    "gravityleaders": _spec(
        "gravityleaders",
        "experimental",
        "optional",
        "Gravity leaderboard if stable for season.",
    ),
    "fantasywidget": _spec(
        "fantasywidget",
        "experimental",
        "optional",
        "Fantasy widget — not core to fit model.",
    ),
    "draftcombineplayeranthro": _spec(
        "draftcombineplayeranthro",
        "draft",
        "optional",
        "Combine anthropometrics.",
    ),
    "draftcombinedrillresults": _spec(
        "draftcombinedrillresults",
        "draft",
        "optional",
        "Combine athletic drills.",
    ),
    "draftcombinespotshooting": _spec(
        "draftcombinespotshooting",
        "draft",
        "optional",
        "Combine shooting results.",
    ),
    "drafthistory": _spec(
        "drafthistory",
        "draft",
        "optional",
        "Historical draft picks.",
        league_id="00",
    ),
}


ESSENTIAL_ENDPOINTS: tuple[str, ...] = tuple(
    name for name, spec in ENDPOINT_REGISTRY.items() if spec.reliability == "essential"
)
