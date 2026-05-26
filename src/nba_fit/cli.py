"""CLI entry point: `python -m nba_fit` or `nba-fit`."""

from __future__ import annotations

import argparse
import sys

from nba_fit.config.endpoints import ENDPOINT_REGISTRY, ESSENTIAL_ENDPOINTS
from nba_fit.config.settings import INGEST_TIER_MVP, get_settings
from nba_fit.data.client import NBAClient
from nba_fit.data.ingest import run_ingest
from nba_fit.data.registry import ProbeRegistry
from nba_fit.features.season_context import DEMO_PLAYER_ID, DEMO_TEAM_ID
from nba_fit.scoring.fit_card import fit_card_to_json
from nba_fit.scoring.ranker import FitRanker


def _cmd_health(_args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        registry = ProbeRegistry.load(settings=settings)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    summary = registry.summary()
    meta = registry.meta
    print("NBA Fit — endpoint health (from probe_all_results.json)")
    print(f"  Probe timestamp: {meta.get('timestamp', 'unknown')}")
    print(f"  Season:          {meta.get('season', 'unknown')}")
    print(f"  Totals:          OK={summary['OK']} EMPTY={summary['EMPTY']} FAIL={summary['FAIL']}")
    print()
    print("Essential registry endpoints:")
    for name in ESSENTIAL_ENDPOINTS:
        spec = ENDPOINT_REGISTRY[name]
        probed = registry.get(name)
        if probed is None:
            status = "NOT_PROBED"
            rows = "-"
        else:
            status = probed.status
            rows = probed.total_rows
        flag = ""
        if spec.reliability == "unreliable":
            flag = " [UNRELIABLE]"
        print(f"  {name:32} {status:10} rows={rows}{flag}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    season = args.season or settings.default_season
    print(f"Ingest tier={args.tier} season={season} (cache={'on' if not args.no_cache else 'off'})...")
    try:
        result = run_ingest(
            season=season,
            tier=args.tier,
            use_cache=not args.no_cache,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Ingest failed: {exc}", file=sys.stderr)
        return 1

    print(f"  players: {result.player_rows} rows -> {result.players_path}")
    print(f"  teams:   {result.team_rows} rows -> {result.teams_path}")
    for endpoint, fetch in result.fetched.items():
        rows = sum(len(df) for df in fetch.frames.values())
        cache_flag = "cache" if fetch.from_cache else "live"
        print(f"    {endpoint}: {rows} rows ({cache_flag})")
    return 0


def _cmd_fetch_sample(args: argparse.Namespace) -> int:
    endpoint = args.endpoint
    season = args.season
    if endpoint not in ENDPOINT_REGISTRY:
        print(f"Unknown endpoint: {endpoint}", file=sys.stderr)
        return 1

    client = NBAClient()
    print(f"Fetching {endpoint} season={season} (cache={'on' if not args.no_cache else 'off'})...")
    try:
        result = client.fetch(
            endpoint,
            season=season,
            use_cache=not args.no_cache,
            persist_cache=not args.no_cache,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Fetch failed: {exc}", file=sys.stderr)
        return 1

    total_rows = sum(len(df) for df in result.frames.values())
    print(f"  from_cache={result.from_cache} elapsed={result.elapsed_sec:.2f}s rows={total_rows}")
    for ds_name, df in result.frames.items():
        print(f"    {ds_name}: {len(df)} rows, {len(df.columns)} cols")
    if result.cache_path:
        print(f"  cache: {result.cache_path}")
    return 0


def _cmd_rank_player(args: argparse.Namespace) -> int:
    player_id = int(args.player_id)
    ranker = FitRanker.from_season(
        args.season,
        prefer_interim=not args.synthetic,
        prefer_api=not args.synthetic,
    )
    rankings = ranker.rank_destinations_for_player(player_id, top_n=args.top)
    if rankings.empty:
        print(f"No rankings for player_id={player_id} (season={args.season})", file=sys.stderr)
        return 1
    print(
        f"Top team destinations for player {player_id} "
        f"({args.season}, source={ranker.context.source})"
    )
    print(rankings.to_string(index=False))
    if args.fit_card_team:
        card = ranker.fit_card(player_id, int(args.fit_card_team))
        print()
        print(fit_card_to_json(card))
    return 0


def _cmd_rank_team(args: argparse.Namespace) -> int:
    team_id = int(args.team_id)
    ranker = FitRanker.from_season(
        args.season,
        prefer_interim=not args.synthetic,
        prefer_api=not args.synthetic,
    )
    rankings = ranker.rank_players_for_team(team_id, top_n=args.top)
    if rankings.empty:
        print(f"No rankings for team_id={team_id} (season={args.season})", file=sys.stderr)
        return 1
    print(
        f"Top player targets for team {team_id} "
        f"({args.season}, source={ranker.context.source})"
    )
    print(rankings.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nba-fit",
        description="NBA player-team fit analysis — foundation CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    health = sub.add_parser("health", help="Show probe-based endpoint health")
    health.set_defaults(func=_cmd_health)

    fetch = sub.add_parser("fetch-sample", help="Smoke-fetch one nba_api endpoint")
    fetch.add_argument(
        "--endpoint",
        default="leaguedashplayerstats",
        help="Endpoint module name (default: leaguedashplayerstats)",
    )
    fetch.add_argument("--season", default=get_settings().default_season)
    fetch.add_argument("--no-cache", action="store_true", help="Bypass Parquet cache")
    fetch.set_defaults(func=_cmd_fetch_sample)

    ingest = sub.add_parser(
        "ingest",
        help="Fetch Option A league-dash endpoints and write interim Parquet",
    )
    ingest.add_argument(
        "--season",
        default=get_settings().default_season,
        help="NBA season string (default from config)",
    )
    ingest.add_argument(
        "--tier",
        default=INGEST_TIER_MVP,
        choices=[INGEST_TIER_MVP],
        help="Ingest bundle (mvp = Option A league-dash core)",
    )
    ingest.add_argument("--no-cache", action="store_true", help="Bypass Parquet cache")
    ingest.set_defaults(func=_cmd_ingest)

    rank_p = sub.add_parser(
        "rank-player",
        help="Rank team destinations for a player (e.g. 2544 LeBron)",
    )
    rank_p.add_argument(
        "player_id",
        nargs="?",
        type=int,
        default=DEMO_PLAYER_ID,
        help=f"NBA player ID (default: {DEMO_PLAYER_ID})",
    )
    rank_p.add_argument("--season", default=get_settings().default_season)
    rank_p.add_argument("--top", type=int, default=10, help="Show top N teams")
    rank_p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic vectors (offline)",
    )
    rank_p.add_argument(
        "--fit-card-team",
        type=int,
        default=None,
        help="Also print JSON fit card vs this team_id",
    )
    rank_p.set_defaults(func=_cmd_rank_player)

    rank_t = sub.add_parser(
        "rank-team",
        help="Rank player targets for a team (e.g. 1610612747 Lakers)",
    )
    rank_t.add_argument(
        "team_id",
        nargs="?",
        type=int,
        default=DEMO_TEAM_ID,
        help=f"NBA team ID (default: {DEMO_TEAM_ID})",
    )
    rank_t.add_argument("--season", default=get_settings().default_season)
    rank_t.add_argument("--top", type=int, default=15, help="Show top N players")
    rank_t.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic vectors (offline)",
    )
    rank_t.set_defaults(func=_cmd_rank_team)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
