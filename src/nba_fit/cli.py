"""CLI entry point: `python -m nba_fit` or `nba-fit`."""

from __future__ import annotations

import argparse
import sys

from nba_fit.config.endpoints import ENDPOINT_REGISTRY, ESSENTIAL_ENDPOINTS
from nba_fit.config.settings import get_settings
from nba_fit.data.client import NBAClient
from nba_fit.data.registry import ProbeRegistry


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
