"""CLI entry point: `python -m nba_fit` or `nba-fit`."""

from __future__ import annotations

import argparse
import sys

from nba_fit.config.endpoints import ENDPOINT_REGISTRY, ESSENTIAL_ENDPOINTS
from nba_fit.config.settings import (
    INGEST_IMPACT_MAX_GAMES_DEV,
    INGEST_TIER_IMPACT,
    INGEST_TIER_MVP,
    INGEST_TIER_ROLE,
    INGEST_TIER_TACTICAL,
    get_settings,
)
from nba_fit.data.client import NBAClient
from nba_fit.data.ingest import run_ingest
from nba_fit.data.registry import ProbeRegistry
from nba_fit.features.season_context import (
    DEMO_PLAYER_ID,
    DEMO_TEAM_ID,
    SeasonFitContext,
)
from nba_fit.features.store import materialize_features
from nba_fit.models.impact_context import train_impact_for_season
from nba_fit.models.role_context import train_roles_for_season
from nba_fit.scoring.archetype_board import archetype_board_for_team
from nba_fit.scoring.fit_card import fit_card_to_json
from nba_fit.scoring.lineup_sim import run_lineup_sim
from nba_fit.data.fetchers.transactions import synthetic_movements
from nba_fit.evaluation.holdout_season import run_holdout_season_smoke
from nba_fit.evaluation.movement_backtest import run_movement_backtest
from nba_fit.models.weight_learning import learn_ensemble_weights
from nba_fit.scoring.constants import ENSEMBLE_COMPONENT_NAMES
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
    max_games = getattr(args, "max_games", None)
    full_season = getattr(args, "full_season", False)
    cap_msg = ""
    if args.tier == INGEST_TIER_IMPACT:
        if full_season or max_games == 0:
            cap_msg = " max_games=unlimited"
        elif max_games is not None:
            cap_msg = f" max_games={max_games}"
    print(
        f"Ingest tier={args.tier} season={season}{cap_msg} "
        f"(cache={'on' if not args.no_cache else 'off'})..."
    )
    try:
        result = run_ingest(
            season=season,
            tier=args.tier,
            use_cache=not args.no_cache,
            max_games=max_games,
            full_season=full_season,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Ingest failed: {exc}", file=sys.stderr)
        return 1

    if result.players_path:
        print(f"  players: {result.player_rows} rows -> {result.players_path}")
    if result.teams_path:
        print(f"  teams:   {result.team_rows} rows -> {result.teams_path}")
    if result.features_materialized:
        print(f"  player_features_scaled -> {result.player_features_path}")
        print(f"  team_features_scaled   -> {result.team_features_path}")
        print(f"  scaling_params         -> {result.scaling_params_path}")
    if result.lineup_units_path:
        print(f"  lineup_units: {result.lineup_units_rows} rows -> {result.lineup_units_path}")
    if result.onoff_path:
        print(f"  onoff:   {result.onoff_rows} rows -> {result.onoff_path}")
    if result.possessions_rows:
        print(
            f"  possessions: {result.possessions_rows} rows "
            f"({result.games_ingested} games, {len(result.possessions_paths)} files)"
        )
        if result.possessions_paths:
            print(f"    first: {result.possessions_paths[0]}")
    for endpoint, fetch in result.fetched.items():
        rows = sum(len(df) for df in fetch.frames.values())
        cache_flag = "cache" if fetch.from_cache else "live"
        print(f"    {endpoint}: {rows} rows ({cache_flag})")
    return 0


def _cmd_materialize_features(args: argparse.Namespace) -> int:
    season = args.season or get_settings().default_season
    print(f"Materializing feature store season={season}...")
    try:
        result = materialize_features(season)
    except FileNotFoundError as exc:
        print(f"materialize-features failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"materialize-features failed: {exc}", file=sys.stderr)
        return 1

    print(f"  player_features_scaled: {result.player_rows} rows -> {result.player_path}")
    print(f"  team_features_scaled:   {result.team_rows} rows -> {result.team_path}")
    print(f"  scaling_params:         {result.scaling_rows} rows -> {result.scaling_path}")
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


def _cmd_train_roles(args: argparse.Namespace) -> int:
    season = args.season or get_settings().default_season
    print(f"Training Option B role model season={season}...")
    try:
        role_ctx = train_roles_for_season(
            season,
            prefer_interim=not args.synthetic,
            prefer_api=not args.synthetic,
            synthetic=args.synthetic,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"train-roles failed: {exc}", file=sys.stderr)
        return 1
    from nba_fit.models.role_embeddings import role_embedding_dir

    out_dir = role_embedding_dir(season)
    n_labels = len(set(role_ctx.archetypes.archetype_labels))
    print(f"  players: {len(role_ctx.embeddings.player_ids)}")
    print(f"  archetypes: {n_labels} cluster labels")
    print(f"  embedding dim: {role_ctx.embeddings.n_components}")
    print(f"  saved: {out_dir}")
    return 0


def _cmd_train_impact(args: argparse.Namespace) -> int:
    season = args.season or get_settings().default_season
    print(f"Training Option C impact models season={season}...")
    try:
        impact_ctx = train_impact_for_season(
            season,
            prefer_interim=not args.synthetic,
            synthetic=args.synthetic,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"train-impact failed: {exc}", file=sys.stderr)
        return 1
    from nba_fit.models.lineup_model import lineup_model_dir
    from nba_fit.models.rapm import rapm_dir

    rapm_out = rapm_dir(impact_ctx.season)
    lineup_out = lineup_model_dir(impact_ctx.season)
    low_n = int(impact_ctx.rapm.low_sample_flag.sum())
    print(f"  rapm players: {len(impact_ctx.rapm.player_ids)} ({low_n} low-sample flags)")
    print(f"  rapm source: {impact_ctx.rapm_source}")
    if impact_ctx.degenerate_rapm:
        print(
            "  WARNING: degenerate RAPM (net_rapm std == 0) — "
            "impact submetrics may be uninformative",
            file=sys.stderr,
        )
    net_std = impact_ctx.metadata.get("net_rapm_std", float(impact_ctx.rapm.net_rapm.std()))
    print(f"  net rapm std: {net_std:.4f}")
    print(
        f"  net rapm range: {impact_ctx.rapm.net_rapm.min():.2f} "
        f".. {impact_ctx.rapm.net_rapm.max():.2f}"
    )
    print(f"  lineup model penalty: {impact_ctx.lineup_model.penalty}")
    print(f"  saved rapm: {rapm_out}")
    print(f"  saved lineup_model: {lineup_out}")
    return 0


def _cmd_fit_weights(args: argparse.Namespace) -> int:
    """Stub: learn ensemble weights from synthetic or movement backtest labels."""
    season = args.season or get_settings().default_season
    print(f"Learning ensemble weights season={season}...")
    try:
        context = SeasonFitContext.from_synthetic(season, n_players=30)
        movements = synthetic_movements(season, n_moves=15)
        result = run_movement_backtest(context, movements, calibrate=False)
        rows = result.rows
        if rows.empty:
            print("No movement rows to fit weights.", file=sys.stderr)
            return 1

        comp_cols = [f"ensemble_{name}" for name in ENSEMBLE_COMPONENT_NAMES]
        missing = [c for c in comp_cols if c not in rows.columns]
        if missing:
            print(f"Missing ensemble columns: {missing}", file=sys.stderr)
            return 1

        X = rows[comp_cols].to_numpy(dtype=float)
        y = rows["post_move_outcome"].to_numpy(dtype=float)
        l2 = args.l2
        fit = learn_ensemble_weights(X, y, l2=l2) if l2 is not None else learn_ensemble_weights(X, y)
    except Exception as exc:  # noqa: BLE001
        print(f"fit-weights failed: {exc}", file=sys.stderr)
        return 1

    print(f"  n_samples: {fit.n_samples}")
    print(f"  train_mse: {fit.train_mse:.4f}")
    print(f"  l2: {fit.l2}")
    for name, w in fit.weights.items():
        prior = fit.prior_weights.get(name, 0.0)
        print(f"  {name:22} learned={w:.3f}  prior={prior:.3f}")
    return 0


def _cmd_lineup_sim(args: argparse.Namespace) -> int:
    player_id = int(args.player_id)
    team_id = int(args.team_id)
    season = args.season or get_settings().default_season
    try:
        result = run_lineup_sim(
            player_id,
            team_id,
            season,
            top_n=args.top,
            prefer_interim=not args.synthetic,
            prefer_api=not args.synthetic,
            synthetic=args.synthetic,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"lineup-sim failed: {exc}", file=sys.stderr)
        return 1

    team_label = str(team_id)
    if not args.synthetic:
        try:
            ctx = SeasonFitContext.build(
                season, prefer_interim=True, prefer_api=False
            )
            team = ctx.teams.get(team_id)
            if team is not None:
                team_label = team.display_name
        except Exception:  # noqa: BLE001
            pass

    print(
        f"Lineup simulation — player {player_id} on {team_label} "
        f"({result.season}, source={result.data_source})"
    )
    if result.projected_net_rating_delta is not None:
        print(
            f"  projected_net_rating_delta: "
            f"{result.projected_net_rating_delta:+.2f} (pts/100 poss)"
        )
    table = result.to_table()
    if table.empty:
        print("  No projected lineup units.", file=sys.stderr)
        return 1
    print(table.to_string(index=False))
    if result.note:
        print(f"\n  {result.note}")
    return 0


def _cmd_archetype_board(args: argparse.Namespace) -> int:
    team_id = int(args.team_id)
    season = args.season or get_settings().default_season
    try:
        board = archetype_board_for_team(
            team_id,
            season,
            top_n=args.top,
            prefer_interim=not args.synthetic,
            prefer_api=not args.synthetic,
            synthetic=args.synthetic,
            force_retrain=args.retrain,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"archetype-board failed: {exc}", file=sys.stderr)
        return 1
    if board.empty:
        print(f"No archetype board for team_id={team_id} (season={season})", file=sys.stderr)
        return 1
    print(f"Archetype target board — team {team_id} ({season})")
    print(board.to_string(index=False))
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
        help="Fetch ingest-tier endpoints and write interim Parquet",
    )
    ingest.add_argument(
        "--season",
        default=get_settings().default_season,
        help="NBA season string (default from config)",
    )
    ingest.add_argument(
        "--tier",
        default=INGEST_TIER_MVP,
        choices=[INGEST_TIER_MVP, INGEST_TIER_ROLE, INGEST_TIER_IMPACT, INGEST_TIER_TACTICAL],
        help=(
            "Ingest bundle (mvp = Option A; role = Option B lineups/on-off; "
            "impact = Option C possessions; tactical = Option D hustle/defend/gravity)"
        ),
    )
    ingest.add_argument(
        "--full-season",
        action="store_true",
        help=(
            f"Fetch all regular-season games for --tier {INGEST_TIER_IMPACT} "
            "(no dev cap; overrides --max-games)"
        ),
    )
    ingest.add_argument(
        "--max-games",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Cap per-game PBP pulls for --tier {INGEST_TIER_IMPACT} "
            f"(default {INGEST_IMPACT_MAX_GAMES_DEV}; 0 = unlimited; "
            "see --full-season)"
        ),
    )
    ingest.add_argument("--no-cache", action="store_true", help="Bypass Parquet cache")
    ingest.set_defaults(func=_cmd_ingest)

    materialize = sub.add_parser(
        "materialize-features",
        help="Build scaled feature matrices from interim player/team tables",
    )
    materialize.add_argument(
        "--season",
        default=get_settings().default_season,
        help="NBA season string (default from config)",
    )
    materialize.set_defaults(func=_cmd_materialize_features)

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

    train_roles = sub.add_parser(
        "train-roles",
        help="Train Option B role embeddings, archetypes, team need, and NN comps",
    )
    train_roles.add_argument("--season", default=get_settings().default_season)
    train_roles.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic vectors (offline)",
    )
    train_roles.set_defaults(func=_cmd_train_roles)

    train_impact = sub.add_parser(
        "train-impact",
        help="Train Option C RAPM + lineup net-rating models",
    )
    train_impact.add_argument("--season", default=get_settings().default_season)
    train_impact.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic lineup stints (offline)",
    )
    train_impact.set_defaults(func=_cmd_train_impact)

    fit_weights = sub.add_parser(
        "fit-weights",
        help="Learn ensemble weights from labeled movement rows (stub)",
    )
    fit_weights.add_argument("--season", default=get_settings().default_season)
    fit_weights.add_argument(
        "--l2",
        type=float,
        default=None,
        help="L2 prior penalty (default from models.constants)",
    )
    fit_weights.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic movement backtest rows",
    )
    fit_weights.set_defaults(func=_cmd_fit_weights)

    archetype = sub.add_parser(
        "archetype-board",
        help="Rank players by team need × role fit for a team",
    )
    archetype.add_argument(
        "team_id",
        type=int,
        help="NBA team ID (e.g. 1610612747 Lakers)",
    )
    archetype.add_argument("--season", default=get_settings().default_season)
    archetype.add_argument("--top", type=int, default=15, help="Show top N players")
    archetype.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic vectors (offline)",
    )
    archetype.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain and persist role model even if artifacts exist",
    )
    archetype.set_defaults(func=_cmd_archetype_board)

    lineup_sim = sub.add_parser(
        "lineup-sim",
        help="Project top five-man units and net-rating deltas (Option C)",
    )
    lineup_sim.add_argument(
        "player_id",
        nargs="?",
        type=int,
        default=DEMO_PLAYER_ID,
        help=f"NBA player ID (default: {DEMO_PLAYER_ID})",
    )
    lineup_sim.add_argument(
        "team_id",
        nargs="?",
        type=int,
        default=DEMO_TEAM_ID,
        help=f"Destination team ID (default: {DEMO_TEAM_ID})",
    )
    lineup_sim.add_argument("--season", default=get_settings().default_season)
    lineup_sim.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of lineup units to show (default: 5)",
    )
    lineup_sim.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic lineups (offline)",
    )
    lineup_sim.set_defaults(func=_cmd_lineup_sim)

    backtest = sub.add_parser(
        "backtest-movement",
        help="Smoke movement backtest for a season (Option D)",
    )
    backtest.add_argument(
        "--season",
        default="2024-25",
        help="NBA season string (default: 2024-25)",
    )
    backtest.add_argument(
        "--synthetic",
        action="store_true",
        default=None,
        help="Use synthetic vectors (default: off when movement cache exists)",
    )
    backtest.add_argument(
        "--no-synthetic",
        action="store_false",
        dest="synthetic",
        help="Use interim/API season context when available",
    )
    backtest.add_argument(
        "--moves",
        type=int,
        default=20,
        help="Number of synthetic movements to simulate (default: 20)",
    )
    backtest.set_defaults(func=_cmd_backtest_movement)

    return parser


def _default_backtest_synthetic(season: str) -> bool:
    """Prefer real interim data when movement labels exist on disk."""
    if movements_cache_path(season).exists():
        return False
    manual = load_manual_movements()
    if not manual.empty:
        return False
    return True


def _cmd_backtest_movement(args: argparse.Namespace) -> int:
    season = args.season or get_settings().default_season
    use_synthetic = (
        args.synthetic
        if args.synthetic is not None
        else _default_backtest_synthetic(season)
    )
    print(f"Movement backtest — season={season} synthetic={use_synthetic}...")
    role_context: RoleFitContext | None = None
    impact_context: ImpactFitContext | None = None
    try:
        if use_synthetic:
            context = SeasonFitContext.from_synthetic(season, n_players=40)
            movements = synthetic_movements(season, n_moves=args.moves)
            role_context = RoleFitContext.from_synthetic(context)
            impact_context = ImpactFitContext.from_synthetic(role_context)
        else:
            context = SeasonFitContext.build(
                season, prefer_interim=True, prefer_api=False
            )
            movements = None
            try:
                role_context = RoleFitContext.from_season(
                    season,
                    prefer_interim=True,
                    prefer_api=False,
                    synthetic=False,
                    persist=False,
                )
            except (FileNotFoundError, ValueError):
                role_context = None
            try:
                impact_context = ImpactFitContext.from_season(
                    season,
                    prefer_interim=True,
                    synthetic=False,
                    persist=False,
                )
            except (FileNotFoundError, ValueError):
                impact_context = None
        result = run_movement_backtest(
            context,
            movements,
            role_context=role_context,
            impact_context=impact_context,
        )
        holdout = run_holdout_season_smoke(season, synthetic=use_synthetic)
    except Exception as exc:  # noqa: BLE001
        print(f"backtest-movement failed: {exc}", file=sys.stderr)
        return 1

    print(f"  movements scored: {result.n_movements}")
    if not result.rows.empty and "calibrated_fit_percentile" in result.rows.columns:
        print(
            f"  mean calibrated fit percentile: "
            f"{result.rows['calibrated_fit_percentile'].mean():.1f}"
        )
    if not result.rows.empty and "post_move_outcome" in result.rows.columns:
        print(f"  mean post-move outcome: {result.rows['post_move_outcome'].mean():.3f}")
    print(f"  holdout train={holdout.train_season} predict={holdout.predict_season}")
    print(
        f"  holdout pairs: train={holdout.train_pairs} predict={holdout.predict_pairs} "
        f"overlap_players={holdout.overlap_players}"
    )
    if holdout.mean_predict_percentile is not None:
        print(
            f"  holdout mean predict percentile: "
            f"{holdout.mean_predict_percentile:.1f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
