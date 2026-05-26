#!/usr/bin/env python3
"""Run Option D full-pipeline smoke on 2024-25 and write reports/validation/option_d/."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "reports" / "validation" / "option_d"
_FIGURES_OUT = _OUT / "figures"

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

SEASON = "2024-25"
PLAYER_ID = 2544
TEAM_ID = 1610612747
IMPACT_MAX_GAMES = 30
BACKTEST_MOVES = 15
MIN_REAL_MOVEMENTS = 5
SCORING_PLAYER_CAP = 80

if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from nba_fit.config.settings import (  # noqa: E402
    INGEST_TIER_IMPACT,
    INGEST_TIER_MVP,
    INGEST_TIER_ROLE,
)
from nba_fit.data.fetchers.transactions import (  # noqa: E402
    get_movements,
    synthetic_movements,
)
from nba_fit.data.ingest import run_ingest  # noqa: E402
from nba_fit.evaluation.holdout_season import run_holdout_season_smoke  # noqa: E402
from nba_fit.evaluation.movement_backtest import run_movement_backtest  # noqa: E402
from nba_fit.features.season_context import SeasonFitContext  # noqa: E402
from nba_fit.models.impact_context import ImpactFitContext, train_impact_for_season  # noqa: E402
from nba_fit.models.role_context import RoleFitContext, train_roles_for_season  # noqa: E402
from nba_fit.scoring.constants import (  # noqa: E402
    ENSEMBLE_COMPONENT_WEIGHTS,
    ENSEMBLE_DERIVED_NAMES,
    PROFILE_SUBMETRIC_WEIGHTS,
    SUBMETRIC_NAMES,
)
from nba_fit.scoring.fit_card import fit_card_to_json  # noqa: E402
from nba_fit.scoring.ensemble import (  # noqa: E402
    calibrated_ensemble,
    component_contributions,
    extract_ensemble_components,
    uncertainty_band,
)
from nba_fit.scoring.fit_index import (  # noqa: E402
    FitIndexTable,
    build_fit_index_table,
    raw_scores_to_percentiles,
)
from nba_fit.scoring.ranker import FitRanker  # noqa: E402
from nba_fit.scoring.submetrics import compute_all_submetrics  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_cli(argv: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "nba_fit", *argv],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _run_pytest() -> tuple[int, str, list[str]]:
    targets = [
        "tests/test_ensemble.py",
        "tests/test_fit_index.py",
        "tests/test_movement_backtest.py",
        "tests/test_ingest_smoke.py",
        "tests/test_config.py",
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *targets, "-q"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out, targets


def _models_ready(season: str) -> bool:
    rapm = _REPO / "models" / "rapm" / f"season={season}" / "player_rapm.parquet"
    role = (
        _REPO
        / "models"
        / "role_embedding"
        / f"season={season}"
        / "player_embeddings.parquet"
    )
    return rapm.is_file() and role.is_file()


def _interim_ready(season: str) -> bool:
    from nba_fit.config.settings import (
        INTERIM_TABLE_LINEUP_UNITS,
        INTERIM_TABLE_ONOFF,
        INTERIM_TABLE_PLAYERS,
        get_settings,
    )

    settings = get_settings()
    checks = (
        settings.interim_path(INTERIM_TABLE_PLAYERS, season),
        settings.interim_path(INTERIM_TABLE_LINEUP_UNITS, season),
        settings.interim_path(INTERIM_TABLE_ONOFF, season),
    )
    return all(p.exists() and any(p.glob("*.parquet")) for p in checks)


def _possessions_ready(season: str) -> bool:
    from nba_fit.config.settings import INTERIM_TABLE_POSSESSIONS, get_settings

    part = get_settings().interim_path(INTERIM_TABLE_POSSESSIONS, season)
    return part.exists() and any(part.rglob("*.parquet"))


def _ingest_all(season: str) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    tiers: list[str] = []
    if _interim_ready(season):
        print("[ingest] mvp/role interim cached; skipping", flush=True)
        summaries["mvp"] = {"skipped": True, "reason": "interim_cached"}
        summaries["role"] = {"skipped": True, "reason": "interim_cached"}
    else:
        tiers.extend([INGEST_TIER_MVP, INGEST_TIER_ROLE])

    if _possessions_ready(season):
        summaries["impact"] = {"skipped": True, "reason": "possessions_cached"}
    else:
        tiers.append(INGEST_TIER_IMPACT)

    if not tiers:
        print("[ingest] all tiers cached; skipping", flush=True)
        return summaries

    for tier in tiers:
        print(f"[ingest] tier={tier} season={season}...", flush=True)
        kwargs: dict[str, Any] = {"season": season, "tier": tier, "use_cache": True}
        if tier == INGEST_TIER_IMPACT:
            kwargs["max_games"] = IMPACT_MAX_GAMES
        result = run_ingest(**kwargs)
        print(f"[ingest] tier={tier} done", flush=True)
        summaries[tier] = {
            "player_rows": result.player_rows,
            "team_rows": result.team_rows,
            "lineup_units_rows": result.lineup_units_rows,
            "onoff_rows": result.onoff_rows,
            "possessions_rows": result.possessions_rows,
            "games_ingested": result.games_ingested,
        }
    return summaries


def _uncertainty_summary(backtest_rows: pd.DataFrame) -> dict[str, float | None]:
    if backtest_rows.empty:
        return {
            "mean_ci_low": None,
            "mean_ci_high": None,
            "mean_disagreement": None,
            "mean_sample_penalty": None,
        }
    out: dict[str, float | None] = {}
    for col, key in (
        ("uncertainty_ci_low", "mean_ci_low"),
        ("uncertainty_ci_high", "mean_ci_high"),
        ("uncertainty_disagreement", "mean_disagreement"),
        ("uncertainty_sample_penalty", "mean_sample_penalty"),
    ):
        if col in backtest_rows.columns:
            out[key] = round(float(backtest_rows[col].mean()), 4)
        else:
            out[key] = None
    return out


def _run_visual_tests() -> list[str]:
    import visual_tests._constants as vt_constants

    vt_constants.FIGURES_DIR = _FIGURES_OUT
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for mod_name in (
        "visual_tests.12_calibration_curve",
        "visual_tests.13_ensemble_weights",
        "visual_tests.14_dashboard_data_health",
    ):
        mod = importlib.import_module(mod_name)
        rc = int(mod.main())
        if rc != 0:
            raise RuntimeError(f"{mod_name} exited {rc}")

    for nested in (_FIGURES_OUT / "option_d", _FIGURES_OUT / "dashboard"):
        if nested.is_dir():
            for png in nested.glob("*.png"):
                dest = _FIGURES_OUT / png.name
                shutil.copy2(png, dest)
                written.append(str(dest.relative_to(_REPO)))

    for png in sorted(_FIGURES_OUT.glob("*.png")):
        rel = str(png.relative_to(_REPO))
        if rel not in written:
            written.append(rel)
    return written


def _scoring_player_ids(context: SeasonFitContext, movements: pd.DataFrame) -> list[int]:
    """Players to score for backtest (movement labels + demo player, capped)."""
    ids: set[int] = {PLAYER_ID}
    if not movements.empty and "player_id" in movements.columns:
        ids.update(int(x) for x in movements["player_id"].dropna().unique())
    for pid in context.players:
        if len(ids) >= SCORING_PLAYER_CAP:
            break
        ids.add(int(pid))
    return sorted(ids)[:SCORING_PLAYER_CAP]


def _subset_context(
    context: SeasonFitContext, player_ids: list[int]
) -> SeasonFitContext:
    pid_set = set(player_ids)
    players = {k: v for k, v in context.players.items() if k in pid_set}
    return SeasonFitContext(
        season=context.season,
        players=players,
        teams=context.teams,
        source=context.source,
    )


def _build_player_destinations_table(
    context: SeasonFitContext,
    player_id: int,
    *,
    role_context: RoleFitContext | None,
    impact_context: ImpactFitContext | None,
) -> FitIndexTable:
    if player_id not in context.players:
        raise ValueError(f"player_id {player_id} not in season context")
    mini = SeasonFitContext(
        season=context.season,
        players={player_id: context.players[player_id]},
        teams=context.teams,
        source=context.source,
    )
    return build_fit_index_table(
        mini, role_context=role_context, impact_context=impact_context
    )


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    log: list[str] = [
        "# Option D validation run log",
        "",
        f"- **Started (UTC):** {_utc_now()}",
        f"- **Season:** {SEASON}",
        f"- **Player / team:** {PLAYER_ID} / {TEAM_ID}",
        "",
        "## Pipeline",
        "",
        f"1. `ingest --tier mvp|role|impact --season {SEASON}` (impact `--max-games {IMPACT_MAX_GAMES}`)",
        f"2. `train-roles --season {SEASON}`",
        f"3. `train-impact --season {SEASON}`",
        f"4. `backtest-movement --season {SEASON} --no-synthetic --moves {BACKTEST_MOVES}`",
        f"5. `rank-player {PLAYER_ID} --fit-card-team {TEAM_ID} --season {SEASON}`",
        "6. `pytest` (Option D subset — see metrics.json)",
        "7. `visual_tests/12_calibration_curve.py`, `13_ensemble_weights.py`, `14_dashboard_data_health.py`",
        "",
    ]

    ingest_ok = False
    ingest_summary: dict[str, Any] = {}
    ingest_note = ""
    try:
        ingest_summary = _ingest_all(SEASON)
        ingest_ok = True
        skipped = all(
            isinstance(v, dict) and v.get("skipped") for v in ingest_summary.values()
        ) if ingest_summary else False
        if skipped:
            log.extend(
                [
                    "## Ingest",
                    "",
                    "- **Status:** skipped (interim / possessions cached)",
                    "",
                ]
            )
        else:
            log.extend(["## Ingest", "", "- **Status:** pass", ""])
    except Exception as exc:  # noqa: BLE001
        ingest_note = str(exc)
        log.extend(["## Ingest", "", f"- **Status:** fail — {exc}", ""])

    train_roles_ok = False
    train_impact_ok = False
    role_ctx: RoleFitContext | None = None
    impact_ctx: ImpactFitContext | None = None

    log.extend(["## Train", ""])
    try:
        if _models_ready(SEASON):
            print("[train] loading persisted role/impact models...", flush=True)
            role_ctx = RoleFitContext.from_season(
                SEASON, prefer_interim=True, prefer_api=False, synthetic=False
            )
            impact_ctx = ImpactFitContext.from_season(
                SEASON, prefer_interim=True, synthetic=False
            )
            train_roles_ok = True
            train_impact_ok = True
            log.append(
                "- **train-roles:** skipped (artifacts on disk; loaded via `from_season`)"
            )
            log.append(
                f"- **train-impact:** skipped (loaded; {len(impact_ctx.rapm.player_ids)} rapm players)"
            )
        else:
            print("[train] train-roles...", flush=True)
            role_ctx = train_roles_for_season(
                SEASON, prefer_interim=True, prefer_api=False, synthetic=False
            )
            train_roles_ok = True
            log.append(
                f"- **train-roles:** pass ({len(role_ctx.embeddings.player_ids)} players)"
            )
            print("[train] train-impact...", flush=True)
            impact_ctx = train_impact_for_season(
                SEASON, prefer_interim=True, synthetic=False
            )
            train_impact_ok = True
            log.append(
                f"- **train-impact:** pass ({len(impact_ctx.rapm.player_ids)} rapm players)"
            )
    except Exception as exc:  # noqa: BLE001
        log.append(f"- **train:** fail — {exc}")

    log.extend(["", "## Backtest", ""])
    movements_synthetic = False
    movements_source = "interim/manual/cache"
    backtest_result = None
    holdout = None

    try:
        context = SeasonFitContext.build(
            SEASON, prefer_interim=True, prefer_api=False
        )
        if role_ctx is None:
            role_ctx = RoleFitContext.from_season(
                SEASON, prefer_interim=True, prefer_api=False, synthetic=False
            )
        if impact_ctx is None:
            impact_ctx = ImpactFitContext.from_season(
                SEASON, prefer_interim=True, synthetic=False
            )

        movements = get_movements(SEASON, use_synthetic_fallback=False)
        n_real = len(movements)
        if n_real < MIN_REAL_MOVEMENTS:
            movements = synthetic_movements(SEASON, n_moves=BACKTEST_MOVES)
            movements_synthetic = True
            movements_source = "synthetic (sparse real movements)"
            log.append(
                f"- **Movements:** sparse real labels ({n_real} "
                f"< {MIN_REAL_MOVEMENTS}); used `synthetic_movements(n={BACKTEST_MOVES})`"
            )
        else:
            movements = movements.head(BACKTEST_MOVES)
            log.append(f"- **Movements:** {len(movements)} from {movements_source}")

        scoring_ids = _scoring_player_ids(context, movements)
        scoring_ctx = _subset_context(context, scoring_ids)
        log.append(
            f"- **Scoring subset:** {len(scoring_ctx.players)} players "
            f"(cap={SCORING_PLAYER_CAP}, validation runtime)"
        )
        backtest_result = run_movement_backtest(
            scoring_ctx,
            movements,
            role_context=role_ctx,
            impact_context=impact_ctx,
        )
        holdout = run_holdout_season_smoke(SEASON, synthetic=not ingest_ok)
        backtest_result.rows.to_csv(_OUT / "backtest_results.csv", index=False)
        log.append(f"- **Status:** pass ({backtest_result.n_movements} scored)")
    except Exception as exc:  # noqa: BLE001
        log.append(f"- **Status:** fail — {exc}")

    log.extend(["", "## Rank / fit card", ""])
    overall_percentile: float | None = None
    ranker_source = ""

    try:
        context = SeasonFitContext.build(
            SEASON, prefer_interim=True, prefer_api=False
        )
        if role_ctx is None:
            role_ctx = RoleFitContext.from_season(
                SEASON, prefer_interim=True, prefer_api=False, synthetic=False
            )
        if impact_ctx is None:
            impact_ctx = ImpactFitContext.from_season(
                SEASON, prefer_interim=True, synthetic=False
            )
        print("[rank] building destination table for demo player...", flush=True)
        table = _build_player_destinations_table(
            context,
            PLAYER_ID,
            role_context=role_ctx,
            impact_context=impact_ctx,
        )
        ranker = FitRanker(context=context, table=table)
        ranker_source = ranker.context.source
        rankings = ranker.rank_destinations_for_player(PLAYER_ID, top_n=10)
        fit_card = ranker.fit_card(PLAYER_ID, TEAM_ID, role_context=role_ctx)
        overall_percentile = fit_card.get("overall_fit_percentile")
        (_OUT / f"fit_card_{PLAYER_ID}_{TEAM_ID}.json").write_text(
            fit_card_to_json(fit_card), encoding="utf-8"
        )
        rankings.to_csv(_OUT / f"rankings_player_{PLAYER_ID}.csv", index=False)
        log.append(f"- **Context:** {ranker_source}")
        log.append(f"- **overall_fit_percentile:** {overall_percentile}")
    except Exception as exc:  # noqa: BLE001
        log.append(f"- **Status:** fail — {exc}")
        cli_rc, cli_out = _run_cli(
            [
                "rank-player",
                str(PLAYER_ID),
                "--season",
                SEASON,
                "--fit-card-team",
                str(TEAM_ID),
                "--synthetic",
            ]
        )
        log.append(f"- **CLI synthetic fallback** (rc={cli_rc})")
        if cli_out.strip():
            log.append(f"```\n{cli_out.strip()}\n```")

    pytest_rc, pytest_out, pytest_targets = _run_pytest()
    log.extend(
        [
            "",
            "## Pytest",
            "",
            f"- **Subset:** {', '.join(pytest_targets)}",
            f"```\n{pytest_out.strip() or '(no output)'}\n```",
            f"- **Exit code:** {pytest_rc}",
            "",
            "## Visual tests",
            "",
        ]
    )

    figure_paths: list[str] = []
    try:
        figure_paths = _run_visual_tests()
        for p in figure_paths:
            log.append(f"- `{p}`")
    except Exception as exc:  # noqa: BLE001
        log.append(f"- **Status:** fail — {exc}")

    calibration: dict[str, Any] = {"method": "isotonic", "fitted": False}
    backtest_metrics: dict[str, Any] = {}
    if backtest_result is not None:
        cal = backtest_result.calibrator
        if cal is not None and cal.is_fitted_:
            calibration = {
                "method": cal.method,
                "fitted": True,
                "n_anchor": int(len(cal._raw_anchor or [])),
            }
        rows = backtest_result.rows
        backtest_metrics = {
            "n_movements": backtest_result.n_movements,
            "movements_synthetic": movements_synthetic,
            "movements_source": movements_source,
            "mean_calibrated_fit_percentile": backtest_result.mean_destination_percentile,
            "mean_post_move_outcome": (
                float(rows["post_move_outcome"].mean())
                if "post_move_outcome" in rows.columns
                else None
            ),
            "mean_pre_move_fit_percentile": (
                float(rows["pre_move_fit_percentile"].mean())
                if "pre_move_fit_percentile" in rows.columns
                and rows["pre_move_fit_percentile"].notna().any()
                else None
            ),
        }
        backtest_metrics["uncertainty"] = _uncertainty_summary(rows)

    metrics: dict[str, Any] = {
        "generated_at_utc": _utc_now(),
        "season": SEASON,
        "ingest_ok": ingest_ok,
        "ingest": ingest_summary,
        "ingest_note": ingest_note,
        "train_roles_ok": train_roles_ok,
        "train_impact_ok": train_impact_ok,
        "ensemble_component_weights": ENSEMBLE_COMPONENT_WEIGHTS,
        "profile_submetric_weights": PROFILE_SUBMETRIC_WEIGHTS,
        "calibration": calibration,
        "backtest": backtest_metrics,
        "holdout": holdout.to_dict() if holdout is not None else {},
        "player_id": PLAYER_ID,
        "team_id": TEAM_ID,
        "ranker_data_source": ranker_source,
        "overall_fit_percentile": overall_percentile,
        "pytest_exit_code": pytest_rc,
        "pytest_targets": pytest_targets,
        "pytest_summary": pytest_out.strip().splitlines()[-1] if pytest_out.strip() else "",
        "visual_test_figures": figure_paths,
    }
    (_OUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    log.append("")
    log.append(f"- **Finished (UTC):** {_utc_now()}")
    (_OUT / "RUN_LOG.md").write_text("\n".join(log), encoding="utf-8")

    print(json.dumps(metrics, indent=2, default=str))
    ok = (
        pytest_rc == 0
        and overall_percentile is not None
        and (backtest_result is None or backtest_result.n_movements > 0)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
