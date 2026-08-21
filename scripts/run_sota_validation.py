#!/usr/bin/env python3
"""SOTA validation bundle: real-data gates, metrics, and report artifacts.

Writes ``reports/validation/sota/{metrics.json,RUN_LOG.md,figures/}``.
Exits 1 when any P0 gate fails (synthetic movements, degenerate RAPM, sparse moves,
missing interim partitions).
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "reports" / "validation" / "sota"
_FIGURES_OUT = _OUT / "figures"

if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from nba_fit.config.settings import (  # noqa: E402
    ENV_NBA_FIT_DATA_ROOT,
    INTERIM_TABLE_LINEUP_UNITS,
    INTERIM_TABLE_ONOFF,
    INTERIM_TABLE_PLAYERS,
    INTERIM_TABLE_POSSESSIONS,
    INTERIM_TABLE_TEAMS,
    get_settings,
)
from nba_fit.data.fetchers.transactions import (  # noqa: E402
    get_movements,
    load_gamelogs_for_season,
    warm_movements_cache,
)
from nba_fit.features.constants import COL_PLAYER_ID  # noqa: E402
from nba_fit.evaluation.ablation import run_ablation, write_ablation_metrics  # noqa: E402
from nba_fit.evaluation.calibration_metrics import (  # noqa: E402
    evaluate_holdout_calibration,
    fit_isotonic_on_train_rows,
    split_movements_by_season,
)
from nba_fit.evaluation.holdout_season import run_holdout_season  # noqa: E402
from nba_fit.evaluation.movement_backtest import run_movement_backtest  # noqa: E402
from nba_fit.evaluation.rapm_benchmark import benchmark_rapm_vs_estimated_net  # noqa: E402
from nba_fit.evaluation.replacement_benchmark import rank_signing_vs_position_pool  # noqa: E402
from nba_fit.evaluation.sota_validation import (  # noqa: E402
    SotaValidationError,
    assert_non_degenerate_rapm,
)
from nba_fit.features.season_context import SeasonFitContext  # noqa: E402
from nba_fit.models.constants import (  # noqa: E402
    MIN_REAL_MOVEMENTS,
    SOTA_MOVEMENT_SEASONS,
)
from nba_fit.models.impact_context import ImpactFitContext, train_impact_for_season  # noqa: E402
from nba_fit.models.role_context import RoleFitContext, train_roles_for_season  # noqa: E402
from nba_fit.models.weight_learning import learn_ensemble_weights, save_ensemble_weights  # noqa: E402
from nba_fit.scoring.constants import (  # noqa: E402
    ENSEMBLE_COMPONENT_NAMES,
    ENSEMBLE_COMPONENT_WEIGHTS,
    PROFILE_SUBMETRIC_WEIGHTS,
)
from nba_fit.scoring.ensemble import resolve_ensemble_component_weights  # noqa: E402
from nba_fit.scoring.fit_index import build_fit_index_table  # noqa: E402

SEASON = os.environ.get("NBA_FIT_SOTA_SEASON", "2024-25")
SCORING_PLAYER_CAP = 120
DEMO_PLAYER_ID = 2544
DEMO_TEAM_ID = 1610612747

REQUIRED_INTERIM_TABLES: tuple[str, ...] = (
    INTERIM_TABLE_PLAYERS,
    INTERIM_TABLE_TEAMS,
    INTERIM_TABLE_LINEUP_UNITS,
    INTERIM_TABLE_ONOFF,
    INTERIM_TABLE_POSSESSIONS,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_line(log: list[str], msg: str) -> None:
    line = f"[{_utc_now()}] {msg}"
    print(line, flush=True)
    log.append(line)


def _allow_api() -> bool:
    return "--allow-api" in sys.argv


def _partition_status(season: str) -> dict[str, dict[str, Any]]:
    settings = get_settings()
    out: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_INTERIM_TABLES:
        part = settings.interim_path(name, season)
        files = list(part.rglob("*.parquet")) if part.exists() else []
        out[name] = {
            "path": str(part),
            "exists": part.exists(),
            "n_parquet": len(files),
            "ready": bool(files),
        }
    return out


def _data_root_info() -> dict[str, Any]:
    settings = get_settings()
    env_raw = os.environ.get(ENV_NBA_FIT_DATA_ROOT)
    return {
        "env_var": ENV_NBA_FIT_DATA_ROOT,
        "env_set": bool(env_raw and str(env_raw).strip()),
        "env_value": str(env_raw).strip() if env_raw else None,
        "resolved_data_root": str(settings.data_root) if settings.data_root else None,
        "data_interim": str(settings.data_interim),
        "models_dir": str(settings.models_dir),
        "using_external_root": settings.data_root is not None,
    }


def _models_ready(season: str) -> bool:
    settings = get_settings()
    rapm = settings.models_dir / "rapm" / f"season={season}" / "player_rapm.parquet"
    role = (
        settings.models_dir
        / "role_embedding"
        / f"season={season}"
        / "player_embeddings.parquet"
    )
    return rapm.is_file() and role.is_file()


def _net_rapm_std(impact_ctx: ImpactFitContext) -> float:
    meta = impact_ctx.metadata.get("net_rapm_std")
    if meta is not None:
        return float(meta)
    series = impact_ctx.rapm.net_rapm
    if series is None or len(series) == 0:
        return 0.0
    return float(np.std(series.to_numpy(dtype=float)))


def _scoring_player_ids(context: SeasonFitContext, movements: pd.DataFrame) -> list[int]:
    ids: set[int] = set()
    if not movements.empty and "player_id" in movements.columns:
        ids.update(int(x) for x in movements["player_id"].dropna().unique())
    for pid in context.players:
        if len(ids) >= SCORING_PLAYER_CAP:
            break
        ids.add(int(pid))
    return sorted(ids)[:SCORING_PLAYER_CAP]


def _subset_context(context: SeasonFitContext, player_ids: list[int]) -> SeasonFitContext:
    pid_set = set(player_ids)
    players = {k: v for k, v in context.players.items() if k in pid_set}
    players_raw = context._players_raw  # noqa: SLF001
    if players_raw is not None and COL_PLAYER_ID in players_raw.columns:
        players_raw = players_raw.loc[players_raw[COL_PLAYER_ID].isin(pid_set)].copy()
    elif players_raw is not None and "player_id" in players_raw.columns:
        players_raw = players_raw.loc[players_raw["player_id"].isin(pid_set)].copy()
    return SeasonFitContext(
        season=context.season,
        players=players,
        teams=context.teams,
        source=context.source,
        gamelogs=context.gamelogs,
        _players_raw=players_raw,
        _teams_raw=context._teams_raw,  # noqa: SLF001
    )


def _resolve_movements(*, allow_api: bool) -> tuple[pd.DataFrame, bool, str]:
    warm_movements_cache(SOTA_MOVEMENT_SEASONS, allow_api=allow_api)
    movements = get_movements(
        SEASON,
        allow_api=allow_api,
        use_synthetic_fallback=False,
    )
    n_real = len(movements)
    if n_real < MIN_REAL_MOVEMENTS:
        return (
            movements,
            True,
            f"insufficient real movements ({n_real} < {MIN_REAL_MOVEMENTS})",
        )
    return movements, False, "cache/manual/gamelog-inferred"


def _run_pytest_subset() -> tuple[int, str, list[str]]:
    targets = [
        "tests/test_ensemble.py",
        "tests/test_fit_index.py",
        "tests/test_movement_backtest.py",
        "tests/test_ablation.py",
        "tests/test_role_taxonomy.py",
        "tests/test_config.py",
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *targets, "-q", "-m", "not network"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out, targets


def _run_visual_tests(backtest_csv: Path) -> tuple[list[str], str | None]:
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    import visual_tests._constants as vt_constants
    import visual_tests._plot_utils as plot_utils
    import visual_tests._validation_artifacts as validation_artifacts

    vt_constants.FIGURES_DIR = _FIGURES_OUT
    plot_utils.FIGURES_DIR = _FIGURES_OUT
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    validation_artifacts.backtest_csv = backtest_csv
    validation_artifacts.fit_card_json = _OUT / "fit_card_placeholder.json"
    if not validation_artifacts.fit_card_json.is_file():
        validation_artifacts.fit_card_json.write_text("{}", encoding="utf-8")
    validation_artifacts.last_figure_caption = None

    written: list[str] = []
    data_health_caption: str | None = None
    for mod_name in (
        "visual_tests.12_calibration_curve",
        "visual_tests.13_ensemble_weights",
        "visual_tests.14_dashboard_data_health",
    ):
        mod = importlib.import_module(mod_name)
        rc = int(mod.main())
        if rc != 0:
            raise RuntimeError(f"{mod_name} exited {rc}")
        if mod_name.endswith("14_dashboard_data_health"):
            data_health_caption = validation_artifacts.last_figure_caption

    for nested in (_FIGURES_OUT / "option_d", _FIGURES_OUT / "dashboard"):
        if nested.is_dir():
            for png in nested.glob("*.png"):
                dest = _FIGURES_OUT / png.name
                shutil.copy2(png, dest)
                rel = str(dest.relative_to(_REPO))
                if rel not in written:
                    written.append(rel)

    for png in sorted(_FIGURES_OUT.glob("*.png")):
        rel = str(png.relative_to(_REPO))
        if rel not in written:
            written.append(rel)
    return written, data_health_caption


def _p0_gates(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    partitions = metrics.get("partitions", {})
    if not partitions.get("all_ready"):
        missing = partitions.get("missing_tables", [])
        failures.append(
            {
                "gate": "missing_required_partitions",
                "detail": f"missing or empty: {missing}",
            }
        )

    backtest = metrics.get("backtest", {})
    if backtest.get("movements_synthetic"):
        failures.append(
            {
                "gate": "movements_synthetic",
                "detail": backtest.get("movements_source", "synthetic"),
            }
        )
    n_moves = int(backtest.get("n_movements") or 0)
    if n_moves < MIN_REAL_MOVEMENTS:
        failures.append(
            {
                "gate": "n_movements_below_threshold",
                "detail": f"n_movements={n_moves} < {MIN_REAL_MOVEMENTS}",
            }
        )

    impact = metrics.get("impact", {})
    net_std = impact.get("net_rapm_std")
    if impact.get("degenerate_rapm") or (
        net_std is not None and float(net_std) == 0.0
    ):
        failures.append(
            {
                "gate": "net_rapm_degenerate",
                "detail": "net_rapm std == 0",
            }
        )
    option_c = metrics.get("option_c", {})
    if option_c.get("degenerate_rapm_gate_failed"):
        failures.append(
            {
                "gate": "option_c_degenerate_rapm",
                "detail": option_c.get("degenerate_rapm_error", "net_rapm std == 0"),
            }
        )
    return failures


def main() -> int:
    allow_api = _allow_api()
    _OUT.mkdir(parents=True, exist_ok=True)
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    log: list[str] = [
        "# SOTA validation run log",
        "",
        f"- **Started (UTC):** {_utc_now()}",
        f"- **Season:** {SEASON}",
        f"- **allow_api:** {allow_api}",
        "",
        "## Commands (reference)",
        "",
        "```bash",
        f"export {ENV_NBA_FIT_DATA_ROOT}=/path/to/nba-fit-data",
        "python scripts/run_sota_validation.py",
        "python scripts/run_sota_validation.py --allow-api",
        f"python -m nba_fit train-roles --season {SEASON}",
        f"python -m nba_fit train-impact --season {SEASON}",
        f"python -m nba_fit backtest-movement --season {SEASON} --no-synthetic",
        f"python -m nba_fit fit-weights --season {SEASON}",
        "```",
        "",
    ]

    data_root_info = _data_root_info()
    _log_line(log, f"data_root: {data_root_info}")
    log.extend(
        [
            "## Data root",
            "",
            f"- **{ENV_NBA_FIT_DATA_ROOT} set:** {data_root_info['env_set']}",
            f"- **External root:** {data_root_info['resolved_data_root']}",
            f"- **Interim path:** `{data_root_info['data_interim']}`",
            "",
        ]
    )

    partition_status = _partition_status(SEASON)
    all_ready = all(v["ready"] for v in partition_status.values())
    missing_tables = [k for k, v in partition_status.items() if not v["ready"]]
    log.extend(
        [
            "## Interim partitions",
            "",
            f"- **All required ready:** {all_ready}",
        ]
    )
    for name, st in partition_status.items():
        log.append(f"- `{name}`: ready={st['ready']} n_parquet={st['n_parquet']}")
    log.append("")

    metrics: dict[str, Any] = {
        "generated_at_utc": _utc_now(),
        "season": SEASON,
        "allow_api": allow_api,
        "data_root": data_root_info,
        "partitions": {
            "tables": partition_status,
            "all_ready": all_ready,
            "missing_tables": missing_tables,
        },
        "ensemble_component_weights_prior": ENSEMBLE_COMPONENT_WEIGHTS,
        "profile_submetric_weights": PROFILE_SUBMETRIC_WEIGHTS,
        "learned_ensemble_weights": None,
        "rapm_benchmark_correlation": None,
        "min_real_movements_required": MIN_REAL_MOVEMENTS,
        "sota_movement_seasons": list(SOTA_MOVEMENT_SEASONS),
        "p0_gates_passed": False,
        "p0_gate_failures": [],
    }

    train_roles_ok = False
    train_impact_ok = False
    role_ctx: RoleFitContext | None = None
    impact_ctx: ImpactFitContext | None = None

    log.extend(["## Train", ""])
    try:
        if _models_ready(SEASON):
            _log_line(log, "loading persisted role/impact models")
            role_ctx = RoleFitContext.from_season(
                SEASON, prefer_interim=True, prefer_api=False, synthetic=False
            )
            impact_ctx = ImpactFitContext.from_season(
                SEASON, prefer_interim=True, synthetic=False
            )
            train_roles_ok = True
            train_impact_ok = True
            log.append("- **train-roles:** skipped (artifacts on disk)")
            log.append("- **train-impact:** skipped (artifacts on disk)")
        else:
            if not all_ready:
                raise RuntimeError(
                    f"cannot train without interim partitions: {missing_tables}"
                )
            _log_line(log, "train-roles")
            role_ctx = train_roles_for_season(
                SEASON, prefer_interim=True, prefer_api=False, synthetic=False
            )
            train_roles_ok = True
            log.append(
                f"- **train-roles:** pass ({len(role_ctx.embeddings.player_ids)} players)"
            )
            _log_line(log, "train-impact")
            impact_ctx = train_impact_for_season(
                SEASON, prefer_interim=True, synthetic=False
            )
            train_impact_ok = True
            log.append(
                f"- **train-impact:** pass ({len(impact_ctx.rapm.player_ids)} rapm players)"
            )
    except Exception as exc:  # noqa: BLE001
        log.append(f"- **train:** fail — {exc}")

    net_rapm_std: float | None = None
    option_c_metrics: dict[str, Any] = {"season": SEASON}
    if impact_ctx is not None:
        net_rapm_std = _net_rapm_std(impact_ctx)
        degenerate = bool(impact_ctx.degenerate_rapm or net_rapm_std == 0.0)
        metrics["impact"] = {
            "n_rapm_players": len(impact_ctx.rapm.player_ids),
            "net_rapm_std": net_rapm_std,
            "net_rapm_min": float(impact_ctx.rapm.net_rapm.min()),
            "net_rapm_max": float(impact_ctx.rapm.net_rapm.max()),
            "degenerate_rapm": degenerate,
            "rapm_source": impact_ctx.rapm_source,
            "ridge_alpha": impact_ctx.rapm.ridge_alpha,
            "recency_half_life_games": impact_ctx.rapm.recency_half_life_games,
            "low_sample_fraction": float(
                impact_ctx.metadata.get("low_sample_fraction", 0.0)
            ),
        }
        try:
            assert_non_degenerate_rapm(impact_ctx.rapm)
            option_c_metrics["degenerate_rapm_gate_failed"] = False
        except SotaValidationError as exc:
            option_c_metrics["degenerate_rapm_gate_failed"] = True
            option_c_metrics["degenerate_rapm_error"] = str(exc)
            log.append(f"- **Option C degenerate RAPM gate:** FAIL — {exc}")

        try:
            bench = benchmark_rapm_vs_estimated_net(
                SEASON, rapm=impact_ctx.rapm, min_players=100
            )
            option_c_metrics["benchmark"] = bench.to_dict()
            metrics["rapm_benchmark_correlation"] = bench.spearman_rho
            log.append(
                f"- **RAPM benchmark:** spearman={bench.spearman_rho} "
                f"n={bench.n_overlap} ({bench.e_net_column})"
            )
        except Exception as exc:  # noqa: BLE001
            option_c_metrics["benchmark"] = {"error": str(exc)}
            log.append(f"- **RAPM benchmark:** fail — {exc}")

        from nba_fit.scoring.ranker import FitRanker  # noqa: E402

        try:
            ranker = FitRanker.from_season(
                SEASON, prefer_interim=True, prefer_api=False
            )
            card = ranker.fit_card(DEMO_PLAYER_ID, DEMO_TEAM_ID)
            delta = card.get("projected_net_rating_delta")
            option_c_metrics["demo_fit_card"] = {
                "player_id": DEMO_PLAYER_ID,
                "team_id": DEMO_TEAM_ID,
                "projected_net_rating_delta": delta,
                "non_zero_delta": delta is not None and abs(float(delta)) > 1e-9,
            }
            log.append(
                f"- **Demo fit card (LAL):** projected_net_rating_delta={delta}"
            )
        except Exception as exc:  # noqa: BLE001
            option_c_metrics["demo_fit_card"] = {"error": str(exc)}
            log.append(f"- **Demo fit card:** fail — {exc}")

    metrics["option_c"] = option_c_metrics
    (_OUT / "option_c_metrics.json").write_text(
        json.dumps(option_c_metrics, indent=2, default=str),
        encoding="utf-8",
    )

    log.extend(["", "## Movement backtest (no synthetic)", ""])
    movements, movements_synthetic, movements_source = _resolve_movements(
        allow_api=allow_api
    )
    backtest_result = None
    scoring_ctx: SeasonFitContext | None = None

    try:
        if not all_ready:
            raise RuntimeError(f"missing partitions: {missing_tables}")
        if movements_synthetic:
            log.append(f"- **Movements gate:** {movements_source}")
        else:
            context = SeasonFitContext.build(
                SEASON, prefer_interim=True, prefer_api=False
            )
            gamelogs = load_gamelogs_for_season(
                SEASON, allow_api=allow_api, use_cache=True
            )
            if gamelogs is not None and not gamelogs.empty:
                context = context.with_gamelogs(gamelogs)
            if role_ctx is None:
                role_ctx = RoleFitContext.from_season(
                    SEASON, prefer_interim=True, prefer_api=False, synthetic=False
                )
            if impact_ctx is None:
                impact_ctx = ImpactFitContext.from_season(
                    SEASON, prefer_interim=True, synthetic=False
                )

            log.append(f"- **Real movements:** {len(movements)} ({movements_source})")
            scoring_ids = _scoring_player_ids(context, movements)
            scoring_ctx = _subset_context(context, scoring_ids)
            log.append(f"- **Scoring subset:** {len(scoring_ctx.players)} players")

            backtest_result = run_movement_backtest(
                scoring_ctx,
                movements,
                role_context=role_ctx,
                impact_context=impact_ctx,
                allow_api_gamelogs=allow_api,
            )
            backtest_result.rows.to_csv(_OUT / "backtest_results.csv", index=False)
            log.append(f"- **Backtest rows:** {backtest_result.n_movements}")
    except Exception as exc:  # noqa: BLE001
        log.append(f"- **Status:** fail — {exc}")

    log.extend(["", "## fit-weights", ""])
    weight_learning: dict[str, Any] = {"fitted": False}
    try:
        if backtest_result is not None and not backtest_result.rows.empty:
            rows = backtest_result.rows
            comp_cols = [f"ensemble_{name}" for name in ENSEMBLE_COMPONENT_NAMES]
            if all(c in rows.columns for c in comp_cols) and "post_move_outcome" in rows.columns:
                X = rows[comp_cols].to_numpy(dtype=float)
                y = rows["post_move_outcome"].to_numpy(dtype=float)
                fit = learn_ensemble_weights(X, y)
                weights_path = save_ensemble_weights(fit, SEASON)
                weight_learning = {
                    "fitted": True,
                    "n_samples": fit.n_samples,
                    "train_mse": fit.train_mse,
                    "l2": fit.l2,
                    "weights": fit.weights,
                    "prior_weights": fit.prior_weights,
                    "artifact_path": str(weights_path.relative_to(_REPO)),
                }
                metrics["learned_ensemble_weights"] = fit.weights
                metrics["ensemble_component_weights_active"] = resolve_ensemble_component_weights(
                    SEASON
                )
                log.append(
                    f"- **fit-weights:** pass n={fit.n_samples} mse={fit.train_mse:.4f} "
                    f"saved={weights_path.name}"
                )
            else:
                log.append("- **fit-weights:** skipped (missing ensemble columns)")
        else:
            log.append("- **fit-weights:** skipped (no backtest rows)")
    except Exception as exc:  # noqa: BLE001
        log.append(f"- **fit-weights:** fail — {exc}")
    metrics["weight_learning"] = weight_learning

    log.extend(["", "## Ablation", ""])
    ablation_summary: list[dict[str, Any]] = []
    try:
        if backtest_result is not None and not movements.empty:
            ablation_ctx = scoring_ctx
            if ablation_ctx is None:
                ablation_ctx = SeasonFitContext.build(
                    SEASON, prefer_interim=True, prefer_api=False
                )
                scoring_ids = _scoring_player_ids(ablation_ctx, movements)
                ablation_ctx = _subset_context(ablation_ctx, scoring_ids)
            report = run_ablation(
                ablation_ctx,
                movements,
                role_context=role_ctx,
                impact_context=impact_ctx,
            )
            table = report.summary_table()
            table.to_csv(_OUT / "ablation_summary.csv", index=False)
            ablation_summary = table.to_dict(orient="records")
            ablation_path = write_ablation_metrics(report, _OUT / "ablation_metrics.json")
            metrics["ablation_metrics_path"] = str(ablation_path.relative_to(_REPO))
            log.append(f"- **Variants:** {list(report.variants.keys())}")
            log.append(f"- **Ablation JSON:** `{ablation_path.name}`")
        else:
            log.append("- **Ablation:** skipped")
    except Exception as exc:  # noqa: BLE001
        log.append(f"- **Ablation:** fail — {exc}")
    metrics["ablation"] = ablation_summary

    log.extend(["", "## Holdout", ""])
    holdout = run_holdout_season(
        SEASON,
        prefer_interim=all_ready,
        synthetic_fallback=True,
    )
    metrics["holdout"] = holdout.to_dict()
    log.append(
        f"- **Holdout:** train={holdout.train_season} predict={holdout.predict_season} "
        f"source={holdout.data_source}"
    )

    log.extend(["", "## Calibration holdout (train-season fit, eval on holdout)", ""])
    calibration_holdout: dict[str, Any] = {"eval_season": SEASON, "fitted": False}
    try:
        if backtest_result is not None and not backtest_result.rows.empty:
            labeled = backtest_result.rows
            train_rows, eval_rows = split_movements_by_season(labeled, eval_season=SEASON)
            if train_rows.empty and len(labeled) > 2:
                n_train = max(2, int(len(labeled) * 0.6))
                train_rows = labeled.iloc[:n_train]
                eval_rows = labeled.iloc[n_train:]
            calibrator = fit_isotonic_on_train_rows(train_rows)
            prior_seasons = tuple(
                sorted({str(s) for s in train_rows["season"].unique()})
            ) if "season" in train_rows.columns and not train_rows.empty else ()
            holdout_cal = evaluate_holdout_calibration(
                calibrator,
                eval_rows,
                train_seasons=prior_seasons,
                eval_season=SEASON,
                n_train=len(train_rows),
            )
            calibration_holdout = holdout_cal.to_dict()
            calibration_holdout["fitted"] = calibrator is not None and calibrator.is_fitted_
            log.append(
                f"- **Calibration holdout:** n_train={holdout_cal.n_train} "
                f"n_eval={holdout_cal.n_eval} brier={holdout_cal.brier}"
            )
        else:
            log.append("- **Calibration holdout:** skipped (no backtest rows)")
    except Exception as exc:  # noqa: BLE001
        log.append(f"- **Calibration holdout:** fail — {exc}")
    metrics["calibration_holdout"] = calibration_holdout

    log.extend(["", "## Replacement-player benchmark", ""])
    replacement_metrics: dict[str, Any] = {"n_movements": 0}
    try:
        if scoring_ctx is not None and not movements.empty:
            fit_table = build_fit_index_table(
                scoring_ctx, role_context=role_ctx, impact_context=impact_ctx
            )
            replacement = rank_signing_vs_position_pool(
                movements, scoring_ctx, fit_table
            )
            replacement_path = _OUT / "replacement_benchmark.json"
            replacement_path.write_text(
                json.dumps(replacement.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
            replacement_metrics = replacement.to_dict()
            replacement_metrics["path"] = str(replacement_path.relative_to(_REPO))
            log.append(
                f"- **Replacement benchmark:** n={replacement.n_movements} "
                f"mean_rank_pct={replacement.mean_pool_rank_pct}"
            )
        else:
            log.append("- **Replacement benchmark:** skipped")
    except Exception as exc:  # noqa: BLE001
        log.append(f"- **Replacement benchmark:** fail — {exc}")
    metrics["replacement_benchmark"] = replacement_metrics

    calibration: dict[str, Any] = {"method": "isotonic", "fitted": False}
    backtest_metrics: dict[str, Any] = {
        "n_movements": int(backtest_result.n_movements) if backtest_result else len(movements),
        "movements_synthetic": movements_synthetic,
        "movements_source": movements_source,
        "n_real_movements_available": len(movements),
    }
    if backtest_result is not None:
        cal = backtest_result.calibrator
        rows = backtest_result.rows
        fitted = (cal is not None and cal.is_fitted_) or (
            not rows.empty and "calibrated_fit_percentile" in rows.columns
        )
        if fitted:
            calibration = {
                "method": cal.method if cal is not None else "isotonic",
                "fitted": True,
                "n_anchor": int(len(rows)),
            }
        backtest_metrics.update(
            {
                "mean_calibrated_fit_percentile": backtest_result.mean_destination_percentile,
                "mean_post_move_outcome": (
                    float(rows["post_move_outcome"].mean())
                    if "post_move_outcome" in rows.columns and not rows.empty
                    else None
                ),
            }
        )
    metrics["calibration"] = calibration
    metrics["backtest"] = backtest_metrics
    metrics["train_roles_ok"] = train_roles_ok
    metrics["train_impact_ok"] = train_impact_ok

    movement_backtest_json = {
        "generated_at_utc": metrics["generated_at_utc"],
        "season": SEASON,
        "movements_synthetic": movements_synthetic,
        "movements_source": movements_source,
        "n_movements": backtest_metrics.get("n_movements"),
        "min_real_movements_required": MIN_REAL_MOVEMENTS,
    }
    if backtest_result is not None and "label_source" in backtest_result.rows.columns:
        movement_backtest_json["label_sources"] = (
            backtest_result.rows["label_source"].value_counts().to_dict()
        )
    (_OUT / "movement_backtest.json").write_text(
        json.dumps(movement_backtest_json, indent=2, default=str),
        encoding="utf-8",
    )

    pytest_rc, pytest_out, pytest_targets = _run_pytest_subset()
    metrics["pytest"] = {
        "exit_code": pytest_rc,
        "targets": pytest_targets,
        "summary": pytest_out.strip().splitlines()[-1] if pytest_out.strip() else "",
    }
    log.extend(
        [
            "",
            "## Pytest",
            "",
            f"- **Exit code:** {pytest_rc}",
            f"```\n{pytest_out.strip() or '(no output)'}\n```",
            "",
            "## Visual tests",
            "",
        ]
    )

    figure_paths: list[str] = []
    backtest_csv = _OUT / "backtest_results.csv"
    if not backtest_csv.is_file():
        pd.DataFrame().to_csv(backtest_csv, index=False)
    try:
        figure_paths, caption = _run_visual_tests(backtest_csv)
        for p in figure_paths:
            log.append(f"- `{p}`")
        if caption:
            log.append(f"- **data_health:** {caption}")
    except Exception as exc:  # noqa: BLE001
        log.append(f"- **Visual tests:** fail — {exc}")
    metrics["visual_test_figures"] = figure_paths

    failures = _p0_gates(metrics)
    metrics["p0_gate_failures"] = failures
    metrics["p0_gates_passed"] = len(failures) == 0

    log.extend(["", "## P0 gates", ""])
    if failures:
        for f in failures:
            log.append(f"- **FAIL** `{f['gate']}`: {f['detail']}")
    else:
        log.append("- **All P0 gates passed**")
    log.append("")
    log.append(f"- **Finished (UTC):** {_utc_now()}")

    (_OUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    (_OUT / "RUN_LOG.md").write_text("\n".join(log), encoding="utf-8")

    print(json.dumps(metrics, indent=2, default=str))
    if failures:
        print("\nP0 GATE FAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f['gate']}: {f['detail']}", file=sys.stderr)
        return 1
    return 0 if pytest_rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
