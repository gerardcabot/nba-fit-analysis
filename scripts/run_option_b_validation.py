#!/usr/bin/env python3
"""Run Option B real-data validation and write reports/validation/option_b/ artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "reports" / "validation" / "option_b"
_FIGURES_OUT = _OUT / "figures"

PLAYER_ID = 2544
TEAM_ID = 1610612747
TOP_N_BOARD = 25
TOP_N_RANK = 30
PRIMARY_SEASON = "2024-25"

if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from nba_fit.config.settings import (  # noqa: E402
    INGEST_TIER_MVP,
    INGEST_TIER_ROLE,
    NBA_TEAM_IDS,
    OPTION_B_ROLE_ENDPOINTS,
)
from nba_fit.data.ingest import run_ingest  # noqa: E402
from nba_fit.normalize.lineups import interim_onoff_path, interim_lineup_units_path  # noqa: E402
from nba_fit.models.archetypes import archetype_label_counts  # noqa: E402
from nba_fit.models.role_context import RoleFitContext, train_roles_for_season  # noqa: E402
from nba_fit.scoring.archetype_board import archetype_board_for_team  # noqa: E402
from nba_fit.scoring.fit_card import build_fit_card, fit_card_to_json  # noqa: E402
from nba_fit.scoring.fit_index import build_fit_index_table  # noqa: E402
from nba_fit.scoring.ranker import FitRanker  # noqa: E402
from nba_fit.features.season_context import SeasonFitContext  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str, log_lines: list[str]) -> None:
    stamp = _utc_now()
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    log_lines.append(line)


def _run_ingest_tier(
    season: str,
    tier: str,
    log_lines: list[str],
    *,
    note: str | None = None,
) -> dict[str, object]:
    if note:
        _log(note, log_lines)
    t0 = time.perf_counter()
    result = run_ingest(season=season, tier=tier, use_cache=True)
    elapsed = time.perf_counter() - t0
    summary: dict[str, object] = {
        "season": season,
        "tier": result.tier,
        "elapsed_sec": round(elapsed, 1),
        "player_rows": result.player_rows,
        "team_rows": result.team_rows,
        "lineup_units_rows": result.lineup_units_rows,
        "onoff_rows": result.onoff_rows,
        "players_path": result.players_path,
        "teams_path": result.teams_path,
        "lineup_units_path": result.lineup_units_path,
        "onoff_path": result.onoff_path,
        "endpoints": {
            ep: sum(len(df) for df in fr.frames.values())
            for ep, fr in result.fetched.items()
        },
    }
    _log(
        f"Ingest tier={tier} done in {elapsed:.1f}s "
        f"(endpoints={list(summary.get('endpoints', {}))})",
        log_lines,
    )
    return summary


def _run_pytest() -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_archetypes.py", "-q"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _role_interim_ready(season: str) -> bool:
    return interim_onoff_path(season).exists() and interim_lineup_units_path(season).exists()


def _ensure_out() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)


def _load_visual_test_module(script_stem: str):
    """Load visual_tests/NN_name.py (numeric stems are invalid as import names)."""
    import importlib.util

    path = _REPO / "visual_tests" / f"{script_stem}.py"
    spec = importlib.util.spec_from_file_location(f"_vt_{script_stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load visual test module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _copy_visual_figure(src: Path, dest_name: str) -> Path | None:
    if not src.is_file():
        return None
    dest = _FIGURES_OUT / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    return dest


def _run_visual_tests(log_lines: list[str]) -> dict[str, object]:
    """Run visual_tests 06–08 in-process; save PNGs under validation figures/."""
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    import visual_tests._constants as vt_constants  # noqa: WPS433
    import visual_tests._plot_utils as plot_utils  # noqa: WPS433

    _ensure_out()
    vt_constants.FIGURES_DIR = _FIGURES_OUT
    plot_utils.FIGURES_DIR = _FIGURES_OUT

    modules = [
        ("06_onoff_minutes", "06_onoff_minutes.png"),
        ("07_archetype_map", "07_archetype_map.png"),
        ("08_team_need_radar", "08_team_need_radar.png"),
    ]
    results: dict[str, object] = {}
    for script_stem, dest_name in modules:
        short = script_stem
        _log(f"Visual test {short}...", log_lines)
        try:
            mod = _load_visual_test_module(script_stem)
            rc = int(mod.main())
        except Exception as exc:  # noqa: BLE001
            rc = 1
            results[short] = {"exit_code": rc, "error": str(exc), "figures": []}
            _log(f"  {short} failed: {exc}", log_lines)
            continue

        copied: list[str] = []
        for candidate in sorted(_FIGURES_OUT.rglob(f"{short}*.png")):
            rel = _copy_visual_figure(candidate, dest_name)
            if rel is not None:
                copied.append(str(rel.relative_to(_OUT)))
                break
        if not copied:
            for candidate in sorted((_REPO / "reports" / "figures").rglob(f"{short}*.png")):
                rel = _copy_visual_figure(candidate, dest_name)
                if rel is not None:
                    copied.append(str(rel.relative_to(_OUT)))
                    break
        results[short] = {"exit_code": rc, "figures": copied}
        _log(f"  {short} exit={rc} figures={len(copied)}", log_lines)
    return results


def _explained_variance_summary(role_ctx: RoleFitContext) -> dict[str, object]:
    evr = role_ctx.embeddings.explained_variance_ratio
    if evr is None or len(evr) == 0:
        return {"available": False}
    return {
        "available": True,
        "n_components": int(role_ctx.embeddings.n_components),
        "per_component": [round(float(x), 4) for x in evr],
        "cumulative": round(float(evr.sum()), 4),
    }


def main() -> int:
    _ensure_out()

    log_lines: list[str] = []
    api_notes: list[str] = []

    log_lines.extend(
        [
            "# Option B validation run log",
            "",
            f"- **Started (UTC):** {_utc_now()}",
            f"- **Season:** {PRIMARY_SEASON}",
            "",
            "## Commands",
            "",
            f"1. `python -m nba_fit ingest --tier mvp --season {PRIMARY_SEASON}`",
            f"2. `python -m nba_fit ingest --tier role --season {PRIMARY_SEASON}`",
            f"3. `python -m nba_fit train-roles --season {PRIMARY_SEASON}` (live interim)",
            f"4. `python -m nba_fit archetype-board {TEAM_ID} --top {TOP_N_BOARD}`",
            f"5. `python -m nba_fit rank-player {PLAYER_ID} --fit-card-team {TEAM_ID}`",
            "6. `pytest tests/test_archetypes.py -q`",
            "7. `visual_tests/06_onoff_minutes.py`, `07_archetype_map.py`, `08_team_need_radar.py`",
            "",
            "## Ingest (MVP)",
            "",
        ]
    )

    ingest_mvp: dict[str, object] = {}
    ingest_role: dict[str, object] = {}
    ingest_ok = False

    try:
        ingest_mvp = _run_ingest_tier(PRIMARY_SEASON, INGEST_TIER_MVP, log_lines)
        log_lines.extend(["", "## Ingest (role — 30 teams × team-scoped endpoints)", ""])
        n_team_eps = sum(
            1 for ep in OPTION_B_ROLE_ENDPOINTS if ep.lower() in {
                "teamdashlineups",
                "teamplayeronoffsummary",
                "teamplayeronoffdetails",
            }
        )
        _log(
            f"Role tier: {len(OPTION_B_ROLE_ENDPOINTS)} endpoints, "
            f"~{len(NBA_TEAM_IDS)} teams each for {n_team_eps} team-scoped pulls "
            f"(cache speeds repeat runs).",
            log_lines,
        )
        if _role_interim_ready(PRIMARY_SEASON):
            _log(
                "Role interim tables already present; skipping role-tier re-ingest.",
                log_lines,
            )
            ingest_role = {
                "season": PRIMARY_SEASON,
                "tier": INGEST_TIER_ROLE,
                "skipped": True,
                "onoff_path": str(interim_onoff_path(PRIMARY_SEASON)),
                "lineup_units_path": str(interim_lineup_units_path(PRIMARY_SEASON)),
            }
        else:
            ingest_role = _run_ingest_tier(
                PRIMARY_SEASON,
                INGEST_TIER_ROLE,
                log_lines,
                note="Starting role-tier ingest (may take several minutes on cold cache)...",
            )
        ingest_ok = True
        log_lines.append("- **Status:** pass")
    except Exception as exc:  # noqa: BLE001
        if _role_interim_ready(PRIMARY_SEASON):
            ingest_ok = True
            ingest_role = {
                "season": PRIMARY_SEASON,
                "tier": INGEST_TIER_ROLE,
                "skipped_after_error": True,
                "onoff_path": str(interim_onoff_path(PRIMARY_SEASON)),
                "lineup_units_path": str(interim_lineup_units_path(PRIMARY_SEASON)),
            }
            api_notes.append(f"role re-ingest failed (using existing interim): {exc}")
            log_lines.append(f"- **Status:** pass (existing interim; re-ingest error: {exc})")
            _log(f"Role re-ingest failed; using existing interim tables: {exc}", log_lines)
        else:
            log_lines.append(f"- **Status:** fail — {exc}")
            api_notes.append(str(exc))
            _log(f"Ingest failed: {exc}", log_lines)

    log_lines.extend(["", "## train-roles (live interim)", ""])

    role_ctx: RoleFitContext | None = None
    train_ok = False
    try:
        _log("Training role embeddings + archetypes from interim tables...", log_lines)
        t0 = time.perf_counter()
        role_ctx = train_roles_for_season(
            PRIMARY_SEASON,
            prefer_interim=True,
            prefer_api=False,
            synthetic=False,
        )
        train_elapsed = time.perf_counter() - t0
        train_ok = True
        n_arch = len(set(role_ctx.archetypes.archetype_labels))
        _log(
            f"train-roles done in {train_elapsed:.1f}s — "
            f"{len(role_ctx.embeddings.player_ids)} players, {n_arch} archetype labels",
            log_lines,
        )
        log_lines.append(f"- **Players embedded:** {len(role_ctx.embeddings.player_ids)}")
        log_lines.append(f"- **Archetype labels (unique):** {n_arch}")
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"- **Status:** fail — {exc}")
        api_notes.append(f"train-roles: {exc}")

    log_lines.extend(["", "## archetype-board", ""])

    board = pd.DataFrame()
    if train_ok and role_ctx is not None:
        try:
            board = archetype_board_for_team(
                TEAM_ID,
                PRIMARY_SEASON,
                top_n=TOP_N_BOARD,
                prefer_interim=True,
                prefer_api=False,
                synthetic=False,
            )
            _ensure_out()
            board.to_csv(_OUT / "archetype_board.csv", index=False)
            log_lines.append(f"- **Rows:** {len(board)}")
        except Exception as exc:  # noqa: BLE001
            log_lines.append(f"- **Status:** fail — {exc}")
            api_notes.append(f"archetype-board: {exc}")

    log_lines.extend(["", "## rank-player + fit card", ""])

    player_rank = pd.DataFrame()
    card: dict[str, object] = {}
    if train_ok:
        try:
            context = SeasonFitContext.build(
                PRIMARY_SEASON, prefer_interim=True, prefer_api=False
            )
            table = build_fit_index_table(context, role_context=role_ctx)
            ranker = FitRanker(context=context, table=table)
            player_rank = ranker.rank_destinations_for_player(PLAYER_ID, top_n=TOP_N_RANK)
            _ensure_out()
            player_rank.to_csv(_OUT / "rankings_player_2544.csv", index=False)
            card = ranker.fit_card(PLAYER_ID, TEAM_ID, role_context=role_ctx)
            _ensure_out()
            (_OUT / "fit_card_2544_1610612747.json").write_text(
                fit_card_to_json(card), encoding="utf-8"
            )
            log_lines.append(f"- **Destination rows:** {len(player_rank)}")
            log_lines.append(
                f"- **Fit card archetype:** {card.get('archetype')} "
                f"(comps={len(card.get('comps') or [])})"
            )
        except Exception as exc:  # noqa: BLE001
            log_lines.append(f"- **Status:** fail — {exc}")
            api_notes.append(f"rank-player: {exc}")

    log_lines.extend(["", "## Pytest", ""])
    pytest_rc, pytest_out = _run_pytest()
    log_lines.append(f"```\n{pytest_out.strip() or '(no output)'}\n```")
    log_lines.append(f"- **Exit code:** {pytest_rc}")

    log_lines.extend(["", "## Visual tests", ""])
    visual_results = _run_visual_tests(log_lines)

    ev_summary = _explained_variance_summary(role_ctx) if role_ctx else {"available": False}
    arch_counts = (
        archetype_label_counts(role_ctx.archetypes) if role_ctx is not None else {}
    )

    metrics: dict[str, object] = {
        "generated_at_utc": _utc_now(),
        "season": PRIMARY_SEASON,
        "ingest_ok": ingest_ok,
        "train_roles_ok": train_ok,
        "ingest_mvp": ingest_mvp,
        "ingest_role": ingest_role,
        "api_notes": api_notes,
        "n_players_embedded": len(role_ctx.embeddings.player_ids) if role_ctx else 0,
        "n_archetype_labels": len(set(role_ctx.archetypes.archetype_labels))
        if role_ctx
        else 0,
        "n_archetype_clusters": len(set(role_ctx.archetypes.cluster_ids))
        if role_ctx
        else 0,
        "archetype_label_counts": arch_counts,
        "explained_variance": ev_summary,
        "embedding_method": role_ctx.embeddings.method if role_ctx else None,
        "pytest_exit_code": pytest_rc,
        "visual_tests": visual_results,
        "top5_archetype_board": (
            board.head(5).to_dict(orient="records") if not board.empty else []
        ),
        "fit_card_archetype": card.get("archetype"),
        "fit_card_comps_count": len(card.get("comps") or []),
    }
    _ensure_out()
    (_OUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    log_lines.extend(["", f"- **Finished (UTC):** {_utc_now()}"])
    _ensure_out()
    (_OUT / "RUN_LOG.md").write_text("\n".join(log_lines), encoding="utf-8")

    print(json.dumps(metrics, indent=2, default=str))
    ok = ingest_ok and train_ok and not board.empty and pytest_rc == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
