#!/usr/bin/env python3
"""End-to-end Option C validation for season 2024-25 (real PBP sample + artifacts)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEASON = "2024-25"
IMPACT_MAX_GAMES = 30
PLAYER_ID = 2544
TEAM_ID = 1610612747
OUT_DIR = REPO_ROOT / "reports" / "validation" / "option_c"
FIGURES_DIR = OUT_DIR / "figures"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

_SRC = str(REPO_ROOT / "src")


def _subprocess_env() -> dict[str, str]:
    """Pin imports to this repo so a different editable install cannot hijack CLI."""
    env = os.environ.copy()
    prefix = _SRC
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = prefix if not existing else f"{prefix}{os.pathsep}{existing}"
    return env


def _log(lines: list[str], msg: str) -> None:
    print(msg)
    lines.append(msg)


def _run_cmd(
    lines: list[str],
    args: list[str],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "nba_fit", *args]
    display = " ".join(cmd)
    _log(lines, f"\n$ {display}")
    proc = subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    if proc.stdout:
        _log(lines, proc.stdout.rstrip())
    if proc.stderr:
        _log(lines, proc.stderr.rstrip())
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {display}")
    return proc


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


def _lineups_have_ratings(season: str) -> bool:
    """True when interim lineup_units includes non-zero Advanced OFF_RATING."""
    if not _interim_ready(season):
        return False
    import pandas as pd

    from nba_fit.normalize.lineups import load_lineup_units_table

    df = load_lineup_units_table(season)
    if "OFF_RATING" not in df.columns:
        return False
    off = pd.to_numeric(df["OFF_RATING"], errors="coerce").fillna(0.0)
    return bool((off.abs() > 1e-6).any())


def _run_pytest(lines: list[str]) -> None:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_rapm.py",
        "tests/test_pbp_smoke.py",
        "-q",
    ]
    display = " ".join(cmd)
    _log(lines, f"\n$ {display}")
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    if proc.stdout:
        _log(lines, proc.stdout.rstrip())
    if proc.stderr:
        _log(lines, proc.stderr.rstrip())
    if proc.returncode != 0:
        raise RuntimeError(f"pytest failed ({proc.returncode})")


def _run_visual_tests(lines: list[str]) -> list[str]:
    import importlib
    import shutil

    import nba_fit.config.settings as nba_settings
    import visual_tests._constants as vt_constants
    import visual_tests._plot_utils as plot_utils

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    vt_constants.FIGURES_DIR = FIGURES_DIR
    plot_utils.FIGURES_DIR = FIGURES_DIR

    prev_season = nba_settings.DEFAULT_SEASON
    nba_settings.DEFAULT_SEASON = SEASON

    figure_paths: list[str] = []
    try:
        for mod_name in (
            "visual_tests.09_possession_rate",
            "visual_tests.10_rapm_distribution",
            "visual_tests.11_lineup_delta_bar",
        ):
            _log(lines, f"\n$ python {mod_name.replace('.', '/')}.py")
            if mod_name in sys.modules:
                mod = importlib.reload(sys.modules[mod_name])
            else:
                mod = importlib.import_module(mod_name)
            if hasattr(mod, "DEFAULT_SEASON"):
                mod.DEFAULT_SEASON = SEASON
            rc = int(mod.main())
            if rc != 0:
                raise RuntimeError(f"{mod_name} exited {rc}")
    finally:
        nba_settings.DEFAULT_SEASON = prev_season

    for png in sorted(FIGURES_DIR.rglob("*.png")):
        figure_paths.append(str(png.relative_to(REPO_ROOT)))

    default_fig = REPO_ROOT / "reports" / "figures"
    if default_fig.exists():
        for png in default_fig.rglob("09_possession_rate.png"):
            dest = FIGURES_DIR / png.name
            if not dest.exists():
                shutil.copy2(png, dest)
                figure_paths.append(str(dest.relative_to(REPO_ROOT)))

    return sorted(set(figure_paths))


def _rapm_summary(season: str) -> dict:
    from nba_fit.models.rapm import load_rapm

    rapm = load_rapm(season)
    net = rapm.net_rapm
    low = int(rapm.low_sample_flag.sum())
    return {
        "n_players": int(len(rapm.player_ids)),
        "low_sample_count": low,
        "net_rapm_min": float(net.min()),
        "net_rapm_max": float(net.max()),
        "net_rapm_mean": float(net.mean()),
        "net_rapm_std": float(net.std()),
        "net_rapm_median": float(__import__("numpy").median(net)),
        "lebron_net_rapm": _player_rapm(rapm, PLAYER_ID),
    }


def _player_rapm(rapm, player_id: int) -> float | None:
    impact = rapm.impact_for(player_id)
    return float(impact[2]) if impact else None


def _possession_metrics(season: str) -> dict:
    from nba_fit.normalize.possessions import (
        load_possessions_table,
        possessions_per_game_counts,
    )

    if not _possessions_ready(season):
        return {"games_ingested": 0, "total_possession_rows": 0, "per_game": {}}
    df = load_possessions_table(season)
    counts = possessions_per_game_counts(df)
    return {
        "games_ingested": int(len(counts)),
        "total_possession_rows": int(len(df)),
        "per_game_min": int(counts.min()) if len(counts) else 0,
        "per_game_max": int(counts.max()) if len(counts) else 0,
        "per_game_mean": float(counts.mean()) if len(counts) else 0.0,
    }


def main() -> int:
    started = datetime.now(timezone.utc)
    log_lines: list[str] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict = {
        "season": SEASON,
        "started_at": started.isoformat(),
        "impact_max_games": IMPACT_MAX_GAMES,
        "player_id": PLAYER_ID,
        "team_id": TEAM_ID,
    }

    _log(log_lines, f"Option C validation — season {SEASON}")

    if not _interim_ready(SEASON):
        _run_cmd(log_lines, ["ingest", "--tier", "mvp", "--season", SEASON])
        _run_cmd(log_lines, ["ingest", "--tier", "role", "--season", SEASON])
    elif not _lineups_have_ratings(SEASON):
        _log(
            log_lines,
            f"Re-ingesting role tier — lineup_units missing Advanced OFF/DEF ratings for {SEASON}",
        )
        _run_cmd(log_lines, ["ingest", "--tier", "role", "--season", SEASON])
    else:
        _log(log_lines, f"Skipping mvp/role ingest — interim tables cached for {SEASON}")

    _run_cmd(log_lines, ["train-roles", "--season", SEASON])

    if not _possessions_ready(SEASON):
        _run_cmd(
            log_lines,
            [
                "ingest",
                "--tier",
                "impact",
                "--season",
                SEASON,
                "--max-games",
                str(IMPACT_MAX_GAMES),
            ],
        )
    else:
        _log(
            log_lines,
            f"Skipping impact ingest — possessions cached for {SEASON}",
        )

    _run_cmd(log_lines, ["train-impact", "--season", SEASON])

    from nba_fit.scoring.lineup_sim import run_lineup_sim
    from nba_fit.scoring.ranker import FitRanker
    from nba_fit.scoring.fit_card import fit_card_to_json

    sim = run_lineup_sim(
        PLAYER_ID,
        TEAM_ID,
        SEASON,
        top_n=5,
        prefer_interim=True,
        prefer_api=False,
        synthetic=False,
    )
    lineup_payload = {
        "player_id": sim.player_id,
        "team_id": sim.team_id,
        "season": sim.season,
        "data_source": sim.data_source,
        "projected_net_rating_delta": sim.projected_net_rating_delta,
        "note": sim.note,
        "top_lineups": sim.lineup_synergy_block()["top_lineups"],
    }
    (OUT_DIR / "lineup_sim.json").write_text(
        json.dumps(lineup_payload, indent=2),
        encoding="utf-8",
    )
    _log(
        log_lines,
        f"lineup-sim projected_net_rating_delta: {sim.projected_net_rating_delta:+.2f}"
        if sim.projected_net_rating_delta is not None
        else "lineup-sim projected_net_rating_delta: n/a",
    )

    ranker = FitRanker.from_season(SEASON, prefer_interim=True, prefer_api=False)
    rankings = ranker.rank_destinations_for_player(PLAYER_ID, top_n=10)
    card = ranker.fit_card(PLAYER_ID, TEAM_ID)
    rankings_snippet = {
        "top_destinations": rankings.head(5).to_dict(orient="records"),
        "lakers_fit_card_excerpt": json.loads(fit_card_to_json(card)),
    }
    (OUT_DIR / "rankings_snippet.json").write_text(
        json.dumps(rankings_snippet, indent=2),
        encoding="utf-8",
    )

    _run_pytest(log_lines)
    figure_paths = _run_visual_tests(log_lines)

    metrics["possessions"] = _possession_metrics(SEASON)
    metrics["rapm"] = _rapm_summary(SEASON)
    metrics["lineup_sim"] = {
        "projected_net_rating_delta": sim.projected_net_rating_delta,
        "data_source": sim.data_source,
    }
    metrics["lineup_deltas"] = [
        {
            "lineup_key": u.lineup_key,
            "lineup_label": u.lineup_label,
            "projected_net_rating_delta": u.projected_net_rating_delta,
            "minutes": u.minutes,
        }
        for u in sim.top_lineups
    ]
    metrics["figures"] = figure_paths
    metrics["finished_at"] = datetime.now(timezone.utc).isoformat()

    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "RUN_LOG.md").write_text(
        "# Option C validation run\n\n"
        f"- Season: `{SEASON}`\n"
        f"- Started: `{started.isoformat()}`\n"
        f"- Impact max games: `{IMPACT_MAX_GAMES}`\n\n"
        "## Command log\n\n```text\n"
        + "\n".join(log_lines)
        + "\n```\n",
        encoding="utf-8",
    )

    _log(log_lines, f"\nWrote artifacts under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
