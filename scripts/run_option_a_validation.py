#!/usr/bin/env python3
"""Run Option A real-data validation and write reports/validation/option_a/ artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "reports" / "validation" / "option_a"
_FIGURES_OUT = _OUT / "figures"

PLAYER_ID = 2544
TEAM_ID = 1610612747
TOP_N = 30
PRIMARY_SEASON = "2024-25"
FALLBACK_SEASON = "2025-26"

if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from nba_fit.config.settings import INGEST_TIER_MVP  # noqa: E402
from nba_fit.data.ingest import run_ingest  # noqa: E402
from nba_fit.features.season_context import SeasonFitContext  # noqa: E402
from nba_fit.scoring.constants import SUBMETRIC_NAMES  # noqa: E402
from nba_fit.scoring.fit_card import build_fit_card, fit_card_to_json  # noqa: E402
from nba_fit.scoring.fit_index import build_fit_index_table  # noqa: E402
from nba_fit.scoring.ranker import FitRanker  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_pytest() -> tuple[int, str]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_fit_index.py",
            "tests/test_ingest_smoke.py",
            "-q",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _submetric_stats_top_destinations(
    ranker: FitRanker, player_id: int, *, top_k: int = 5
) -> dict[str, dict[str, float]]:
    dest = ranker.rank_destinations_for_player(player_id, top_n=top_k)
    if dest.empty:
        return {}
    stats: dict[str, dict[str, float]] = {}
    for name in SUBMETRIC_NAMES:
        if name not in dest.columns:
            continue
        vals = dest[name].astype(float)
        stats[name] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std(ddof=0)), 4) if len(vals) > 1 else 0.0,
        }
    return stats


def _ingest_summary(season: str) -> tuple[str, dict[str, object], list[str]]:
    notes: list[str] = []
    for label, try_season in (("primary", PRIMARY_SEASON), ("fallback", FALLBACK_SEASON)):
        try:
            result = run_ingest(season=try_season, tier=INGEST_TIER_MVP, use_cache=True)
            summary: dict[str, object] = {
                "season": try_season,
                "tier": result.tier,
                "player_rows": result.player_rows,
                "team_rows": result.team_rows,
                "players_path": result.players_path,
                "teams_path": result.teams_path,
                "endpoints": {
                    ep: sum(len(df) for df in fr.frames.values())
                    for ep, fr in result.fetched.items()
                },
            }
            if label == "fallback":
                notes.append(f"Ingest used fallback season {try_season} (primary failed).")
            return try_season, summary, notes
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Ingest {try_season} failed: {exc}")
            if label == "fallback":
                raise
    raise RuntimeError("unreachable")


def _build_ranker(season: str, *, synthetic: bool) -> FitRanker:
    context = (
        SeasonFitContext.from_synthetic(season, n_players=120)
        if synthetic
        else SeasonFitContext.build(season, prefer_interim=True, prefer_api=False)
    )
    table = build_fit_index_table(context)
    return FitRanker(context=context, table=table)


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = [
        "# Option A validation run log",
        "",
        f"- **Started (UTC):** {_utc_now()}",
        f"- **Primary season:** {PRIMARY_SEASON}",
        f"- **Fallback season:** {FALLBACK_SEASON}",
        "",
        "## Commands",
        "",
        f"1. `python -m nba_fit ingest --season {PRIMARY_SEASON} --tier mvp`",
        f"2. Season context from interim (`prefer_interim=True`, `prefer_api=False`)",
        f"3. `rank-player {PLAYER_ID} --top {TOP_N}`",
        f"4. `rank-team {TEAM_ID} --top {TOP_N}`",
        f"5. Fit card player {PLAYER_ID} vs team {TEAM_ID}",
        "",
        "## Ingest",
        "",
    ]

    season = PRIMARY_SEASON
    ingest_ok = False
    ingest_summary: dict[str, object] = {}
    api_notes: list[str] = []

    try:
        season, ingest_summary, api_notes = _ingest_summary(season)
        ingest_ok = True
        log_lines.append(f"- **Status:** pass ({season})")
        log_lines.append(
            f"- **Rows:** players={ingest_summary.get('player_rows')}, "
            f"teams={ingest_summary.get('team_rows')}"
        )
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"- **Status:** fail — {exc}")
        api_notes.append(str(exc))

    for note in api_notes:
        log_lines.append(f"- {note}")

    log_lines.extend(["", "## Rankings (live / interim)", ""])

    used_synthetic = False
    ranker: FitRanker | None = None
    try:
        ranker = _build_ranker(season, synthetic=False)
        log_lines.append(
            f"- **Context source:** {ranker.context.source} (season={season})"
        )
    except Exception as exc:  # noqa: BLE001
        log_lines.append(f"- **Interim rankings failed:** {exc}")
        log_lines.extend(["", "## Rankings (synthetic fallback)", ""])
        used_synthetic = True
        ranker = _build_ranker(season, synthetic=True)
        log_lines.append(f"- **Context source:** synthetic (season={season})")

    assert ranker is not None

    player_rank = ranker.rank_destinations_for_player(PLAYER_ID, top_n=TOP_N)
    team_rank = ranker.rank_players_for_team(TEAM_ID, top_n=TOP_N)
    player_rank.to_csv(_OUT / "rankings_player_2544.csv", index=False)
    team_rank.to_csv(_OUT / "rankings_team_1610612747.csv", index=False)

    card = build_fit_card(PLAYER_ID, TEAM_ID, ranker.table, ranker.context)
    (_OUT / "fit_card_2544_1610612747.json").write_text(
        fit_card_to_json(card), encoding="utf-8"
    )

    submetric_top5 = _submetric_stats_top_destinations(ranker, PLAYER_ID, top_k=5)

    pytest_rc, pytest_out = _run_pytest()
    pytest_pass_count: int | None = None
    for line in reversed(pytest_out.strip().splitlines()):
        if " passed" in line:
            head = line.strip().split()[0]
            if head.isdigit():
                pytest_pass_count = int(head)
            break

    metrics: dict[str, object] = {
        "generated_at_utc": _utc_now(),
        "season": season,
        "ingest_ok": ingest_ok,
        "data_source": ranker.context.source,
        "used_synthetic_fallback": used_synthetic,
        "n_players": len(ranker.context.players),
        "n_teams": len(ranker.context.teams),
        "ingest": ingest_summary,
        "api_notes": api_notes,
        "player_id": PLAYER_ID,
        "team_id": TEAM_ID,
        "submetric_top5_destinations": submetric_top5,
        "pytest_exit_code": pytest_rc,
        "pytest_pass_count": pytest_pass_count,
        "pytest_summary": pytest_out.strip().splitlines()[-1] if pytest_out.strip() else "",
        "top3_player_destinations": (
            player_rank.head(3)[["rank", "team_id", "team", "overall_fit_percentile"]]
            .to_dict(orient="records")
            if not player_rank.empty
            else []
        ),
        "top3_team_targets": (
            team_rank.head(3)[["rank", "player_id", "overall_fit_percentile"]]
            .to_dict(orient="records")
            if not team_rank.empty
            else []
        ),
    }
    (_OUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    log_lines.extend(
        [
            f"- **Player rankings rows:** {len(player_rank)}",
            f"- **Team rankings rows:** {len(team_rank)}",
            f"- **Synthetic fallback:** {used_synthetic}",
            "",
            "## Pytest",
            "",
            f"```\n{pytest_out.strip() or '(no output)'}\n```",
            f"- **Exit code:** {pytest_rc}",
            "",
            "## Visual tests",
            "",
            "Run separately; PNGs copied to `figures/`.",
            "",
        ]
    )

    (_OUT / "RUN_LOG.md").write_text("\n".join(log_lines), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    return 0 if player_rank.shape[0] > 0 and team_rank.shape[0] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
