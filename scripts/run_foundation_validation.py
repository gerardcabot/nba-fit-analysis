#!/usr/bin/env python3
"""Run Foundation real-data validation and write reports/validation/foundation/ artifacts."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "reports" / "validation" / "foundation"
_FIGURES_OUT = _OUT / "figures"
_FIGURES_SRC = _REPO / "reports" / "figures"

FETCH_ENDPOINT = "leaguedashplayerstats"
FETCH_SEASON = "2024-25"
VISUAL_STEMS = ("01_endpoint_health", "02_pbp_action_type_distribution")

if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from nba_fit.config.settings import get_settings  # noqa: E402
from nba_fit.data.registry import ProbeRegistry  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd or _REPO,
        capture_output=True,
        text=True,
        env=env,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, combined


def _parse_fetch_rows(output: str) -> int | None:
    match = re.search(r"rows=(\d+)", output)
    return int(match.group(1)) if match else None


def _live_pbp_row_count() -> int | None:
    """Row count from live PlayByPlayV3 (same game as visual_tests/02_pbp_schema.py)."""
    import importlib.util

    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    try:
        spec = importlib.util.spec_from_file_location(
            "pbp_schema",
            _REPO / "visual_tests" / "02_pbp_schema.py",
        )
        if not spec or not spec.loader:
            return None
        pbp_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pbp_mod)
        game_id = pbp_mod._probe_game_id()
        df = pbp_mod._fetch_playbyplay_v3(game_id)
        return len(df)
    except Exception:
        return None


def _parse_pytest_summary(output: str) -> dict[str, object]:
    summary_line = ""
    for line in reversed(output.strip().splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            summary_line = line.strip()
            break
    passed = failed = skipped = 0
    m_pass = re.search(r"(\d+)\s+passed", summary_line)
    m_fail = re.search(r"(\d+)\s+failed", summary_line)
    m_skip = re.search(r"(\d+)\s+skipped", summary_line)
    if m_pass:
        passed = int(m_pass.group(1))
    if m_fail:
        failed = int(m_fail.group(1))
    if m_skip:
        skipped = int(m_skip.group(1))
    return {
        "summary_line": summary_line,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
    }


def _endpoint_counts() -> dict[str, int]:
    settings = get_settings()
    if not settings.probe_results_path.exists():
        return {"OK": 0, "EMPTY": 0, "FAIL": 0, "total": 0}
    registry = ProbeRegistry.load(settings=settings)
    summary = registry.summary()
    summary["total"] = sum(summary.values())
    return summary


def _copy_figures(log_lines: list[str]) -> list[str]:
    copied: list[str] = []
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    for stem in VISUAL_STEMS:
        src = _FIGURES_SRC / f"{stem}.png"
        if not src.is_file():
            log_lines.append(f"- **Figure missing:** `{src.name}`")
            continue
        dest = _FIGURES_OUT / src.name
        shutil.copy2(src, dest)
        copied.append(dest.name)
        log_lines.append(f"- Copied `{src.name}` -> `figures/{src.name}`")
    return copied


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    _FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = [
        "# Foundation validation run log",
        "",
        f"- **Started (UTC):** {_utc_now()}",
        f"- **Branch:** stage/00-foundation",
        "",
        "## Commands",
        "",
        "1. `python -m nba_fit health`",
        (
            f"2. `python -m nba_fit fetch-sample --endpoint {FETCH_ENDPOINT} "
            f"--season {FETCH_SEASON}`"
        ),
        "3. `pytest tests/test_config.py tests/test_client_smoke.py -q`",
        "4. `python visual_tests/01_endpoint_health.py`",
        "5. `python visual_tests/02_pbp_schema.py` (live PlayByPlayV3, one game)",
        "",
    ]

    endpoint_counts = _endpoint_counts()

    log_lines.extend(["## Health", ""])
    health_rc, health_out = _run([sys.executable, "-m", "nba_fit", "health"])
    (_OUT / "health_output.txt").write_text(health_out, encoding="utf-8")
    log_lines.append(f"- **Exit code:** {health_rc}")
    log_lines.append("```")
    log_lines.extend(health_out.strip().splitlines() or ["(no output)"])
    log_lines.append("```")
    log_lines.append("")

    log_lines.extend(["## Fetch sample", ""])
    fetch_cmd = [
        sys.executable,
        "-m",
        "nba_fit",
        "fetch-sample",
        "--endpoint",
        FETCH_ENDPOINT,
        "--season",
        FETCH_SEASON,
    ]
    fetch_rc, fetch_out = _run(fetch_cmd)
    fetch_rows = _parse_fetch_rows(fetch_out)
    log_lines.append(f"- **Exit code:** {fetch_rc}")
    log_lines.append(f"- **Rows:** {fetch_rows if fetch_rows is not None else 'unknown'}")
    log_lines.append("```")
    log_lines.extend(fetch_out.strip().splitlines() or ["(no output)"])
    log_lines.append("```")
    log_lines.append("")

    log_lines.extend(["## Pytest", ""])
    pytest_rc, pytest_out = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_config.py",
            "tests/test_client_smoke.py",
            "-q",
        ]
    )
    pytest_info = _parse_pytest_summary(pytest_out)
    log_lines.append(f"- **Exit code:** {pytest_rc}")
    log_lines.append(f"- **Summary:** {pytest_info.get('summary_line') or '(none)'}")
    log_lines.append("```")
    log_lines.extend(pytest_out.strip().splitlines() or ["(no output)"])
    log_lines.append("```")
    log_lines.append("")

    vis_env = {**os.environ, "PYTHONPATH": str(_REPO)}
    log_lines.extend(["## Visual tests", ""])

    vis01_rc, vis01_out = _run(
        [sys.executable, str(_REPO / "visual_tests" / "01_endpoint_health.py")],
        cwd=_REPO,
        env=vis_env,
    )
    log_lines.append(f"- **01_endpoint_health.py** exit={vis01_rc}")
    if vis01_out.strip():
        log_lines.append(f"  `{vis01_out.strip().splitlines()[-1]}`")

    vis02_rc, vis02_out = _run(
        [sys.executable, str(_REPO / "visual_tests" / "02_pbp_schema.py")],
        cwd=_REPO,
        env=vis_env,
    )
    pbp_rows = _live_pbp_row_count() if vis02_rc == 0 else None
    log_lines.append(f"- **02_pbp_schema.py** exit={vis02_rc}, pbp_rows={pbp_rows}")
    if vis02_out.strip():
        log_lines.append(f"  `{vis02_out.strip().splitlines()[-1]}`")
    log_lines.append("")

    log_lines.append("## Figures")
    log_lines.append("")
    figure_names = _copy_figures(log_lines)
    log_lines.append("")

    log_lines.append(f"- **Finished (UTC):** {_utc_now()}")

    metrics: dict[str, object] = {
        "generated_at_utc": _utc_now(),
        "branch": "stage/00-foundation",
        "endpoint_health": endpoint_counts,
        "health_cli_exit_code": health_rc,
        "fetch_sample": {
            "endpoint": FETCH_ENDPOINT,
            "season": FETCH_SEASON,
            "exit_code": fetch_rc,
            "row_count": fetch_rows,
        },
        "pytest": {
            "exit_code": pytest_rc,
            **pytest_info,
        },
        "pbp_live": {
            "exit_code": vis02_rc,
            "row_count": pbp_rows,
        },
        "visual_tests": {
            "01_endpoint_health_exit_code": vis01_rc,
            "02_pbp_schema_exit_code": vis02_rc,
            "figures": figure_names,
        },
    }

    (_OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (_OUT / "RUN_LOG.md").write_text("\n".join(log_lines), encoding="utf-8")

    print(json.dumps(metrics, indent=2))

    ok = (
        health_rc == 0
        and fetch_rc == 0
        and pytest_rc == 0
        and vis01_rc == 0
        and vis02_rc == 0
        and fetch_rows is not None
        and fetch_rows > 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
