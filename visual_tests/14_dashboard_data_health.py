#!/usr/bin/env python3
"""Static matplotlib replica of the dashboard Data Health panel (no Streamlit in CI)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from nba_fit.app.data_health import (
    build_data_health_figure,
    essential_health_table,
    probe_summary,
)
from nba_fit.config.settings import get_settings
from nba_fit.data.registry import ProbeRegistry
from visual_tests import _validation_artifacts as validation_artifacts
from visual_tests._plot_utils import save_figure

OUTPUT_STEM = "14_dashboard_data_health"


def data_health_caption(registry: ProbeRegistry) -> str:
    """One-line summary for validation RUN_LOG."""
    summary = probe_summary(registry)
    essentials = essential_health_table(registry)
    n_ok = int((essentials["status"] == "OK").sum())
    return (
        f"probe OK={summary.get('OK', 0)} EMPTY={summary.get('EMPTY', 0)} "
        f"FAIL={summary.get('FAIL', 0)}; essential {n_ok}/{len(essentials)} OK"
    )


def main() -> int:
    settings = get_settings()
    try:
        registry = ProbeRegistry.load(settings=settings)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    fig = build_data_health_figure(registry)
    path = save_figure(fig, OUTPUT_STEM)
    caption = data_health_caption(registry)
    validation_artifacts.last_figure_caption = caption
    print(f"Wrote {path}")
    print(caption)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
