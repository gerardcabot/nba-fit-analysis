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

from nba_fit.app.data_health import build_data_health_figure, essential_health_table
from nba_fit.config.settings import get_settings
from nba_fit.data.registry import ProbeRegistry
from visual_tests._plot_utils import save_figure

OUTPUT_STEM = "14_dashboard_data_health"


def main() -> int:
    settings = get_settings()
    try:
        registry = ProbeRegistry.load(settings=settings)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    fig = build_data_health_figure(registry)
    path = save_figure(fig, OUTPUT_STEM, subdir="dashboard")
    print(f"Wrote {path}")

    essentials = essential_health_table(registry)
    n_ok = int((essentials["status"] == "OK").sum())
    print(f"Essential endpoints OK: {n_ok}/{len(essentials)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
