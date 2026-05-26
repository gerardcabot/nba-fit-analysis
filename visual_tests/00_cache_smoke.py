#!/usr/bin/env python3
"""Matplotlib bar chart of endpoint row counts from probe registry."""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root on path when run as script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import matplotlib.pyplot as plt

from nba_fit.config.settings import get_settings
from nba_fit.data.registry import ProbeRegistry


def main() -> int:
    settings = get_settings()
    registry = ProbeRegistry.load(settings=settings)
    counts = registry.row_counts()
    # Top 25 by rows for readability
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:25]
    names = [n for n, _ in top]
    values = [v for _, v in top]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(names[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("Total rows (probe)")
    ax.set_title("NBA API endpoint row counts (top 25 from probe)")
    fig.tight_layout()
    out = settings.reports_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "00_endpoint_row_counts.png"
    fig.savefig(path, dpi=120)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
