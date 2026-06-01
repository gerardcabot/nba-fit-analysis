"""Data health panel helpers (dashboard + static visual tests)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from nba_fit.config.endpoints import ENDPOINT_REGISTRY, ESSENTIAL_ENDPOINTS
from nba_fit.data.registry import EndpointHealth, ProbeRegistry

STATUS_COLORS = {
    "OK": "#2ca02c",
    "EMPTY": "#ff7f0e",
    "FAIL": "#d62728",
}


def probe_summary(registry: ProbeRegistry) -> dict[str, int]:
    """Counts of OK / EMPTY / FAIL across all probed endpoints."""
    return registry.summary()


def essential_health_rows(registry: ProbeRegistry) -> list[dict[str, Any]]:
    """One row per essential registry endpoint with probe status."""
    rows: list[dict[str, Any]] = []
    for name in ESSENTIAL_ENDPOINTS:
        spec = ENDPOINT_REGISTRY[name]
        probed: EndpointHealth | None = registry.get(name)
        if probed is None:
            status = "NOT_PROBED"
            total_rows: int | str = "-"
            elapsed = "-"
        else:
            status = probed.status
            total_rows = probed.total_rows
            elapsed = (
                f"{probed.elapsed_sec:.2f}s"
                if probed.elapsed_sec is not None
                else "-"
            )
        rows.append(
            {
                "endpoint": name,
                "status": status,
                "rows": total_rows,
                "reliability": spec.reliability,
                "category": spec.category,
                "elapsed": elapsed,
            }
        )
    return rows


def essential_health_table(registry: ProbeRegistry) -> pd.DataFrame:
    return pd.DataFrame(essential_health_rows(registry))


def build_data_health_figure(registry: ProbeRegistry) -> Figure:
    """
  Matplotlib replica of the dashboard Data Health status panel.

  Bar chart of OK / EMPTY / FAIL counts from probe_all_results.json.
  """
    summary = probe_summary(registry)
    categories = ["OK", "EMPTY", "FAIL"]
    counts = [summary.get(c, 0) for c in categories]
    colors = [STATUS_COLORS[c] for c in categories]

    meta = registry.meta
    season = meta.get("season", "unknown")
    timestamp = meta.get("timestamp", "unknown")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        categories,
        counts,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )
    ax.bar_label(bars, labels=[str(c) for c in counts], padding=4)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Endpoints")
    ax.set_xlabel("Probe status")
    ax.set_title(f"Endpoint probe health ({season}, probed {timestamp})")
    fig.tight_layout()
    return fig
