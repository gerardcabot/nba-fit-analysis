"""Shared matplotlib/seaborn styling and figure export for visual tests."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from visual_tests import _constants

FIGURE_DPI = 120
DEFAULT_FIG_WIDTH_IN = 10.0
DEFAULT_FIG_HEIGHT_IN = 6.0
SEABORN_STYLE = "whitegrid"
SEABORN_CONTEXT = "notebook"
SEABORN_PALETTE = "colorblind"
FONT_SIZE_TITLE = 14
FONT_SIZE_AXIS_LABEL = 12
FONT_SIZE_TICK = 10
BAR_EDGE_WIDTH = 0.8
BAR_ALPHA = 0.9

COLOR_SUCCESS = "#2ca02c"
COLOR_EMPTY = "#ff7f0e"
COLOR_FAIL = "#d62728"
COLOR_NEUTRAL = "#4c72b0"


def apply_plot_style() -> None:
    """Apply a consistent seaborn/matplotlib theme for all visual tests."""
    sns.set_theme(
        style=SEABORN_STYLE,
        context=SEABORN_CONTEXT,
        palette=SEABORN_PALETTE,
        font_scale=1.0,
    )
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.labelsize": FONT_SIZE_AXIS_LABEL,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
        }
    )


def new_figure(
    width_in: float = DEFAULT_FIG_WIDTH_IN,
    height_in: float = DEFAULT_FIG_HEIGHT_IN,
) -> tuple[plt.Figure, plt.Axes]:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    return fig, ax


def _figures_dir() -> Path:
    """Resolve output dir (``run_*_validation`` may rebind ``_constants.FIGURES_DIR``)."""
    return _constants.FIGURES_DIR


def ensure_figures_dir() -> Path:
    out = _figures_dir()
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(fig: plt.Figure, stem: str, *, subdir: str | None = None) -> Path:
    """Save *fig* under reports/figures/ as PNG; return the output path."""
    out_dir = ensure_figures_dir()
    if subdir:
        out_dir = out_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    plt.close(fig)
    return path
