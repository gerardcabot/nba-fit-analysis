"""Paths and captions injected by ``scripts/run_*_validation.py`` before visual tests."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPTION_D_OUT = REPO_ROOT / "reports" / "validation" / "option_d"

# Set by run_option_d_validation.py immediately before visual tests run.
backtest_csv: Path | None = None
fit_card_json: Path | None = None
last_figure_caption: str | None = None


def resolve_backtest_csv() -> Path:
    if backtest_csv is not None:
        return backtest_csv
    return DEFAULT_OPTION_D_OUT / "backtest_results.csv"


def resolve_fit_card_json() -> Path:
    if fit_card_json is not None:
        return fit_card_json
    return DEFAULT_OPTION_D_OUT / "fit_card_2544_1610612747.json"
