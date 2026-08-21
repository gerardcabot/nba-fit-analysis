"""Load SOTA validation artifacts from ``reports/validation/sota/``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
SOTA_VALIDATION_DIR = _REPO_ROOT / "reports" / "validation" / "sota"


def sota_dir() -> Path:
    return SOTA_VALIDATION_DIR


def load_metrics() -> dict | None:
    path = SOTA_VALIDATION_DIR / "metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_degradation_warnings() -> list[str]:
    path = SOTA_VALIDATION_DIR / "degradation_warnings.json"
    if not path.is_file():
        metrics = load_metrics() or {}
        warnings = metrics.get("degradation_warnings")
        if isinstance(warnings, list):
            return [str(w) for w in warnings]
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(w) for w in data]
    return [str(w) for w in data.get("warnings", [])]


def figure_path(name: str) -> Path | None:
    """Resolve a figure under ``figures/`` (e.g. ``12_calibration_curve``)."""
    for sub in ("", "dashboard", "option_d"):
        candidate = SOTA_VALIDATION_DIR / "figures" / sub / f"{name}.png"
        if candidate.is_file():
            return candidate
    flat = SOTA_VALIDATION_DIR / "figures" / f"{name}.png"
    return flat if flat.is_file() else None


def learned_weights_frame(metrics: dict) -> pd.DataFrame | None:
    learned = metrics.get("learned_ensemble_weights") or metrics.get("ensemble_learned_weights")
    if isinstance(learned, dict) and learned:
        return pd.DataFrame(
            {"component": list(learned.keys()), "weight": list(learned.values())}
        )
    prior = metrics.get("ensemble_component_weights")
    if isinstance(prior, dict) and prior:
        return pd.DataFrame(
            {"component": list(prior.keys()), "weight": list(prior.values())}
        )
    return None


def movement_source_label(metrics: dict) -> str:
    backtest = metrics.get("backtest") or {}
    if isinstance(backtest, dict) and backtest.get("movements_source"):
        return str(backtest["movements_source"])
    if metrics.get("movements_source"):
        return str(metrics["movements_source"])
    return "unknown"


def calibration_summary(metrics: dict) -> dict:
    cal = metrics.get("calibration") or {}
    if not isinstance(cal, dict):
        return {}
    return cal
