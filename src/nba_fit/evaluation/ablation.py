"""Ablation runner: score Option A/B/C/D variants by toggling fit contexts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from nba_fit.evaluation.calibration_metrics import (
    brier_score,
    decile_lift,
    expected_calibration_error,
    spearman_rank_corr,
)
from nba_fit.evaluation.movement_backtest import (
    freeze_pre_move_features,
    label_post_move_outcomes,
)
from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.calibration import calibrate_fit_table
from nba_fit.models.impact_context import ImpactFitContext
from nba_fit.models.role_context import RoleFitContext

AblationVariant = Literal["A", "AB", "ABC", "D"]


@dataclass
class AblationVariantResult:
    """Scores for one ablation variant on movement rows."""

    variant: AblationVariant
    n_movements: int
    mean_raw_fit: float | None
    mean_calibrated_percentile: float | None
    brier: float | None
    ece: float | None
    decile_lift: float | None
    spearman: float | None
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class AblationReport:
    """Comparison table across A → D variants."""

    season: str
    variants: dict[str, AblationVariantResult] = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for name, result in self.variants.items():
            rows.append(
                {
                    "variant": name,
                    "n_movements": result.n_movements,
                    "mean_raw_fit": result.mean_raw_fit,
                    "mean_calibrated_pct": result.mean_calibrated_percentile,
                    "brier": result.brier,
                    "ece": result.ece,
                    "decile_lift": result.decile_lift,
                    "spearman": result.spearman,
                }
            )
        return pd.DataFrame(rows)


def _contexts_for_variant(
    variant: AblationVariant,
    *,
    role_context: RoleFitContext | None,
    impact_context: ImpactFitContext | None,
) -> tuple[Any | None, Any | None]:
    if variant == "A":
        return None, None
    if variant == "AB":
        return role_context, None
    if variant == "ABC":
        return role_context, impact_context
    # D — full stack (same contexts as ABC; calibration applied separately)
    return role_context, impact_context


def score_ablation_variant(
    context: SeasonFitContext,
    movements: pd.DataFrame,
    variant: AblationVariant,
    *,
    role_context: RoleFitContext | None = None,
    impact_context: ImpactFitContext | None = None,
    calibrate: bool = True,
) -> AblationVariantResult:
    """
    Score one ablation variant on historical movements.

    * **A** — Option A profile submetrics only (no role / impact contexts).
    * **AB** — adds Option B role embeddings and team need.
    * **ABC** — adds Option C lineup impact / RAPM context.
    * **D** — ABC + isotonic calibration on post-move outcomes.
    """
    role_ctx, impact_ctx = _contexts_for_variant(
        variant, role_context=role_context, impact_context=impact_context
    )

    if movements.empty:
        return AblationVariantResult(variant=variant, n_movements=0, mean_raw_fit=None, mean_calibrated_percentile=None, brier=None, ece=None, decile_lift=None, spearman=None)

    frozen = freeze_pre_move_features(
        context,
        movements,
        role_context=role_ctx,
        impact_context=impact_ctx,
    )
    labeled = label_post_move_outcomes(frozen, movements, context)

    use_calibration = calibrate and variant == "D"
    if use_calibration and not labeled.empty:
        labeled = calibrate_fit_table(
            labeled,
            raw_col="raw_fit_score",
            outcome_col="post_move_outcome",
            method="isotonic",
        )

    mean_raw = float(labeled["raw_fit_score"].mean()) if not labeled.empty else None
    mean_pct = None
    if "calibrated_fit_percentile" in labeled.columns and not labeled.empty:
        mean_pct = float(labeled["calibrated_fit_percentile"].mean())
    elif "pre_move_fit_percentile" in labeled.columns and not labeled.empty:
        mean_pct = float(labeled["pre_move_fit_percentile"].dropna().mean())

    brier = ece = lift = rho = None
    if not labeled.empty and "post_move_outcome" in labeled.columns:
        y = labeled["post_move_outcome"].to_numpy(dtype=float)
        if use_calibration and "calibrated_fit_percentile" in labeled.columns:
            p = labeled["calibrated_fit_percentile"].to_numpy(dtype=float) / 100.0
            s = p
        else:
            p = labeled["raw_fit_score"].to_numpy(dtype=float)
            s = p
        brier = brier_score(y, p)
        ece = expected_calibration_error(y, p)
        lift = decile_lift(y, s)
        rho = spearman_rank_corr(y, s)

    return AblationVariantResult(
        variant=variant,
        n_movements=len(labeled),
        mean_raw_fit=mean_raw,
        mean_calibrated_percentile=mean_pct,
        brier=brier,
        ece=ece,
        decile_lift=lift,
        spearman=rho,
        rows=labeled,
    )


def run_ablation(
    context: SeasonFitContext,
    movements: pd.DataFrame,
    *,
    role_context: RoleFitContext | None = None,
    impact_context: ImpactFitContext | None = None,
    variants: tuple[AblationVariant, ...] = ("A", "AB", "ABC", "D"),
) -> AblationReport:
    """Run all ablation variants and return a comparison report."""
    if role_context is None:
        try:
            role_context = RoleFitContext.from_synthetic(context)
        except ValueError:
            role_context = None

    if impact_context is None and role_context is not None:
        try:
            impact_context = ImpactFitContext.from_synthetic(role_context)
        except ValueError:
            impact_context = None

    report = AblationReport(season=context.season)
    for variant in variants:
        report.variants[variant] = score_ablation_variant(
            context,
            movements,
            variant,
            role_context=role_context,
            impact_context=impact_context,
        )
    return report


def ablation_report_to_dict(report: AblationReport) -> dict[str, object]:
    """Serialize ablation comparison for ``reports/validation/sota/ablation_metrics.json``."""
    variants: dict[str, dict[str, object]] = {}
    for key, result in report.variants.items():
        variants[key] = {
            "variant": result.variant,
            "label": _variant_label(result.variant),
            "n_movements": result.n_movements,
            "mean_raw_fit": result.mean_raw_fit,
            "mean_calibrated_percentile": result.mean_calibrated_percentile,
            "brier": result.brier,
            "ece": result.ece,
            "decile_lift": result.decile_lift,
            "spearman": result.spearman,
        }
    return {"season": report.season, "variants": variants}


def _variant_label(variant: AblationVariant) -> str:
    labels = {
        "A": "A (profile only)",
        "AB": "A+B (role/need)",
        "ABC": "A+B+C (lineup impact)",
        "D": "D (full + isotonic calibration)",
    }
    return labels.get(variant, variant)


def write_ablation_metrics(
    report: AblationReport,
    path: Path,
) -> Path:
    """Write ablation comparison JSON under ``reports/validation/sota/``."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ablation_report_to_dict(report), indent=2), encoding="utf-8")
    return path
