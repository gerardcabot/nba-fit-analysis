"""Historical movement backtest: freeze pre-move features, label post-move outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nba_fit.data.fetchers.transactions import MOVEMENT_COLUMNS, get_movements
from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.calibration import FitCalibrator, calibrate_fit_table
from nba_fit.scoring.constants import ENSEMBLE_DERIVED_NAMES, SUBMETRIC_NAMES
from nba_fit.scoring.ensemble import extract_ensemble_components, raw_ensemble_score
from nba_fit.scoring.fit_index import build_fit_index_table
from nba_fit.scoring.submetrics import compute_all_submetrics
from nba_fit.scoring.uncertainty import uncertainty_from_submetrics

# Proxy rate stats for estimated post-move metric delta (per-36)
_EST_RATE_COLS: tuple[str, ...] = ("pts", "usg", "ts", "ast")


@dataclass
class MovementBacktestResult:
    """Labeled movement rows with pre-move predictions and post-move outcomes."""

    season: str
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibrator: FitCalibrator | None = None

    @property
    def n_movements(self) -> int:
        return len(self.rows)

    @property
    def mean_destination_percentile(self) -> float | None:
        """Mean calibrated (or pre-move) fit percentile at destination teams."""
        if self.rows.empty:
            return None
        if "calibrated_fit_percentile" in self.rows.columns:
            return float(self.rows["calibrated_fit_percentile"].mean())
        if "pre_move_fit_percentile" in self.rows.columns:
            valid = self.rows["pre_move_fit_percentile"].dropna()
            return float(valid.mean()) if not valid.empty else None
        return None

    def calibration_frame(self) -> pd.DataFrame:
        """Rows with predicted percentile and realized outcome for reliability plots."""
        if self.rows.empty:
            return pd.DataFrame(
                columns=["calibrated_fit_percentile", "post_move_outcome", "movement_bucket"]
            )
        df = self.rows.copy()
        if "calibrated_fit_percentile" not in df.columns:
            df["calibrated_fit_percentile"] = df.get("pre_move_fit_percentile")
        df["movement_bucket"] = pd.qcut(
            df["calibrated_fit_percentile"].rank(method="first"),
            q=min(5, len(df)),
            duplicates="drop",
            labels=False,
        )
        return df[
            ["calibrated_fit_percentile", "post_move_outcome", "movement_bucket", "player_id"]
        ]


def _player_minutes_proxy(player: Any) -> float:
    feats = getattr(player, "features", {}) or {}
    for key in ("min", "minutes", "MIN"):
        if key in feats:
            return float(feats[key])
    return 800.0


def _estimate_metric_delta(pre_player: Any, post_rates: dict[str, float]) -> float:
    """
    Weighted change in per-36 proxies (PTS, USG, TS, AST) vs pre-move profile.

    Uses scaled feature dict on ``PlayerVector`` when present; else neutral 0.
    """
    pre = getattr(pre_player, "features", {}) or {}
    weights = {"pts": 0.35, "usg": 0.25, "ts": 0.25, "ast": 0.15}
    delta = 0.0
    for key, w in weights.items():
        pre_val = float(pre.get(key, pre.get(key.upper(), 0.5)))
        post_val = post_rates.get(key, pre_val)
        delta += w * (post_val - pre_val)
    return float(delta)


def _synthetic_post_rates(move: pd.Series, rng: np.random.Generator) -> dict[str, float]:
    """Demo post-move rates when game logs are unavailable."""
    seed = int(move["player_id"]) + int(move["to_team_id"])
    rng = np.random.default_rng(seed)
    return {
        "pts": float(np.clip(0.45 + rng.normal(0, 0.08), 0, 1)),
        "usg": float(np.clip(0.40 + rng.normal(0, 0.07), 0, 1)),
        "ts": float(np.clip(0.52 + rng.normal(0, 0.06), 0, 1)),
        "ast": float(np.clip(0.35 + rng.normal(0, 0.07), 0, 1)),
    }


def freeze_pre_move_features(
    context: SeasonFitContext,
    movements: pd.DataFrame,
    *,
    role_context: Any = None,
    impact_context: Any = None,
) -> pd.DataFrame:
    """
    Score each movement using only pre-move season context (no post-move team stats).

    Returns one row per movement with submetrics and ``raw_fit_score`` for the
    destination team at move time.
    """
    if movements.empty:
        return pd.DataFrame()

    table = build_fit_index_table(
        context,
        role_context=role_context,
        impact_context=impact_context,
    )
    rows: list[dict[str, Any]] = []

    for _, move in movements.iterrows():
        player_id = int(move["player_id"])
        to_team = int(move["to_team_id"])
        from_team = int(move["from_team_id"])
        player = context.players.get(player_id)
        team = context.teams.get(to_team)
        if player is None or team is None:
            continue

        team_need = None
        if role_context is not None:
            team_need = role_context.team_needs.get(to_team)

        pair_row = table.pairs[
            (table.pairs["player_id"] == player_id)
            & (table.pairs["team_id"] == to_team)
        ]
        if not pair_row.empty:
            pr = pair_row.iloc[0]
            raw = float(pr["raw_fit_score"])
            pre_pct = float(pr["overall_fit_percentile"])
            sub = {name: float(pr[name]) for name in SUBMETRIC_NAMES if name in pr.index}
            ensemble = {
                name: float(pr[f"ensemble_{name}"])
                for name in ENSEMBLE_DERIVED_NAMES
                if f"ensemble_{name}" in pr.index
            }
        else:
            sub = compute_all_submetrics(
                player,
                team,
                team_need=team_need,
                embeddings=role_context.embeddings if role_context else None,
                archetypes=role_context.archetypes if role_context else None,
                impact_context=impact_context,
            )
            ensemble = extract_ensemble_components(sub, player=player)
            raw = raw_ensemble_score(ensemble)
            pre_pct = None

        rows.append(
            {
                "player_id": player_id,
                "from_team_id": from_team,
                "to_team_id": to_team,
                "move_date": move["move_date"],
                "season": move["season"],
                "movement_type": move.get("movement_type", "team_change"),
                "pre_move_minutes": _player_minutes_proxy(player),
                "raw_fit_score": raw,
                "pre_move_fit_percentile": pre_pct,
                **{name: sub[name] for name in SUBMETRIC_NAMES},
                **{f"ensemble_{k}": v for k, v in ensemble.items()},
            }
        )

    return pd.DataFrame(rows)


def label_post_move_outcomes(
    frozen: pd.DataFrame,
    movements: pd.DataFrame,
    context: SeasonFitContext,
    *,
    post_minutes_scale: float = 1.05,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Attach post-move labels: minutes earned proxy and estimated metric delta.

    ``post_move_minutes`` — pre-move minutes × (0.85 + 0.3 × raw_fit_score) capped.
    ``estimated_metric_delta`` — change in weighted per-36 proxies vs pre profile.
    ``post_move_outcome`` — unit interval success score for calibration
        (0.6 × minutes_norm + 0.4 × sigmoid(metric_delta)).
    """
    if frozen.empty:
        return frozen

    rng = rng or np.random.default_rng(7)
    out = frozen.copy()
    post_minutes: list[float] = []
    metric_deltas: list[float] = []
    outcomes: list[float] = []

    move_lookup = movements.set_index(["player_id", "to_team_id"], drop=False)

    for _, row in out.iterrows():
        key = (int(row["player_id"]), int(row["to_team_id"]))
        move = move_lookup.loc[key] if key in move_lookup.index else None
        if isinstance(move, pd.DataFrame):
            move = move.iloc[0]

        player = context.players.get(int(row["player_id"]))
        pre_min = float(row["pre_move_minutes"])
        fit_signal = float(row["raw_fit_score"])
        minutes = pre_min * (0.85 + 0.3 * fit_signal) * post_minutes_scale
        minutes = float(np.clip(minutes, 0.0, 3500.0))

        post_rates = _synthetic_post_rates(move if move is not None else row, rng)
        delta = _estimate_metric_delta(player, post_rates) if player else 0.0

        minutes_norm = np.clip(minutes / 2000.0, 0.0, 1.0)
        outcome = 0.6 * minutes_norm + 0.4 * (1.0 / (1.0 + np.exp(-5.0 * delta)))

        post_minutes.append(minutes)
        metric_deltas.append(delta)
        outcomes.append(float(outcome))

    out["post_move_minutes"] = post_minutes
    out["estimated_metric_delta"] = metric_deltas
    out["post_move_outcome"] = outcomes
    return out


def run_movement_backtest(
    context_or_season: SeasonFitContext | str,
    movements: pd.DataFrame | None = None,
    *,
    synthetic: bool = False,
    n_moves: int = 3,
    n_players: int = 60,
    role_context: Any = None,
    impact_context: Any = None,
    calibrate: bool = True,
    calibration_method: str = "isotonic",
) -> MovementBacktestResult:
    """
    End-to-end movement backtest for a season context.

    Pass a :class:`SeasonFitContext` or a season label (e.g. ``"2024-25"``).
    When the first argument is a season string, builds a synthetic context unless
    *movements* is supplied; set ``synthetic=True`` to use demo movement rows.
    Loads movements via ``get_movements`` when *movements* is omitted.
    """
    if isinstance(context_or_season, str):
        season = context_or_season
        context = SeasonFitContext.from_synthetic(season, n_players=n_players)
        if movements is None:
            from nba_fit.data.fetchers.transactions import synthetic_movements

            movements = (
                synthetic_movements(season, n_moves=n_moves)
                if synthetic
                else get_movements(season, use_synthetic_fallback=True)
            )
        return run_movement_backtest(
            context,
            movements,
            role_context=role_context,
            impact_context=impact_context,
            calibrate=calibrate,
            calibration_method=calibration_method,
        )

    context = context_or_season
    season = context.season
    if movements is None:
        movements = get_movements(season, use_synthetic_fallback=True)

    if movements.empty:
        return MovementBacktestResult(season=season, rows=pd.DataFrame())

    mv = movements[list(MOVEMENT_COLUMNS)].copy()
    frozen = freeze_pre_move_features(
        context,
        mv,
        role_context=role_context,
        impact_context=impact_context,
    )
    labeled = label_post_move_outcomes(frozen, mv, context)

    calibrator: FitCalibrator | None = None
    if calibrate and not labeled.empty:
        calibrator = FitCalibrator(method=calibration_method)  # type: ignore[arg-type]
        labeled = calibrate_fit_table(
            labeled,
            raw_col="raw_fit_score",
            outcome_col="post_move_outcome",
            method=calibration_method,  # type: ignore[arg-type]
        )
        # Uncertainty per row from submetrics
        unc_low: list[float] = []
        unc_high: list[float] = []
        unc_disagree: list[float] = []
        unc_penalty: list[float] = []
        for _, r in labeled.iterrows():
            u = uncertainty_from_submetrics(r)
            unc_low.append(u.ci_low)
            unc_high.append(u.ci_high)
            unc_disagree.append(u.disagreement_std)
            unc_penalty.append(u.sample_penalty)
        labeled["uncertainty_ci_low"] = unc_low
        labeled["uncertainty_ci_high"] = unc_high
        labeled["uncertainty_disagreement"] = unc_disagree
        labeled["uncertainty_sample_penalty"] = unc_penalty

    return MovementBacktestResult(season=season, rows=labeled, calibrator=calibrator)
