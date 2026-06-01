"""Lineup net-rating regression from player embeddings and team context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge

from nba_fit.config.settings import get_settings
from nba_fit.features.constants import COL_PLAYER_ID, COL_TEAM_ID
from nba_fit.models.constants import (
    LINEUP_MODEL_ELASTICNET_ALPHA,
    LINEUP_MODEL_ELASTICNET_L1_RATIO,
    LINEUP_MODEL_PENALTY,
    LINEUP_MODEL_RIDGE_ALPHA,
    MODEL_RANDOM_STATE,
)
from nba_fit.models.rapm import stints_from_lineup_units
from nba_fit.models.role_embeddings import RoleEmbeddingArtifacts

LineupPenalty = Literal["ridge", "elasticnet"]


@dataclass
class LineupModelArtifacts:
    """Fitted lineup net-rating model for one season."""

    season: str
    penalty: LineupPenalty
    feature_names: tuple[str, ...]
    team_ids: np.ndarray
    embedding_dim: int
    regressor: Ridge | ElasticNet

    def predict_row(self, features: np.ndarray) -> float:
        return float(self.regressor.predict(features.reshape(1, -1))[0])


def _team_index(team_ids: np.ndarray) -> dict[int, int]:
    return {int(tid): i for i, tid in enumerate(team_ids)}


def _lineup_embedding_sum(
    player_ids: list[int],
    embeddings: RoleEmbeddingArtifacts,
) -> np.ndarray | None:
    dim = embeddings.n_components
    total = np.zeros(dim, dtype=float)
    count = 0
    for pid in player_ids:
        emb = embeddings.embedding_for(int(pid))
        if emb is None:
            continue
        total += emb[:dim]
        count += 1
    if count == 0:
        return None
    return total / count


def build_lineup_features(
    stints: pd.DataFrame,
    embeddings: RoleEmbeddingArtifacts,
    *,
    team_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Feature matrix: mean embedding of on-court players + team one-hot.

    Returns ``(X, y, sample_weight)``.
    """
    if stints.empty:
        return (
            np.zeros((0, 0), dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    team_col_vals = stints["team_id"].astype(int).to_numpy()
    if team_ids is None:
        team_ids = np.unique(team_col_vals)

    team_idx = _team_index(team_ids)
    dim = embeddings.n_components
    n_teams = len(team_ids)
    feat_dim = dim + n_teams
    rows: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []

    for _, row in stints.iterrows():
        players = list(row["player_ids"])
        emb_mean = _lineup_embedding_sum(players, embeddings)
        if emb_mean is None:
            continue
        vec = np.zeros(feat_dim, dtype=float)
        vec[:dim] = emb_mean
        tid = int(row["team_id"])
        tcol = team_idx.get(tid)
        if tcol is not None:
            vec[dim + tcol] = 1.0
        rows.append(vec)
        net = row.get("net_rating", np.nan)
        if pd.isna(net):
            net = float(row.get("off_rating", 0.0)) - float(row.get("def_rating", 0.0))
        targets.append(float(net))
        weights.append(float(row.get("stint_weight", row.get("minutes", 1.0))))

    if not rows:
        return (
            np.zeros((0, feat_dim), dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    return np.vstack(rows), np.asarray(targets, dtype=float), np.asarray(weights, dtype=float)


def _build_regressor(penalty: LineupPenalty) -> Ridge | ElasticNet:
    if penalty == "elasticnet":
        return ElasticNet(
            alpha=LINEUP_MODEL_ELASTICNET_ALPHA,
            l1_ratio=LINEUP_MODEL_ELASTICNET_L1_RATIO,
            random_state=MODEL_RANDOM_STATE,
            max_iter=5000,
        )
    return Ridge(alpha=LINEUP_MODEL_RIDGE_ALPHA, random_state=MODEL_RANDOM_STATE)


def fit_lineup_model(
    lineups: pd.DataFrame,
    embeddings: RoleEmbeddingArtifacts,
    *,
    season: str,
    penalty: LineupPenalty | None = None,
) -> LineupModelArtifacts:
    """Train lineup net-rating model on embedding sums + team context."""
    penalty = penalty or LINEUP_MODEL_PENALTY  # type: ignore[assignment]
    stints = stints_from_lineup_units(lineups)
    if stints.empty:
        raise ValueError("No lineup stints available for lineup model training")

    team_ids = np.unique(stints["team_id"].astype(int).to_numpy())
    x, y, w = build_lineup_features(stints, embeddings, team_ids=team_ids)
    if x.shape[0] < 5:
        raise ValueError(f"Need at least 5 lineup rows with embeddings, got {x.shape[0]}")

    regressor = _build_regressor(penalty)
    regressor.fit(x, y, sample_weight=w)

    dim = embeddings.n_components
    feat_names = tuple(f"emb_{i}" for i in range(dim)) + tuple(
        f"team_{int(t)}" for t in team_ids
    )

    return LineupModelArtifacts(
        season=season,
        penalty=penalty,
        feature_names=feat_names,
        team_ids=team_ids,
        embedding_dim=dim,
        regressor=regressor,
    )


def lineup_feature_vector(
    player_ids: list[int],
    team_id: int,
    embeddings: RoleEmbeddingArtifacts,
    artifacts: LineupModelArtifacts,
) -> np.ndarray | None:
    """Build one feature row for a hypothetical five-man (or partial) unit."""
    emb_mean = _lineup_embedding_sum(player_ids, embeddings)
    if emb_mean is None:
        return None
    dim = artifacts.embedding_dim
    n_teams = len(artifacts.team_ids)
    vec = np.zeros(dim + n_teams, dtype=float)
    vec[:dim] = emb_mean
    team_idx = _team_index(artifacts.team_ids)
    col = team_idx.get(int(team_id))
    if col is not None:
        vec[dim + col] = 1.0
    return vec


def predict_lineup_net_rating(
    player_ids: list[int],
    team_id: int,
    embeddings: RoleEmbeddingArtifacts,
    artifacts: LineupModelArtifacts,
) -> float | None:
    """Predict net rating per 100 for a player group on a team."""
    vec = lineup_feature_vector(player_ids, team_id, embeddings, artifacts)
    if vec is None:
        return None
    return artifacts.predict_row(vec)


def predict_with_player_swap(
    lineup_player_ids: list[int],
    team_id: int,
    *,
    out_player_id: int,
    in_player_id: int,
    embeddings: RoleEmbeddingArtifacts,
    artifacts: LineupModelArtifacts,
) -> float | None:
    """Predict net rating after swapping one player in a lineup."""
    swapped = [in_player_id if pid == out_player_id else pid for pid in lineup_player_ids]
    if out_player_id not in lineup_player_ids:
        swapped = lineup_player_ids + [in_player_id]
    return predict_lineup_net_rating(swapped, team_id, embeddings, artifacts)


def lineup_model_dir(season: str, *, root: Path | None = None) -> Path:
    settings = get_settings()
    base = root or settings.models_dir
    return base / "lineup_model" / f"season={season}"


def save_lineup_model(artifacts: LineupModelArtifacts, path: Path | None = None) -> Path:
    out_dir = path or lineup_model_dir(artifacts.season)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.regressor, out_dir / "regressor.joblib")
    meta = {
        "season": artifacts.season,
        "penalty": artifacts.penalty,
        "feature_names": list(artifacts.feature_names),
        "team_ids": artifacts.team_ids.tolist(),
        "embedding_dim": artifacts.embedding_dim,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def load_lineup_model(season: str, *, root: Path | None = None) -> LineupModelArtifacts:
    in_dir = lineup_model_dir(season, root=root)
    meta = json.loads((in_dir / "metadata.json").read_text(encoding="utf-8"))
    regressor = joblib.load(in_dir / "regressor.joblib")
    return LineupModelArtifacts(
        season=meta["season"],
        penalty=meta["penalty"],
        feature_names=tuple(meta["feature_names"]),
        team_ids=np.asarray(meta["team_ids"], dtype=int),
        embedding_dim=int(meta["embedding_dim"]),
        regressor=regressor,
    )


def synthetic_lineup_table(
    player_ids: list[int],
    *,
    team_id: int = 1610612747,
    n_lineups: int = 40,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build a minimal lineup_units-shaped table for offline tests."""
    rng = rng or np.random.default_rng(MODEL_RANDOM_STATE)
    rows = []
    for i in range(n_lineups):
        k = int(rng.integers(3, min(6, len(player_ids) + 1)))
        group = sorted(rng.choice(player_ids, size=k, replace=False).tolist())
        gid = "-".join(str(p) for p in group)
        rows.append(
            {
                "GROUP_ID": gid,
                "TEAM_ID": team_id,
                "MIN": float(rng.uniform(20, 120)),
                "NET_RATING": float(rng.normal(0, 8)),
                "OFF_RATING": float(rng.normal(110, 6)),
                "DEF_RATING": float(rng.normal(110, 6)),
            }
        )
    return pd.DataFrame(rows)
