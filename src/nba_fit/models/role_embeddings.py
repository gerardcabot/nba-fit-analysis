"""PCA / truncated SVD role embeddings on scaled player feature matrices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD

from nba_fit.config.settings import get_settings
from nba_fit.features.constants import COL_PLAYER_ID, PLAYER_FEATURE_PREFIX
from nba_fit.models.constants import (
    MODEL_RANDOM_STATE,
    ROLE_EMBEDDING_METHOD,
    ROLE_EMBEDDING_MIN_PLAYERS,
    ROLE_EMBEDDING_N_COMPONENTS,
)

if TYPE_CHECKING:
    from nba_fit.models.archetypes import ArchetypeArtifacts

EmbeddingMethod = Literal["pca", "truncated_svd"]


@dataclass
class RoleEmbeddingArtifacts:
    """Fitted embedding model and per-player coordinates for one season."""

    season: str
    method: EmbeddingMethod
    n_components: int
    feature_names: tuple[str, ...]
    player_ids: np.ndarray
    embeddings: np.ndarray
    explained_variance_ratio: np.ndarray | None
    decomposer: PCA | TruncatedSVD

    def embedding_for(self, player_id: int) -> np.ndarray | None:
        hits = np.where(self.player_ids == player_id)[0]
        if len(hits) == 0:
            return None
        return self.embeddings[int(hits[0])]


def _player_feature_matrix(player_scaled: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    feat_cols = sorted(c for c in player_scaled.columns if c.startswith(f"z_{PLAYER_FEATURE_PREFIX}"))
    if not feat_cols:
        raise ValueError("player_scaled must contain z_pf_* columns")
    matrix = player_scaled[feat_cols].to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0)
    player_ids = player_scaled[COL_PLAYER_ID].to_numpy(dtype=int)
    return matrix, tuple(feat_cols), player_ids


def _build_decomposer(
    method: EmbeddingMethod,
    n_components: int,
) -> PCA | TruncatedSVD:
    if method == "pca":
        return PCA(n_components=n_components, random_state=MODEL_RANDOM_STATE)
    return TruncatedSVD(n_components=n_components, random_state=MODEL_RANDOM_STATE)


def fit_role_embeddings(
    player_scaled: pd.DataFrame,
    *,
    season: str,
    method: EmbeddingMethod | None = None,
    n_components: int | None = None,
) -> RoleEmbeddingArtifacts:
    """Fit season role embeddings from a scaled player feature matrix."""
    matrix, feat_names, player_ids = _player_feature_matrix(player_scaled)
    if len(matrix) < ROLE_EMBEDDING_MIN_PLAYERS:
        raise ValueError(
            f"Need at least {ROLE_EMBEDDING_MIN_PLAYERS} players for embeddings, got {len(matrix)}"
        )

    method = method or ROLE_EMBEDDING_METHOD  # type: ignore[assignment]
    n_components = n_components or ROLE_EMBEDDING_N_COMPONENTS
    n_components = min(n_components, matrix.shape[0] - 1, matrix.shape[1])

    decomposer = _build_decomposer(method, n_components)
    embeddings = decomposer.fit_transform(matrix)
    evr: np.ndarray | None = None
    if hasattr(decomposer, "explained_variance_ratio_"):
        evr = np.asarray(decomposer.explained_variance_ratio_, dtype=float)

    return RoleEmbeddingArtifacts(
        season=season,
        method=method,
        n_components=n_components,
        feature_names=feat_names,
        player_ids=player_ids,
        embeddings=np.asarray(embeddings, dtype=float),
        explained_variance_ratio=evr,
        decomposer=decomposer,
    )


def role_embedding_dir(season: str, *, root: Path | None = None) -> Path:
    settings = get_settings()
    base = root or settings.models_dir
    return base / "role_embedding" / f"season={season}"


def save_role_embeddings(artifacts: RoleEmbeddingArtifacts, path: Path | None = None) -> Path:
    """Persist embeddings and fitted decomposer under ``models/role_embedding/``."""
    out_dir = path or role_embedding_dir(artifacts.season)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "embeddings.npy", artifacts.embeddings)
    pd.DataFrame(
        {
            COL_PLAYER_ID: artifacts.player_ids,
            **{f"re_{i}": artifacts.embeddings[:, i] for i in range(artifacts.embeddings.shape[1])},
        }
    ).to_parquet(out_dir / "player_embeddings.parquet", index=False)

    joblib.dump(artifacts.decomposer, out_dir / "decomposer.joblib")

    meta = {
        "season": artifacts.season,
        "method": artifacts.method,
        "n_components": artifacts.n_components,
        "feature_names": list(artifacts.feature_names),
        "explained_variance_ratio": (
            artifacts.explained_variance_ratio.tolist()
            if artifacts.explained_variance_ratio is not None
            else None
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def load_role_embeddings(season: str, *, root: Path | None = None) -> RoleEmbeddingArtifacts:
    """Load season artifacts written by :func:`save_role_embeddings`."""
    in_dir = role_embedding_dir(season, root=root)
    meta = json.loads((in_dir / "metadata.json").read_text(encoding="utf-8"))
    decomposer = joblib.load(in_dir / "decomposer.joblib")
    table = pd.read_parquet(in_dir / "player_embeddings.parquet")
    re_cols = sorted(c for c in table.columns if c.startswith("re_"))
    embeddings = table[re_cols].to_numpy(dtype=float)
    evr = meta.get("explained_variance_ratio")
    return RoleEmbeddingArtifacts(
        season=meta["season"],
        method=meta["method"],
        n_components=int(meta["n_components"]),
        feature_names=tuple(meta["feature_names"]),
        player_ids=table[COL_PLAYER_ID].to_numpy(dtype=int),
        embeddings=embeddings,
        explained_variance_ratio=np.asarray(evr, dtype=float) if evr else None,
        decomposer=decomposer,
    )
