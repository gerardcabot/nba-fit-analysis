"""Player archetype clustering on role embeddings (GMM or HDBSCAN)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.mixture import GaussianMixture

from nba_fit.features.constants import COL_PLAYER_ID, PLAYER_FEATURE_PREFIX
from nba_fit.features.player_vector import (
    FEATURE_GROUP_CREATION,
    FEATURE_GROUP_DEFENSE,
    FEATURE_GROUP_REBOUNDING,
    FEATURE_GROUP_ROLE_VOLUME,
    FEATURE_GROUP_SCORING_STYLE,
)
from nba_fit.models.constants import (
    ARCHETYPE_CLUSTERER,
    ARCHETYPE_N_COMPONENTS,
    ARCHETYPE_NOISE_LABEL,
    ARCHETYPE_THRESH_BLK,
    ARCHETYPE_THRESH_CONNECTOR_AST,
    ARCHETYPE_THRESH_CONNECTOR_USG_CAP,
    ARCHETYPE_THRESH_DEF_GUARD_USG,
    ARCHETYPE_THRESH_HIGH_AST,
    ARCHETYPE_THRESH_HIGH_USG,
    ARCHETYPE_THRESH_LOW_USG,
    ARCHETYPE_THRESH_MED_AST,
    ARCHETYPE_THRESH_MED_USG,
    ARCHETYPE_THRESH_REB,
    ARCHETYPE_THRESH_RIM_FREQ,
    ARCHETYPE_THRESH_SHOOTER_FG3,
    ARCHETYPE_THRESH_SPACER_FG3,
    ARCHETYPE_THRESH_STL,
    ARCHETYPE_THRESH_STRETCH_FG3,
    ARCHETYPE_THRESH_STRETCH_REB,
    ARCHETYPE_THRESH_VERY_LOW_USG,
    HDBSCAN_MIN_CLUSTER_SIZE,
    MODEL_RANDOM_STATE,
)
from nba_fit.models.role_embeddings import RoleEmbeddingArtifacts, role_embedding_dir
from nba_fit.models.role_taxonomy import (
    industry_roles_for_labels,
    map_heuristic_to_industry,
    soft_role_display,
)

# Heuristic archetype vocabulary (basketball-facing labels for cluster centroids).
ARCHETYPE_LABEL_VOCAB: tuple[str, ...] = (
    "high_usage_creator",
    "advantage_creator",
    "movement_shooter",
    "low_usage_spacer",
    "connector_wing",
    "rim_running_big",
    "stretch_big",
    "defensive_guard",
    "rim_protector",
    "rebound_finisher",
    "versatile_forward",
    ARCHETYPE_NOISE_LABEL,
)


def _aggregate_gmm_industry_probs(
    proba_row: np.ndarray,
    component_labels: list[str],
) -> dict[str, float]:
    """Map GMM component posteriors to industry-role probability mass."""
    out: dict[str, float] = {}
    for weight, label in zip(proba_row, component_labels, strict=False):
        industry = map_heuristic_to_industry(str(label))
        out[industry] = out.get(industry, 0.0) + float(weight)
    return out


@dataclass
class ArchetypeArtifacts:
    """Cluster assignments and human-readable labels for one season."""

    season: str
    clusterer_name: str
    n_components: int
    player_ids: np.ndarray
    cluster_ids: np.ndarray
    archetype_labels: np.ndarray
    industry_roles: np.ndarray
    centroids: np.ndarray
    clusterer: GaussianMixture | HDBSCAN
    industry_role_probs: dict[int, dict[str, float]] | None = None

    def label_for(self, player_id: int) -> str | None:
        hits = np.where(self.player_ids == player_id)[0]
        if len(hits) == 0:
            return None
        return str(self.archetype_labels[int(hits[0])])

    def industry_role_for(self, player_id: int) -> str | None:
        hits = np.where(self.player_ids == player_id)[0]
        if len(hits) == 0:
            return None
        return str(self.industry_roles[int(hits[0])])

    def cluster_for(self, player_id: int) -> int | None:
        hits = np.where(self.player_ids == player_id)[0]
        if len(hits) == 0:
            return None
        return int(self.cluster_ids[int(hits[0])])

    def industry_probs_for(self, player_id: int) -> dict[str, float]:
        """Soft industry-role probabilities (GMM posteriors when available)."""
        if self.industry_role_probs is not None:
            probs = self.industry_role_probs.get(int(player_id))
            if probs is not None:
                return dict(probs)
        label = self.industry_role_for(player_id)
        if label is None:
            return {}
        return {label: 1.0}

    def soft_role_display_for(self, player_id: int, *, top_n: int = 3) -> str:
        """Formatted top-k industry roles for fit cards."""
        return soft_role_display(self.industry_probs_for(player_id), top_n=top_n)


def _feature_centroids(
    player_scaled: pd.DataFrame,
    cluster_ids: np.ndarray,
) -> dict[int, pd.Series]:
    """Mean scaled features per cluster for heuristic naming (row-aligned with embeddings)."""
    feat_cols = [c for c in player_scaled.columns if c.startswith(f"z_{PLAYER_FEATURE_PREFIX}")]
    work = player_scaled[feat_cols].iloc[: len(cluster_ids)].copy()
    work["_cluster"] = cluster_ids
    centroids: dict[int, pd.Series] = {}
    for cid, group in work.groupby("_cluster"):
        centroids[int(cid)] = group.drop(columns="_cluster").mean()
    return centroids


def _stem_centroid(centroid: pd.Series, group: str, stem: str) -> float:
    prefix = f"z_{PLAYER_FEATURE_PREFIX}{group}__{stem}"
    matches = [c for c in centroid.index if c.startswith(prefix)]
    if not matches:
        return 0.0
    return float(centroid[matches[0]])


def label_cluster_from_centroid(centroid: pd.Series) -> str:
    """Map a cluster feature centroid to a basketball-facing archetype name."""
    usg = _stem_centroid(centroid, FEATURE_GROUP_ROLE_VOLUME, "usg_pct")
    rim = _stem_centroid(centroid, FEATURE_GROUP_SCORING_STYLE, "rim_freq")
    fg3 = _stem_centroid(centroid, FEATURE_GROUP_SCORING_STYLE, "fg3a_rate")
    ast = _stem_centroid(centroid, FEATURE_GROUP_CREATION, "ast_pct")
    blk = _stem_centroid(centroid, FEATURE_GROUP_DEFENSE, "blk_pg")
    stl = _stem_centroid(centroid, FEATURE_GROUP_DEFENSE, "stl_pg")
    reb = _stem_centroid(centroid, FEATURE_GROUP_REBOUNDING, "reb_pct")

    if usg >= ARCHETYPE_THRESH_HIGH_USG and ast >= ARCHETYPE_THRESH_HIGH_AST:
        return "high_usage_creator"
    if usg >= ARCHETYPE_THRESH_MED_USG and ast >= ARCHETYPE_THRESH_MED_AST:
        return "advantage_creator"
    if fg3 >= ARCHETYPE_THRESH_SHOOTER_FG3 and usg < ARCHETYPE_THRESH_LOW_USG:
        return "movement_shooter"
    if fg3 >= ARCHETYPE_THRESH_SPACER_FG3 and usg < ARCHETYPE_THRESH_VERY_LOW_USG:
        return "low_usage_spacer"
    if ast >= ARCHETYPE_THRESH_CONNECTOR_AST and usg < ARCHETYPE_THRESH_CONNECTOR_USG_CAP:
        return "connector_wing"
    if blk >= ARCHETYPE_THRESH_BLK and rim >= ARCHETYPE_THRESH_LOW_USG:
        return "rim_protector"
    if rim >= ARCHETYPE_THRESH_RIM_FREQ and reb >= ARCHETYPE_THRESH_LOW_USG:
        return "rim_running_big"
    if fg3 >= ARCHETYPE_THRESH_STRETCH_FG3 and reb >= ARCHETYPE_THRESH_STRETCH_REB:
        return "stretch_big"
    if stl >= ARCHETYPE_THRESH_STL and usg < ARCHETYPE_THRESH_DEF_GUARD_USG:
        return "defensive_guard"
    if reb >= ARCHETYPE_THRESH_REB:
        return "rebound_finisher"
    return "versatile_forward"


def fit_archetypes(
    embeddings: RoleEmbeddingArtifacts,
    player_scaled: pd.DataFrame,
    *,
    clusterer: str | None = None,
    n_components: int | None = None,
) -> ArchetypeArtifacts:
    """Cluster embedding space and assign heuristic basketball labels.

    **Audit requirement:** ``n_components`` (GMM) or HDBSCAN density should be
    chosen with an elbow / BIC sweep on embeddings **and** a manual basketball
    review of cluster centroids — statistical clusters may not match real roles.
    """
    clusterer_name = clusterer or ARCHETYPE_CLUSTERER
    n_components = n_components or ARCHETYPE_N_COMPONENTS
    matrix = embeddings.embeddings
    player_ids = embeddings.player_ids

    if clusterer_name == "hdbscan":
        model: GaussianMixture | HDBSCAN = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE)
        cluster_ids = model.fit_predict(matrix)
        n_fit = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    else:
        n_fit = min(n_components, len(matrix) - 1)
        model = GaussianMixture(
            n_components=n_fit,
            random_state=MODEL_RANDOM_STATE,
            reg_covar=1e-4,
        )
        cluster_ids = model.fit_predict(matrix)

    feature_centroids = _feature_centroids(player_scaled, cluster_ids)
    labels: list[str] = []
    unique_ids = sorted(set(int(c) for c in cluster_ids))
    centroid_matrix = np.zeros((len(unique_ids), matrix.shape[1]), dtype=float)
    for i, cid in enumerate(unique_ids):
        mask = cluster_ids == cid
        centroid_matrix[i] = matrix[mask].mean(axis=0)
        if cid == -1:
            labels.append(ARCHETYPE_NOISE_LABEL)
        else:
            labels.append(label_cluster_from_centroid(feature_centroids[cid]))

    id_to_label = dict(zip(unique_ids, labels, strict=False))
    archetype_labels = np.array([id_to_label[int(c)] for c in cluster_ids], dtype=object)
    industry_roles = np.array(industry_roles_for_labels(archetype_labels.tolist()), dtype=object)

    industry_role_probs: dict[int, dict[str, float]] | None = None
    if isinstance(model, GaussianMixture) and hasattr(model, "predict_proba"):
        component_labels = [id_to_label.get(i, ARCHETYPE_NOISE_LABEL) for i in range(n_fit)]
        proba = model.predict_proba(matrix)
        industry_role_probs = {}
        for idx, pid in enumerate(player_ids):
            industry_role_probs[int(pid)] = _aggregate_gmm_industry_probs(
                proba[idx], component_labels
            )

    return ArchetypeArtifacts(
        season=embeddings.season,
        clusterer_name=clusterer_name,
        n_components=n_fit if clusterer_name == "gmm" else int(len(unique_ids)),
        player_ids=player_ids,
        cluster_ids=cluster_ids.astype(int),
        archetype_labels=archetype_labels,
        industry_roles=industry_roles,
        centroids=centroid_matrix,
        clusterer=model,
        industry_role_probs=industry_role_probs,
    )


def archetype_dir(season: str, *, root: Path | None = None) -> Path:
    return role_embedding_dir(season, root=root) / "archetypes"


def save_archetypes(artifacts: ArchetypeArtifacts, path: Path | None = None) -> Path:
    out_dir = path or archetype_dir(artifacts.season)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            COL_PLAYER_ID: artifacts.player_ids,
            "cluster_id": artifacts.cluster_ids,
            "archetype_label": artifacts.archetype_labels,
            "industry_role": artifacts.industry_roles,
        }
    )
    if artifacts.industry_role_probs:
        table["industry_role_probs_json"] = [
            json.dumps(artifacts.industry_role_probs.get(int(pid), {}))
            for pid in artifacts.player_ids
        ]
    table.to_parquet(out_dir / "player_archetypes.parquet", index=False)
    np.save(out_dir / "centroids.npy", artifacts.centroids)
    joblib.dump(artifacts.clusterer, out_dir / "clusterer.joblib")
    meta = {
        "season": artifacts.season,
        "clusterer_name": artifacts.clusterer_name,
        "n_components": artifacts.n_components,
        "label_vocab": list(ARCHETYPE_LABEL_VOCAB),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def load_archetypes(season: str, *, root: Path | None = None) -> ArchetypeArtifacts:
    in_dir = archetype_dir(season, root=root)
    meta = json.loads((in_dir / "metadata.json").read_text(encoding="utf-8"))
    table = pd.read_parquet(in_dir / "player_archetypes.parquet")
    clusterer = joblib.load(in_dir / "clusterer.joblib")
    centroids = np.load(in_dir / "centroids.npy")
    if "industry_role" in table.columns:
        industry_roles = table["industry_role"].to_numpy(dtype=object)
    else:
        industry_roles = np.array(
            [map_heuristic_to_industry(str(l)) for l in table["archetype_label"].to_numpy(dtype=object)],
            dtype=object,
        )
    industry_role_probs: dict[int, dict[str, float]] | None = None
    if "industry_role_probs_json" in table.columns:
        industry_role_probs = {
            int(row[COL_PLAYER_ID]): json.loads(row["industry_role_probs_json"] or "{}")
            for _, row in table.iterrows()
        }
    return ArchetypeArtifacts(
        season=meta["season"],
        clusterer_name=meta["clusterer_name"],
        n_components=int(meta["n_components"]),
        player_ids=table[COL_PLAYER_ID].to_numpy(dtype=int),
        cluster_ids=table["cluster_id"].to_numpy(dtype=int),
        archetype_labels=table["archetype_label"].to_numpy(dtype=object),
        industry_roles=industry_roles,
        centroids=centroids,
        clusterer=clusterer,
        industry_role_probs=industry_role_probs,
    )


def archetype_label_counts(artifacts: ArchetypeArtifacts) -> Mapping[str, int]:
    labels, counts = np.unique(artifacts.archetype_labels, return_counts=True)
    return {str(l): int(c) for l, c in zip(labels, counts, strict=False)}
