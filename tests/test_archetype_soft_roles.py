"""GMM posterior → industry soft role display."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from nba_fit.features.constants import COL_PLAYER_ID
from nba_fit.features.player_vector import build_player_features
from nba_fit.features.scaling import scale_player_features
from nba_fit.models.archetypes import fit_archetypes
from nba_fit.models.role_embeddings import RoleEmbeddingArtifacts
from nba_fit.models.role_taxonomy import soft_role_display


def test_gmm_soft_role_probs_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    n = 40
    matrix = rng.normal(size=(n, 6))
    player_ids = np.arange(1000, 1000 + n)
    pca = PCA(n_components=6, random_state=0)
    coords = pca.fit_transform(matrix)
    embeddings = RoleEmbeddingArtifacts(
        season="2025-26",
        method="pca",
        n_components=6,
        feature_names=tuple(f"z_pf_role_volume__stub_{i}" for i in range(matrix.shape[1])),
        player_ids=player_ids,
        embeddings=coords,
        explained_variance_ratio=pca.explained_variance_ratio_,
        decomposer=pca,
    )
    stub = pd.DataFrame(
        {
            COL_PLAYER_ID: player_ids,
            "MIN": [30.0] * n,
            "GP": [60] * n,
            "FGA": [15.0] * n,
            "FG3A": [5.0] * n,
            "PTS": [20.0] * n,
            "AST": [5.0] * n,
            "TOV": [2.0] * n,
            "USG_PCT": [0.25] * n,
            "PLAYER_HEIGHT_INCHES": [78.0] * n,
        }
    )
    scaled = scale_player_features(build_player_features(stub))
    artifacts = fit_archetypes(embeddings, scaled, clusterer="gmm", n_components=4)
    probs = artifacts.industry_probs_for(int(player_ids[0]))
    assert probs
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    text = artifacts.soft_role_display_for(int(player_ids[0]))
    assert text
    assert soft_role_display(probs) == text
