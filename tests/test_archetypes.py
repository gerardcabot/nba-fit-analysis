"""Synthetic tests for Option B role embeddings, archetypes, and team need."""

from __future__ import annotations

import numpy as np
import pytest

from nba_fit.features.constants import COL_PLAYER_ID
from nba_fit.features.season_context import DEMO_PLAYER_ID, DEMO_TEAM_ID, SeasonFitContext
from nba_fit.models.archetypes import (
    ARCHETYPE_LABEL_VOCAB,
    fit_archetypes,
    label_cluster_from_centroid,
)
from nba_fit.models.constants import (
    ARCHETYPE_N_COMPONENTS,
    LINEUP_IMPACT_WEIGHT,
    ROLE_EMBEDDING_N_COMPONENTS,
    ROLE_FIT_WEIGHT,
)
from nba_fit.models.role_context import RoleFitContext
from nba_fit.models.role_embeddings import fit_role_embeddings, save_role_embeddings
from nba_fit.scoring.constants import SUBMETRIC_NAMES, SUBMETRIC_WEIGHTS
from nba_fit.scoring.role_fit import team_need_fit
from nba_fit.scoring.submetrics import compute_all_submetrics


@pytest.fixture
def context() -> SeasonFitContext:
    return SeasonFitContext.from_synthetic("2025-26", n_players=50)


@pytest.fixture
def role_context(context: SeasonFitContext) -> RoleFitContext:
    return RoleFitContext.from_synthetic(context)


def test_role_embedding_shape(context: SeasonFitContext) -> None:
    role = RoleFitContext.from_synthetic(context)
    assert role.embeddings.embeddings.shape[0] == len(role.embeddings.player_ids)
    assert role.embeddings.embeddings.shape[1] <= ROLE_EMBEDDING_N_COMPONENTS


def test_archetype_labels_in_vocab(role_context: RoleFitContext) -> None:
    for label in role_context.archetypes.archetype_labels:
        assert str(label) in ARCHETYPE_LABEL_VOCAB


def test_archetype_industry_roles(role_context: RoleFitContext) -> None:
    from nba_fit.models.role_taxonomy import INDUSTRY_OFFENSIVE_ROLES, map_heuristic_to_industry

    for heuristic, industry in zip(
        role_context.archetypes.archetype_labels,
        role_context.archetypes.industry_roles,
        strict=False,
    ):
        assert str(industry) == map_heuristic_to_industry(str(heuristic))
        assert str(industry) in INDUSTRY_OFFENSIVE_ROLES


def test_team_need_profiles_cover_teams(role_context: RoleFitContext, context: SeasonFitContext) -> None:
    assert set(role_context.team_needs.keys()) == set(context.teams.keys())
    for profile in role_context.team_needs.values():
        assert profile.values.ndim == 1
        assert len(profile.values) >= 1


def test_save_load_role_embeddings(tmp_path, context: SeasonFitContext) -> None:
    role = RoleFitContext.from_synthetic(context)
    out = save_role_embeddings(role.embeddings, tmp_path / "role_embedding" / f"season={context.season}")
    assert (out / "metadata.json").is_file()
    assert (out / "player_embeddings.parquet").is_file()


def test_team_need_fit_in_unit_interval(role_context: RoleFitContext, context: SeasonFitContext) -> None:
    player = context.players[DEMO_PLAYER_ID]
    need = role_context.team_needs[DEMO_TEAM_ID]
    score = team_need_fit(
        player,
        need,
        embeddings=role_context.embeddings,
        archetypes=role_context.archetypes,
    )
    assert 0.0 <= score <= 1.0


def test_submetrics_include_team_need_fit(role_context: RoleFitContext, context: SeasonFitContext) -> None:
    player = context.players[DEMO_PLAYER_ID]
    team = context.teams[DEMO_TEAM_ID]
    sub = compute_all_submetrics(
        player,
        team,
        team_need=role_context.team_needs[DEMO_TEAM_ID],
        embeddings=role_context.embeddings,
        archetypes=role_context.archetypes,
    )
    assert "team_need_fit" in sub
    assert set(sub.keys()) == set(SUBMETRIC_NAMES)


def test_weights_include_role_fit_and_sum_to_one() -> None:
    assert "team_need_fit" in SUBMETRIC_WEIGHTS
    assert "lineup_impact_fit" in SUBMETRIC_WEIGHTS
    assert SUBMETRIC_WEIGHTS["team_need_fit"] == pytest.approx(ROLE_FIT_WEIGHT)
    assert SUBMETRIC_WEIGHTS["lineup_impact_fit"] == pytest.approx(LINEUP_IMPACT_WEIGHT)
    assert abs(sum(SUBMETRIC_WEIGHTS.values()) - 1.0) < 1e-9


def test_label_cluster_from_centroid_returns_vocab_member() -> None:
    import pandas as pd

    centroid = pd.Series(
        {
            "z_pf_role_volume__usg_pct": 1.5,
            "z_pf_creation__ast_pct": 1.0,
            "z_pf_scoring_style__fg3a_rate": -0.5,
            "z_pf_scoring_style__rim_freq": 0.0,
            "z_pf_defense__blk_pg": 0.0,
            "z_pf_defense__stl_pg": 0.0,
            "z_pf_rebounding__reb_pct": 0.0,
        }
    )
    label = label_cluster_from_centroid(centroid)
    assert label in ARCHETYPE_LABEL_VOCAB


def test_nearest_comps_exclude_self(role_context: RoleFitContext) -> None:
    from nba_fit.models.constants import NEAREST_NEIGHBOR_K

    comps = role_context.nearest_comps(DEMO_PLAYER_ID)
    assert len(comps) <= NEAREST_NEIGHBOR_K
    assert all(c["player_id"] != DEMO_PLAYER_ID for c in comps)


def test_gmm_archetype_count(context: SeasonFitContext) -> None:
    from nba_fit.features._synthetic import synthetic_player_df
    from nba_fit.features.player_vector import build_player_features
    from nba_fit.features.scaling import scale_player_features

    raw = synthetic_player_df(n_rows=50)
    raw[COL_PLAYER_ID] = range(1, len(raw) + 1)
    player_feats = build_player_features(raw)
    scaled = scale_player_features(player_feats)
    emb = fit_role_embeddings(scaled, season="2025-26")
    arch = fit_archetypes(emb, scaled, clusterer="gmm", n_components=ARCHETYPE_N_COMPONENTS)
    assert len(np.unique(arch.cluster_ids)) <= ARCHETYPE_N_COMPONENTS
