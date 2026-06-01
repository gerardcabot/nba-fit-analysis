"""Tests for ensemble weight artifact save/load and scoring resolution."""

from __future__ import annotations

import pytest

from nba_fit.models.weight_learning import (
    WeightLearningResult,
    ensemble_weights_path,
    load_ensemble_weights,
    save_ensemble_weights,
)
from nba_fit.scoring.constants import ENSEMBLE_COMPONENT_NAMES, ENSEMBLE_COMPONENT_WEIGHTS
from nba_fit.scoring.ensemble import (
    clear_ensemble_weights_cache,
    raw_ensemble_score,
    resolve_ensemble_component_weights,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_ensemble_weights_cache()


def test_save_and_load_ensemble_weights(tmp_path) -> None:
    weights = {name: 0.2 for name in ENSEMBLE_COMPONENT_NAMES}
    result = WeightLearningResult(
        weights=weights,
        prior_weights=dict(ENSEMBLE_COMPONENT_WEIGHTS),
        l2=0.01,
        train_mse=0.05,
        n_samples=10,
    )
    season = "2024-25"
    path = save_ensemble_weights(result, season, path=ensemble_weights_path(season, root=tmp_path))
    assert path.is_file()
    loaded = load_ensemble_weights(season, root=tmp_path)
    assert loaded is not None
    assert loaded == weights


def test_resolve_ensemble_weights_falls_back_to_prior() -> None:
    resolved = resolve_ensemble_component_weights("2099-00", weights=None)
    assert resolved == ENSEMBLE_COMPONENT_WEIGHTS


def test_resolve_ensemble_weights_uses_learned_artifact(tmp_path) -> None:
    season = "2024-25"
    learned = {name: (0.1 if i == 0 else 0.225) for i, name in enumerate(ENSEMBLE_COMPONENT_NAMES)}
    save_ensemble_weights(
        WeightLearningResult(
            weights=learned,
            prior_weights=dict(ENSEMBLE_COMPONENT_WEIGHTS),
            l2=0.01,
            train_mse=0.01,
            n_samples=5,
        ),
        season,
        path=ensemble_weights_path(season, root=tmp_path),
    )
    from nba_fit.scoring import ensemble as ens_mod

    ens_mod._WEIGHTS_CACHE.clear()
    with pytest.MonkeyPatch.context() as mp:
        from nba_fit.models import weight_learning as wl

        mp.setattr(
            wl,
            "load_ensemble_weights",
            lambda s, root=None: load_ensemble_weights(s, root=tmp_path),
        )
        resolved = resolve_ensemble_component_weights(season)
    assert resolved == learned


def test_raw_ensemble_score_uses_explicit_weights() -> None:
    components = {name: 0.8 for name in ENSEMBLE_COMPONENT_NAMES}
    components["risk_penalty"] = 0.0
    custom = {name: 1.0 / len(ENSEMBLE_COMPONENT_NAMES) for name in ENSEMBLE_COMPONENT_NAMES}
    score = raw_ensemble_score(components, weights=custom)
    assert score == pytest.approx(0.8, abs=1e-6)
