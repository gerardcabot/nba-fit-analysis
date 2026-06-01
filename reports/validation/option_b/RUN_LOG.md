# Option B validation run log

- **Started (UTC):** 2026-05-26T21:47:57Z
- **Season:** 2024-25

## Commands

1. `python -m nba_fit ingest --tier mvp --season 2024-25`
2. `python -m nba_fit ingest --tier role --season 2024-25`
3. `python -m nba_fit train-roles --season 2024-25` (live interim)
4. `python -m nba_fit archetype-board 1610612747 --top 25`
5. `python -m nba_fit rank-player 2544 --fit-card-team 1610612747`
6. `pytest tests/test_archetypes.py -q`
7. `visual_tests/06_onoff_minutes.py`, `07_archetype_map.py`, `08_team_need_radar.py`

## Ingest (MVP)

[2026-05-26T21:47:59Z] Ingest tier=mvp done in 2.0s (endpoints=['leaguedashplayerstats', 'leaguedashteamstats', 'playerestimatedmetrics', 'teamestimatedmetrics', 'leaguedashplayerbiostats', 'leaguedashplayershotlocations', 'leaguedashteamshotlocations'])

## Ingest (role — 30 teams × team-scoped endpoints)

[2026-05-26T21:47:59Z] Role tier: 4 endpoints, ~30 teams each for 3 team-scoped pulls (cache speeds repeat runs).
[2026-05-26T21:47:59Z] Role interim tables already present; skipping role-tier re-ingest.
- **Status:** pass

## train-roles (live interim)

[2026-05-26T21:47:59Z] Training role embeddings + archetypes from interim tables...
[2026-05-26T21:48:09Z] train-roles done in 10.3s — 569 players, 6 archetype labels
- **Players embedded:** 569
- **Archetype labels (unique):** 6

## archetype-board

- **Rows:** 25

## rank-player + fit card

- **Destination rows:** 30
- **Fit card archetype:** high_usage_creator (comps=5)

## Pytest

```
..........                                                               [100%]
============================== warnings summary ===============================
tests/test_archetypes.py::test_role_embedding_shape
tests/test_archetypes.py::test_archetype_labels_in_vocab
tests/test_archetypes.py::test_team_need_profiles_cover_teams
tests/test_archetypes.py::test_save_load_role_embeddings
tests/test_archetypes.py::test_team_need_fit_in_unit_interval
tests/test_archetypes.py::test_submetrics_include_team_need_fit
tests/test_archetypes.py::test_nearest_comps_exclude_self
tests/test_archetypes.py::test_gmm_archetype_count
  C:\Users\geric\AppData\Roaming\Python\Python310\site-packages\sklearn\decomposition\_truncated_svd.py:273: RuntimeWarning: invalid value encountered in divide
    self.explained_variance_ratio_ = exp_var / full_var

tests/test_archetypes.py::test_role_embedding_shape
tests/test_archetypes.py::test_archetype_labels_in_vocab
tests/test_archetypes.py::test_team_need_profiles_cover_teams
tests/test_archetypes.py::test_save_load_role_embeddings
tests/test_archetypes.py::test_team_need_fit_in_unit_interval
tests/test_archetypes.py::test_submetrics_include_team_need_fit
tests/test_archetypes.py::test_nearest_comps_exclude_self
tests/test_archetypes.py::test_gmm_archetype_count
  C:\Users\geric\AppData\Roaming\Python\Python310\site-packages\sklearn\base.py:1365: ConvergenceWarning: Number of distinct clusters (1) found smaller than n_clusters (8). Possibly due to duplicate points in X.
    return fit_method(estimator, *args, **kwargs)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
10 passed, 16 warnings in 59.98s
```
- **Exit code:** 0

## Visual tests

[2026-05-26T21:55:49Z] Visual test 06_onoff_minutes...
[2026-05-26T21:55:54Z]   06_onoff_minutes exit=0 figures=1
[2026-05-26T21:55:54Z] Visual test 07_archetype_map...
[2026-05-26T21:56:02Z]   07_archetype_map exit=0 figures=1
[2026-05-26T21:56:02Z] Visual test 08_team_need_radar...
[2026-05-26T21:56:08Z]   08_team_need_radar exit=0 figures=1

- **Finished (UTC):** 2026-05-26T21:56:08Z