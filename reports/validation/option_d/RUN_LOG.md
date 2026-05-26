# Option D validation run log

- **Started (UTC):** 2026-05-26T22:04:34Z
- **Season:** 2024-25
- **Player / team:** 2544 / 1610612747

## Pipeline

1. `ingest --tier mvp|role|impact --season 2024-25` (impact `--max-games 30`)
2. `train-roles --season 2024-25`
3. `train-impact --season 2024-25`
4. `backtest-movement --season 2024-25 --no-synthetic --moves 15`
5. `rank-player 2544 --fit-card-team 1610612747 --season 2024-25`
6. `pytest` (Option D subset — see metrics.json)
7. `visual_tests/12_calibration_curve.py`, `13_ensemble_weights.py`, `14_dashboard_data_health.py`

## Ingest

- **Status:** skipped (interim / possessions cached)

## Train

- **train-roles:** skipped (artifacts on disk; loaded via `from_season`)
- **train-impact:** skipped (loaded; 549 rapm players)

## Backtest

- **Movements:** sparse real labels (0 < 5); used `synthetic_movements(n=15)`
- **Scoring subset:** 80 players (cap=80, validation runtime)
- **Status:** pass (15 scored)

## Rank / fit card

- **Context:** interim
- **overall_fit_percentile:** 41.666666666666664

## Pytest

- **Subset:** tests/test_ensemble.py, tests/test_fit_index.py, tests/test_movement_backtest.py, tests/test_ingest_smoke.py, tests/test_config.py
```
........................................                                 [100%]
============================== warnings summary ===============================
tests/test_ensemble.py: 6 warnings
tests/test_fit_index.py: 6 warnings
tests/test_movement_backtest.py: 3 warnings
  C:\Users\geric\AppData\Roaming\Python\Python310\site-packages\sklearn\decomposition\_truncated_svd.py:273: RuntimeWarning: invalid value encountered in divide
    self.explained_variance_ratio_ = exp_var / full_var

tests/test_ensemble.py: 6 warnings
tests/test_fit_index.py: 6 warnings
tests/test_movement_backtest.py: 3 warnings
  C:\Users\geric\AppData\Roaming\Python\Python310\site-packages\sklearn\base.py:1365: ConvergenceWarning: Number of distinct clusters (1) found smaller than n_clusters (8). Possibly due to duplicate points in X.
    return fit_method(estimator, *args, **kwargs)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
40 passed, 30 warnings in 127.07s (0:02:07)
```
- **Exit code:** 0

## Visual tests

- `reports\validation\option_d\figures\13_ensemble_weights.png`
- `reports\validation\option_d\figures\14_dashboard_data_health.png`
- `reports\validation\option_d\figures\12_calibration_curve.png`

- **Finished (UTC):** 2026-05-26T22:12:13Z