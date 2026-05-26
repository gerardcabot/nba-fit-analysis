# Option A validation run log

- **Started (UTC):** 2026-05-26T20:31:44Z
- **Primary season:** 2024-25
- **Fallback season:** 2025-26

## Commands

1. `python -m nba_fit ingest --season 2024-25 --tier mvp`
2. Season context from interim (`prefer_interim=True`, `prefer_api=False`)
3. `rank-player 2544 --top 30`
4. `rank-team 1610612747 --top 30`
5. Fit card player 2544 vs team 1610612747

## Ingest

- **Status:** pass (2024-25)
- **Rows:** players=569, teams=30

## Rankings (live / interim)

- **Context source:** interim (season=2024-25)
- **Player rankings rows:** 30
- **Team rankings rows:** 30
- **Synthetic fallback:** False

## Pytest

```
...........                                                              [100%]
11 passed in 8.04s
```
- **Exit code:** 0

## Visual tests

- `PYTHONPATH=<repo> python visual_tests/03_player_stat_distributions.py` (2024-25 interim via settings patch) — **pass**
- `PYTHONPATH=<repo> python visual_tests/05_fit_score_heatmap.py` — **pass** (synthetic demo heatmap)
- Figures saved: `figures/03_player_stat_distributions.png`, `figures/05_fit_score_heatmap.png`

## Pytest (re-run for log)

```
...........                                                              [100%]
11 passed in 8.62s
```

- **Exit code:** 0

- **Finished (UTC):** 2026-05-26T20:33:00Z
