# Option A validation run log

- **Started (UTC):** 2026-05-28T17:10:45Z
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
11 passed in 19.27s
```
- **Exit code:** 0

## Visual tests

Visual test 03_player_stat_distributions...
  03_player_stat_distributions exit=0 figures=1
Visual test 05_fit_score_heatmap...
  05_fit_score_heatmap exit=0 figures=1
- `03_player_stat_distributions` — **pass** (exit=0, figures=['figures\\03_player_stat_distributions.png'])
- `05_fit_score_heatmap` — **pass** (exit=0, figures=['figures\\05_fit_score_heatmap.png'])
- **Heatmap vs CSV:** pass — #1 Philadelphia 76ers @ 67.6889 percentile
