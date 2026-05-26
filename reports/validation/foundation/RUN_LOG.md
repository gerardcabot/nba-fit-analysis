# Foundation validation run log

- **Started (UTC):** 2026-05-26T22:16:04Z
- **Branch:** stage/00-foundation

## Commands

1. `python -m nba_fit health`
2. `python -m nba_fit fetch-sample --endpoint leaguedashplayerstats --season 2024-25`
3. `pytest tests/test_config.py tests/test_client_smoke.py -q`
4. `python visual_tests/01_endpoint_health.py`
5. `python visual_tests/02_pbp_schema.py` (live PlayByPlayV3, one game)

## Health

- **Exit code:** 0
```
NBA Fit — endpoint health (from probe_all_results.json)
  Probe timestamp: 2026-05-26 17:44 UTC
  Season:          2025-26
  Totals:          OK=100 EMPTY=6 FAIL=36

Essential registry endpoints:
  commonallplayers                 OK         rows=587
  commonteamroster                 OK         rows=25
  commonplayerinfo                 OK         rows=28
  leaguedashplayerbiostats         OK         rows=20
  leaguedashplayerstats            OK         rows=20
  playerestimatedmetrics           OK         rows=582
  playergamelogs                   OK         rows=66
  playergamelog                    OK         rows=64
  leaguedashteamstats              OK         rows=1
  teamestimatedmetrics             OK         rows=30
  teamgamelogs                     OK         rows=98
  teamdashboardbygeneralsplits     OK         rows=20
  shotchartdetail                  OK         rows=1477
  leaguedashplayershotlocations    OK         rows=20
  leaguedashteamshotlocations      OK         rows=1
  playerdashptshots                OK         rows=27
  leaguedashplayerptshot           OK         rows=19
  leaguedashteamptshot             OK         rows=1
  teamdashptshots                  OK         rows=26
  playerdashptpass                 OK         rows=34
  teamdashptpass                   OK         rows=40
  playerdashptreb                  OK         rows=14
  teamdashptreb                    OK         rows=14
  leaguehustlestatsplayer          OK         rows=20
  leaguehustlestatsteam            OK         rows=1
  leaguedashptdefend               OK         rows=1
  leaguedashptteamdefend           OK         rows=1
  boxscorematchupsv3               OK         rows=207
  leaguedashlineups                OK         rows=641
  teamdashlineups                  OK         rows=251
  teamplayeronoffsummary           OK         rows=41
  teamplayeronoffdetails           OK         rows=41
  gamerotation                     OK         rows=52
  playbyplayv3                     OK         rows=500
  leaguegamefinder                 OK         rows=66
  leaguegamelog                    OK         rows=26651
  scheduleleaguev2                 OK         rows=1440
  boxscoresummaryv3                OK         rows=25
  boxscoreadvancedv3               OK         rows=28
  boxscorefourfactorsv3            OK         rows=28
  boxscoreusagev3                  OK         rows=28
  boxscoreplayertrackv3            OK         rows=28
```

## Fetch sample

- **Exit code:** 0
- **Rows:** 24
```
Fetching leaguedashplayerstats season=2024-25 (cache=on)...
  from_cache=True elapsed=0.00s rows=24
    LeagueDashPlayerStats: 24 rows, 67 cols
  cache: C:\Users\geric\Desktop\REPOS\propis\nba-fit-analysis\data\raw\nba_api\leaguedashplayerstats\season=2024-25\data_23cacbff7ba1.parquet
```

## Pytest

- **Exit code:** 0
- **Summary:** 7 passed in 5.16s
```
.......                                                                  [100%]
7 passed in 5.16s
```

## Visual tests

- **01_endpoint_health.py** exit=0
  `Wrote C:\Users\geric\Desktop\REPOS\propis\nba-fit-foundation-wt\reports\figures\01_endpoint_health.png`
- **02_pbp_schema.py** exit=0, pbp_rows=499
  `Wrote C:\Users\geric\Desktop\REPOS\propis\nba-fit-foundation-wt\reports\figures\02_pbp_action_type_distribution.png`

## Figures

- Copied `01_endpoint_health.png` -> `figures/01_endpoint_health.png`
- Copied `02_pbp_action_type_distribution.png` -> `figures/02_pbp_action_type_distribution.png`

- **Finished (UTC):** 2026-05-26T22:16:38Z