# SOTA multi-season ingest run log

**Date:** 2026-06-01  
**Branch:** `stage/06-sota-final`  
**Data root:** `E:\Nba Fit Data` (`$env:NBA_FIT_DATA_ROOT`)  
**Network:** `stats.nba.com:443` reachable  
**Package:** `pip install -e .` from `nba-fit-analysis` (required so `get_settings()` picks up `NBA_FIT_DATA_ROOT`)

## Environment verification

| Check | Result |
|-------|--------|
| `NBA_FIT_DATA_ROOT` set | `E:\Nba Fit Data` |
| Layout (`raw/`, `interim/`, `features/`, `models/`) | Created via `python scripts/init_nba_fit_data_root.py` |
| Repo `data/raw/` bytes before/after MVP | **5,756,687** unchanged (no new large writes to repo) |
| E: parquet files after MVP | **42** files, **~2.4 MB** under `E:\Nba Fit Data\raw\` |

## Commands and outcomes

| Command | Season | Wall (s) | Status | Notes |
|---------|--------|----------|--------|-------|
| `ingest --tier mvp` | 2023-24 | 18.1 | OK | 572 players, 30 teams; all endpoints **live** |
| `ingest --tier mvp` | 2024-25 | 14.3 | OK | 569 players, 30 teams; all endpoints **live** |
| `ingest --tier mvp` (repeat) | 2023-24 | ~63 | OK | All 7 endpoints **cache** |
| `ingest --tier tactical` | 2023-24 | 28.3 | **FAIL** | `gravityleaders` failed after 3 attempts |
| `ingest --tier tactical` | 2024-25 | 49.9 | **FAIL** | Same `gravityleaders` failure |
| `ingest --tier role` | 2023-24 | 664.5 | OK | 2000 lineup_units, 1352 onoff; log `role_ingest_2023-24.log` |
| `ingest --tier role` | 2024-25 | — | **NOT STARTED** | Run next with `NBA_FIT_DATA_ROOT` set |
| `ingest --tier impact` | either | — | **SKIPPED** | Full season ~1,230 games × PBP; use `--max-games 50` or `--full-season` in dedicated run |
| `materialize-features` | 2023-24 | 12.9 | OK | 572 / 30 / 257 scaling rows |
| `materialize-features` | 2024-25 | 9.7 | OK | 569 / 30 / 257 scaling rows |

## Row counts on E: (interim / features)

| Table | 2023-24 | 2024-25 |
|-------|---------|---------|
| `interim/players` | 572 | 569 |
| `interim/teams` | 30 | 30 |
| `features/player_features_scaled` | 572 | 569 |
| `features/team_features_scaled` | 30 | 30 |
| `features/scaling_params` | 257 | 257 |
| `interim/lineup_units` | 2000 | — |
| `interim/onoff` | 1352 | — |
| `interim/possessions` | — | — |

## Cache hit summary (MVP re-run 2023-24)

All Option A endpoints served from Parquet cache: `leaguedashplayerstats`, `leaguedashteamstats`, `playerestimatedmetrics`, `teamestimatedmetrics`, `leaguedashplayerbiostats`, `leaguedashplayershotlocations`, `leaguedashteamshotlocations`.

## Follow-up commands

```powershell
$env:NBA_FIT_DATA_ROOT = 'E:\Nba Fit Data'
cd C:\Users\geric\Desktop\REPOS\propis\nba-fit-analysis

# After role 2023-24 finishes:
python -m nba_fit ingest --tier role --season 2024-25

# Tactical without gravity (or retry when API stable):
python -m nba_fit ingest --tier tactical --season 2023-24

# Impact dev cap:
python -m nba_fit ingest --tier impact --season 2023-24 --max-games 50
```
