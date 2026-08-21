# Storage layout

NBA Fit uses a **hybrid** persistence model: durable **Parquet** on local disk for anything reused across runs, and **ephemeral** scratch for one-off joins or probes.

## Layers

| Layer | Default path (repo) | External root (`NBA_FIT_DATA_ROOT`) | Purpose |
|-------|---------------------|-------------------------------------|---------|
| Raw | `data/raw/nba_api/` | `{root}/raw/nba_api/` | Cached API responses, hive-partitioned |
| Interim | `data/interim/` | `{root}/interim/` | Source-shaped tables after light cleaning |
| Features | `data/features/` | `{root}/features/` | Modeling matrices keyed by season |
| Models | `models/` | `{root}/models/` | Trained artifacts (RAPM, role embeddings, …) |
| Reports | `reports/` (always in repo) | — | Validation figures, ingest logs |
| Ephemeral | `data/.ephemeral/` | — | Short-lived; safe to delete |

Implementation: `src/nba_fit/data/storage.py` (`StorageStrategy`, path helpers, `read_chunks`). Path resolution: `src/nba_fit/config/settings.py` (`get_settings()`).

## External data root (recommended on Windows)

Large seasons should not live inside the git checkout. Set **`NBA_FIT_DATA_ROOT`** to a dedicated drive or folder; only `raw`, `interim`, `features`, and `models` relocate. **`reports/`** and **`probe_all_results.json`** stay in the repo.

### Windows (PowerShell)

```powershell
$env:NBA_FIT_DATA_ROOT = 'E:\Nba Fit Data'
python scripts\init_nba_fit_data_root.py
```

Persist for your user account (optional):

```powershell
[Environment]::SetEnvironmentVariable('NBA_FIT_DATA_ROOT', 'E:\Nba Fit Data', 'User')
```

Or run `.\scripts\init_nba_fit_data_root.ps1` (creates layout and prints the persist command).

### Expected layout on `E:\` (or any root)

```
E:\Nba Fit Data\
  raw\           # nba_api cache, pbpstats, transactions
  interim\       # players, teams, lineup_units, onoff, possessions, …
  features\      # player_features_scaled, team_features_scaled, …
  models\        # per-season trained weights
```

Init helper: `python scripts/init_nba_fit_data_root.py [optional-path]`

### macOS / Linux

```bash
export NBA_FIT_DATA_ROOT="$HOME/NbaFitData"
python scripts/init_nba_fit_data_root.py
```

## Why not all in-memory?

The full `nba_api` surface is **143** endpoints; play-by-play alone is ~**500** events × ~**1,230** games per season. Materializing several seasons of PBP, shot charts, and dashboards fits comfortably on disk (compressed Parquet) but not as repeated in-memory DataFrames across workers. Chunked reads via PyArrow keep peak RAM bounded.

## Partitioning (high level)

- **League / dashboard endpoints** — `season=…`
- **shotchartdetail** — `season=…` / `player_id=…`
- **playbyplayv3** — `game_id=…`
- **leaguedashlineups**, **teamdashlineups** — `season=…` / `team_id=…`

Raw paths are gitignored; regenerate via the ingestion pipeline.
