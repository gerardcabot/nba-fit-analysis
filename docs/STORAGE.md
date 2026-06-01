# Storage layout

NBA Fit uses a **hybrid** persistence model: durable **Parquet** on local disk for anything reused across runs, and **ephemeral** scratch for one-off joins or probes.

## Layers

| Layer | Path | Purpose |
|-------|------|---------|
| Raw | `data/raw/nba_api/` | Cached API responses, hive-partitioned |
| Interim | `data/interim/` | Source-shaped tables after light cleaning |
| Features | `data/features/` | Modeling matrices keyed by season |
| Ephemeral | `data/.ephemeral/` | Short-lived; safe to delete |

Implementation: `src/nba_fit/data/storage.py` (`StorageStrategy`, path helpers, `read_chunks`).

## Why not all in-memory?

The full `nba_api` surface is **143** endpoints; play-by-play alone is ~**500** events × ~**1,230** games per season. Materializing several seasons of PBP, shot charts, and dashboards fits comfortably on disk (compressed Parquet) but not as repeated in-memory DataFrames across workers. Chunked reads via PyArrow keep peak RAM bounded.

## Partitioning (high level)

- **League / dashboard endpoints** — `season=…`
- **shotchartdetail** — `season=…` / `player_id=…`
- **playbyplayv3** — `game_id=…`
- **leaguedashlineups**, **teamdashlineups** — `season=…` / `team_id=…`

Raw paths are gitignored; regenerate via the ingestion pipeline.
