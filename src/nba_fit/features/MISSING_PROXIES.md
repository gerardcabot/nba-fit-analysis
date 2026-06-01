# Missing proxies (leaguedash-only constraint)

Option A feature vectors intentionally use **only** columns available from `leaguedash*` endpoints.
The following plan items are **not** represented (or only weakly proxied) until supplemental
endpoints are joined in later phases.

| Planned signal | Ideal source | leaguedash-only status | Phase 1 ingest |
|----------------|--------------|------------------------|----------------|
| Touches / time of possession | `playerdashptshots`, `boxscoreplayertrackv3` | **Missing** — no touch-time bins | — |
| Pull-up vs catch-and-shoot | `playerdashptshots` (`DRIBBLE_RANGE`) | **Missing** | — |
| Shot-clock / dribble bins | `playerdashptshots` | **Missing** | — |
| Pass frequency / network | `playerdashptpass`, `teamdashptpass` | **Missing** — only AST/AST% from dash | — |
| Contested / uncontested rebounding | `playerdashptreb` | **Missing** — only OREB/DREB% | — |
| Hustle / deflections / charges | `leaguehustlestatsplayer` | **Ingested** — raw cache via `--tier tactical`; not yet joined into player vector | `ingest --tier tactical` |
| Gravity / spacing pressure | `gravityleaders`, tracking | **Ingested** — raw cache via `--tier tactical`; not yet joined into player vector | `ingest --tier tactical` |
| Estimated metrics (NBA model) | `playerestimatedmetrics` | **Partial** — use `OFF_RATING`/`NET_RATING` from `leaguedashplayerstats` Advanced | `ingest --tier mvp` |
| Matchup defense | `boxscorematchupsv3` | **Missing** | — |
| On/off lineup impact | `teamplayeronoffsummary` | **Missing** from Option A vector (available via `--tier role`) | `ingest --tier role` |
| Lineup weakness gaps | `leaguedashlineups` | **Not aggregated** in team vector yet | `ingest --tier role` |
| Injury / DNP history | `playergamelogs`, external | **Missing** — only GP/MIN availability proxy | — |
| Position (stable) | `commonplayerinfo` | **Partial** — `PLAYER_POSITION` from `leaguedashptdefend` or height/USG heuristics | `ingest --tier tactical` (defend raw) |
| Player tracking distance/speed | `leaguedashptstats` | **Endpoint deprecated** in 2025-26 probe | — |
| Play-type labels | `synergyplaytypes` | **Unreliable** (empty in probe) | — |

## Weak proxies in use

- **Defense:** STL/BLK/DEF_RATING plus `leaguedashptdefend` frequency and `PCT_PLUSMINUS` on contested shots (defend table ingested; join pending).
- **Hustle:** `leaguehustlestatsplayer` / `leaguehustlestatsteam` cached raw; feature join pending.
- **Gravity:** `gravityleaders` cached raw; feature join pending.
- **Creation:** AST%, AST/TOV, AST_RATIO — no passing volume.
- **Availability:** GP and total minutes; no game-missed reason.
- **Roster composition:** optional aggregate of `leaguedashplayerstats` / biostats by `TEAM_ID` (usage concentration).

## Temporal cutoff (Phase 1)

`SeasonFitContext.as_of(move_date)` sets a leakage-safe cutoff for movement backtests.
Use `filter_gamelogs_as_of(gamelogs, move_date)` when building rates from game logs;
full re-materialization from filtered interim tables is deferred to a later phase.
