# Missing proxies (leaguedash-only constraint)

Option A feature vectors start from `leaguedash*` columns; **SOTA supplemental joins**
add tactical hustle/defend/gravity, on/off deltas, and lineup weakness proxies when
interim or raw caches exist (see `features/supplemental_join.py`).

| Planned signal | Ideal source | Status | Ingest |
|----------------|--------------|--------|--------|
| Touches / time of possession | `playerdashptshots`, `boxscoreplayertrackv3` | **Missing** | — |
| Pull-up vs catch-and-shoot | `playerdashptshots` (`DRIBBLE_RANGE`) | **Missing** | — |
| Shot-clock / dribble bins | `playerdashptshots` | **Missing** | — |
| Pass frequency / network | `playerdashptpass`, `teamdashptpass` | **Missing** | — |
| Contested / uncontested rebounding | `playerdashptreb` | **Missing** | — |
| Hustle / deflections / charges | `leaguehustlestatsplayer` | **Joined** → `pf_tactical__*` when raw cache present | `ingest --tier tactical` |
| Gravity / spacing pressure | `gravityleaders` | **Joined** → `pf_tactical__gravity_*` when raw cache present | `ingest --tier tactical` |
| Pt defend (freq, D_FG%, +/-) | `leaguedashptdefend` | **Joined** → `pf_tactical__defend_*` / `pf_defense__*` | `ingest --tier tactical` |
| Estimated metrics (NBA model) | `playerestimatedmetrics` | **Partial** — `OFF_RATING`/`NET_RATING` from dash Advanced | `ingest --tier mvp` |
| Matchup defense | `boxscorematchupsv3` | **Missing** | — |
| On/off lineup impact | `teamplayeronoffsummary` | **Joined** → `pf_onoff__*` when role interim present | `ingest --tier role` |
| Lineup weakness gaps | `leaguedashlineups` | **Joined** → `tf_weaknesses__lineup_*` on team vector | `ingest --tier role` |
| Injury / DNP history | `playergamelogs`, external | **Missing** — GP/MIN availability proxy only | — |
| Position (stable) | `commonplayerinfo` | **Partial** — `PLAYER_POSITION` from defend or height/USG heuristics | `ingest --tier tactical` |
| Player tracking distance/speed | `leaguedashptstats` | **Endpoint deprecated** in 2025-26 probe | — |
| Play-type labels | `synergyplaytypes` | **Unreliable** (empty in probe) | — |

## Weak proxies in use

- **Defense:** STL/BLK/DEF_RATING plus defend frequency and `PCT_PLUSMINUS` when tactical join succeeds.
- **Hustle / gravity:** league hustle and gravity leaders (per-game rates in vector).
- **On/off:** focal-player on minus off NET_RATING (minutes-weighted) from role-tier interim.
- **Lineup weakness:** worst/mean five-man NET_RATING and negative-lineup minute share per team.
- **Creation:** AST%, AST/TOV, AST_RATIO — no passing volume.
- **Availability:** GP and total minutes; no game-missed reason.

## Temporal cutoff (Phase 1)

`SeasonFitContext.as_of(move_date)` sets a leakage-safe cutoff for movement backtests.
Use `filter_gamelogs_as_of(gamelogs, move_date)` when building rates from game logs;
full re-materialization from filtered interim tables is deferred to a later phase.

## Coverage report

After ingest + `SeasonFitContext.from_interim`, fill `reports/validation/sota/feature_coverage.md`
from `supplemental_coverage_report()` and feature group lists.
