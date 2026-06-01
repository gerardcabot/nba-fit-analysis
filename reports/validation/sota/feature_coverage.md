# Feature coverage — SOTA validation

**Season:** `{{SEASON}}`  
**Generated (UTC):** `{{GENERATED_AT}}`  
**Data source:** `{{DATA_SOURCE}}`

## Supplemental joins

| Signal | Interim / cache | Join status | Feature group |
|--------|-----------------|-------------|---------------|
| Hustle (deflections, charges, …) | `leaguehustlestatsplayer` raw | `{{HUSTLE_STATUS}}` | `pf_tactical__*` |
| Pt defend (freq, D_FG%, +/-) | `leaguedashptdefend` raw | `{{DEFEND_STATUS}}` | `pf_tactical__*` / `pf_defense__*` |
| Gravity score | `gravityleaders` raw | `{{GRAVITY_STATUS}}` | `pf_tactical__*` |
| On/off NET delta | `onoff` interim | `{{ONOFF_STATUS}}` | `pf_onoff__*` |
| Lineup weakness | `lineup_units` interim | `{{LINEUP_STATUS}}` | `tf_weaknesses__lineup_*` |

## Still missing (see `src/nba_fit/features/MISSING_PROXIES.md`)

- Touches / time of possession
- Pull-up vs catch-and-shoot bins
- Pass network volume
- Contested rebounding splits
- Injury / DNP reasons
- Full play-type synergy labels

## Player feature groups

{{PLAYER_GROUPS}}

## Team feature groups

{{TEAM_GROUPS}}

## Notes

- Joins are **left** merges; NaN features are scaled to zero after `scale_*_features`.
- Re-run `ingest --tier tactical` and `ingest --tier role` before validation when caches are stale.
