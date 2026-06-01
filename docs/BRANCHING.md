# Concatenated Branch Strategy (Option D)

Progressive stages; each branch is cut from the previous stage branch.

| Order | Branch | Contains |
|------:|--------|----------|
| 0 | `main` | Probes, endpoint reference, validation benchmark |
| 1 | `stage/00-foundation` | Package layout, hybrid Parquet cache, registry, CLI health |
| 2 | `stage/01-option-a` | Interpretable fit index MVP |
| 3 | `stage/02-option-b` | Role embeddings, archetypes, team need |
| 4 | `stage/03-option-c` | PBP possessions, RAPM, lineup impact |
| 5 | `stage/04-option-d` | Ensemble calibration, movement backtest, dashboard (production) |

## GitHub remote

```bash
git remote add origin https://github.com/YOUR_ORG/nba-fit-analysis.git
git push -u origin main
git push -u origin stage/00-foundation
git push -u origin stage/01-option-a
git push -u origin stage/02-option-b
git push -u origin stage/03-option-c
git push -u origin stage/04-option-d
```

Default development branch for the full product: **`stage/04-option-d`**.
