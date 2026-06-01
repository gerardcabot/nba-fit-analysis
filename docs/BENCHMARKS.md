# Benchmarks

External systems and metrics used to judge NBA Fit Analysis quality. See [BIBLIOGRAPHY.md](BIBLIOGRAPHY.md) for URLs and citations.

---

## Impact & value metrics

| System | What it measures | Our analogue | Gap |
|--------|------------------|--------------|-----|
| **RAPTOR** | Luck-adjusted RAPM + box prior | Option C ridge RAPM | Full-season stints; opponent/home context |
| **LEBRON** | Role-aware estimated PM | `projected_impact` ensemble component | Not raw points; needs calibration to PM scale |
| **DARKO** | Daily RAPM with recency | — | No daily refresh or on/off ingest at scale |
| **EPM** | Tracking-informed EPM | Lineup sim + RAPM | No Second Spectrum tracking |

---

## Role & fit semantics

| System | Role model | Our analogue | Gap |
|--------|------------|--------------|-----|
| **BBall Index roles** | Offensive/defensive role database | `industry_role` via `role_taxonomy.py` | Heuristic centroid labels, not learned soft roles |
| **Synergy play types** | Event-tagged creation/shooting | Box/shot-location proxies | No PNR handler rate, C&S gravity |
| **CTG** | Position-adjusted role tables | `ctg_position_group()` | No CTG-style filtering or percentile tables |

---

## Validation benchmarks

| Benchmark | Target | Current evidence (2024-25) |
|-----------|--------|---------------------------|
| **Movement backtest calibration** | Mean calibrated fit ≈ post-move outcome | Option D: mean 0.43 vs pre-move 81.8 — **15 synthetic moves** ([metrics.json](../reports/validation/option_d/metrics.json)) |
| **Holdout season** | Stable rank ordering on new season | Synthetic smoke only ([option_d metrics](../reports/validation/option_d/metrics.json)) |
| **Lineup delta** | Non-zero projected NR delta on real units | Option C: delta 0.0 on 30-game sample ([metrics.json](../reports/validation/option_c/metrics.json)) |
| **Role embedding variance** | Explained variance > 80% | Option B: **84.0% cumulative** on 12 components ([option_b metrics](../reports/validation/option_b/metrics.json)) |
| **Unit tests** | CI green, no network | 40 passed Option D subset; full suite via `pytest -m "not network"` |

---

## Success criteria (SOTA bar)

1. Real transaction labels drive movement backtest (not synthetic fallback).
2. Full-season impact ingest; RAPM coefficients with non-degenerate spread.
3. Ensemble weights learned or grid-searched on held-out seasons with leakage guards.
4. Industry roles exposed on fit cards with soft-role probabilities where available.
5. Calibration curve within tolerance on decile bins (see Option D figure `12_calibration_curve.png`).
