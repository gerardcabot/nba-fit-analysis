# NBA Fit Analysis — SOTA Completion Plan

**Status:** Option D integrated MVP → research-grade player–team fit system  
**Baseline branch:** `stage/04-option-d` / `main` after PR stack merge  
**References:** [BIBLIOGRAPHY.md](../BIBLIOGRAPHY.md) · [BENCHMARKS.md](../BENCHMARKS.md) · [NON_GOALS.md](../NON_GOALS.md)

---

## Executive summary

NBA Fit Analysis today delivers an **interpretable Option D ensemble**: profile submetrics (A), role embeddings and heuristic archetypes (B), lineup/RAPM impact proxies (C), and isotonic calibration plus uncertainty bands (D). Validation on **2024-25** shows a working ingest → train → rank → fit-card pipeline with **569 embedded players**, **84% cumulative embedding variance**, and **40 passing unit tests** — but critical SOTA gaps remain.

**What works (evidence):**

| Layer | Evidence |
|-------|----------|
| Option B roles | 569 players, 6 archetype labels, 8 GMM clusters ([option_b/metrics.json](../../reports/validation/option_b/metrics.json)) |
| Embeddings | 84.0% cumulative explained variance on 12 SVD components |
| Option D ensemble | Isotonic calibration fitted; backtest smoke on 15 moves ([option_d/metrics.json](../../reports/validation/option_d/metrics.json)) |
| CI / tests | `pytest -m "not network"`; GitHub Actions workflow |
| Taxonomy (new) | Heuristic → BBall Index industry roles via `role_taxonomy.py` |

**What blocks SOTA:**

1. **Movement backtest** uses synthetic labels (15 moves; pre-move fit 81.8 vs post 0.43 — not meaningful yet).
2. **RAPM / lineup impact** trained on **30 games**; all 549 players flagged low-sample; net RAPM std = 0 ([option_c/metrics.json](../../reports/validation/option_c/metrics.json)).
3. **Weights are priors** — ensemble and submetric weights not learned from held-out outcomes.
4. **Feature gaps** — no touches, C&S, gravity on leaguedash-only path (`features/MISSING_PROXIES.md`).
5. **Archetype labels** — heuristic centroid names now mapped to industry roles, but clustering is not constrained to industry taxonomy.

**Target:** Research/portfolio SOTA aligned with original vision: feature store, lineup-aware impact, real movement validation, calibrated rankings, dashboard — using **free public data only** ([NON_GOALS.md](../NON_GOALS.md)).

---

## Prioritized backlog

### P0 — Must ship for credible SOTA claims

| ID | Problem | Solution | Files | Validation |
|----|---------|----------|-------|------------|
| P0-1 | Synthetic movement labels | Wire `transactions` fetcher + external CSV; require ≥50 real moves before reporting calibration | `evaluation/movement_backtest.py`, `data/fetchers/transactions.py` | Mean decile calibration on real moves |
| P0-2 | 30-game RAPM cap | Full-season impact ingest; tune `RAPM_RIDGE_ALPHA` vs literature | `data/ingest.py`, `models/rapm.py`, `config/settings.py` | Non-zero RAPM spread; Option C metrics |
| P0-3 | Unproven ensemble weights | Grid search / bootstrap on held-out seasons with leakage guards | `scoring/constants.py`, `scoring/ensemble.py`, `evaluation/holdout_season.py` | Ablation A→D on 2024-25 holdout |
| P0-4 | Industry role on outputs | ✅ `industry_role` on archetypes + fit cards | `role_taxonomy.py`, `archetypes.py`, `fit_card.py` | `test_role_taxonomy.py` |
| P0-5 | CI regression gate | ✅ GitHub Actions `pytest -m "not network"` | `.github/workflows/ci.yml` | Green on push/PR |

### P1 — Should ship for research-grade quality

| ID | Problem | Solution | Files | Validation |
|----|---------|----------|-------|------------|
| P1-1 | Heuristic-only archetypes | Soft role probabilities from GMM posteriors + industry mapping | `archetypes.py`, `role_taxonomy.soft_role_display` | Manual review vs BBall Index sample |
| P1-2 | Inline magic numbers | Move literals in `submetrics.py` to named constants | `scoring/constants.py`, `submetrics.py` | Unit tests per constant |
| P1-3 | Missing proxies | Add hustle, passing, tracking bins from public endpoints | `config/endpoints.py`, `features/player_vector.py` | Feature coverage report |
| P1-4 | Uncertainty heuristic | Bootstrap conformal bands on ensemble components | `scoring/uncertainty.py` | Coverage on backtest deciles |
| P1-5 | Fit card contract | ✅ TypedDict in `contracts/fit_card_schema.py` | CLI/dashboard parity test | `test_fit_card_structure` |

### P2 — Nice to have

| ID | Problem | Solution | Files | Validation |
|----|---------|----------|-------|------------|
| P2-1 | UMAP archetype map | Optional `umap-learn` for figure 07 | `visual_tests/07_archetype_map.py` | Visual regression |
| P2-2 | Export rankings API | JSON/Parquet export command | `cli.py`, `scoring/ranker.py` | Golden file |
| P2-3 | Notebook narratives | Walkthrough for portfolio readers | `notebooks/` | — |
| P2-4 | Constrained clustering | Must-link industry role priors in GMM | `models/archetypes.py` | Archetype purity vs map |

---

## 1. Current state

### Implemented stack (Options A–D)

```
Ingest (nba_api + cache) → interim Parquet
  → player/team vectors (A)
  → role embeddings + GMM archetypes + team need (B)
  → possessions + RAPM + lineup sim (C)
  → ensemble + calibration + uncertainty + dashboard (D)
```

### Validation artifacts (2024-25)

| Report | Key metrics |
|--------|-------------|
| [option_b/metrics.json](../../reports/validation/option_b/metrics.json) | 569 embedded; labels: versatile_forward (212), movement_shooter (114), high_usage_creator (82), rim_running_big (67), rim_protector (61), connector_wing (33) |
| [option_c/metrics.json](../../reports/validation/option_c/metrics.json) | 30 games, 5991 possession rows; RAPM degenerate (all zeros) |
| [option_d/metrics.json](../../reports/validation/option_d/metrics.json) | Ensemble weights as above; backtest synthetic; holdout synthetic smoke |
| [option_d/RUN_LOG.md](../../reports/validation/option_d/RUN_LOG.md) | Endpoint health: 42/42 essential OK; 36 FAIL on extended probe |

### New in this plan tranche

- `data/reference/archetype_industry_map.csv` — heuristic → industry role → CTG group
- `src/nba_fit/models/role_taxonomy.py` — mapping API
- `docs/BIBLIOGRAPHY.md`, `BENCHMARKS.md`, `NON_GOALS.md`
- `docs/model_cards/option_d.md`
- `.github/workflows/ci.yml`

---

## 2. Gaps vs original vision

Coverage matrix (original plan → implementation):

| Plan section | Status | Evidence / path |
|--------------|--------|-----------------|
| Feature store (multi-season Parquet) | **Partial** | `data/interim/` single-season MVP |
| Interpretable fit index | **Done** | Option A submetrics + fit cards |
| Role embeddings & archetypes | **Partial** | GMM + heuristic labels; industry map added |
| Lineup-aware impact | **Partial** | RAPM exists; low-sample / zero spread |
| Movement backtest | **Partial** | Pipeline exists; synthetic labels |
| Calibrated rankings | **Done** | Isotonic percentile |
| Uncertainty bands | **Partial** | Heuristic; not conformal |
| Streamlit dashboard | **Done** | `app/dashboard.py` |
| Weight learning | **Missing** | Priors in constants |
| Real transaction pipeline | **Missing** | `transactions.py` stub / sparse |
| Full-season PBP | **Missing** | `--max-games 30` in validation |

---

## 3. Hardcoded / unproven parameters

### High-priority inventory

| Symbol | Value | Location | Rationale stated | Evidence gap |
|--------|-------|----------|------------------|--------------|
| `ENSEMBLE_COMPONENT_WEIGHTS` | 0.28/0.12/0.18/0.22/0.20 | `scoring/constants.py` | Analyst prior | No movement-learned weights |
| `SUBMETRIC_WEIGHTS` | various | `scoring/constants.py` | Analyst prior | No ablation on holdout |
| `ARCHETYPE_N_COMPONENTS` | 8 | `models/constants.py` | Elbow + manual audit | Only 6 labels on 2024-25 |
| `RAPM_RIDGE_ALPHA` | 2500 | `models/constants.py` | Literature range | Not tuned on full season |
| `ROLE_FIT_*` blend weights | 0.60/0.40 | `models/constants.py` | Design choice | No A/B test |
| Archetype thresholds | z-score cutoffs | `models/constants.py` | Basketball heuristics | No golden-file regression |
| Inline `0.3`, `0.25` | — | `submetrics.py` | Undocumented | Move to constants + tests |

### Proposed fixes

1. **Learn ensemble weights** — constrained optimization on movement backtest OUTCOMES with player-level holdout.
2. **BIC/elbow sweep** — store sweep results under `reports/validation/option_b/` for GMM k.
3. **Constants module tests** — assert sums-to-one, bounds, and snapshot of threshold ordering.
4. **Literature anchors** — cite [BIBLIOGRAPHY.md](../BIBLIOGRAPHY.md) defaults for ridge alpha and recency half-life.

---

## 4. Archetype & role taxonomy upgrade

### Current (Option B validation)

Heuristic labels from cluster centroids in z-scored feature space. Example mislabels observed: Khris Middleton as `rim_protector` on archetype board ([option_b/metrics.json](../../reports/validation/option_b/metrics.json)) — centroid rules are fragile.

### Implemented (Phase 0)

| Artifact | Purpose |
|----------|---------|
| `archetype_industry_map.csv` | 12 heuristic → industry mappings |
| `INDUSTRY_OFFENSIVE_ROLES` | BBall Index-style vocabulary |
| `map_heuristic_to_industry()` | Single-label display |
| `soft_role_display()` | Top-k soft role string for fit cards |
| `ctg_position_group()` | Guard / Wing / Big aggregation |
| `industry_role` column | On `player_archetypes.parquet` |

### Recommended Phase 1–2

1. **GMM soft assignments** — expose posterior weights per industry role (aggregate clusters sharing mapped role).
2. **Manual audit set** — 30 players with known BBall Index roles; compute mapping accuracy.
3. **Constrained clustering** — penalize clusters whose centroid maps to conflicting industry roles.
4. **Team need by CTG group** — roll archetype gaps to Guard/Wing/Big for dashboard radar.

Reference: [BBall Index roles](../BIBLIOGRAPHY.md#bball-index--offensive-roles), [CTG](../BIBLIOGRAPHY.md#cleaning-the-glass-ctg).

---

## 5. Data & features

### Endpoint integration (free data)

| Missing signal | Endpoint / source | Priority |
|----------------|---------------------|----------|
| Passing | `leaguedashptstats` (if available) / PBP aggregates | P1 |
| Hustle | `leaguehustlestatsplayer` | P1 |
| Lineups | `leaguedashlineups` | P0 (Option C) |
| On/off | team-scoped pulls (role tier) | P0 |
| Tracking bins | limited public summaries | P2 |

### Feature store target

```
data/interim/{players,teams,lineup_units,onoff,possessions}/season=YYYY-YY/
data/features/player_vectors/season=YYYY-YY/
data/features/role_embeddings/season=YYYY-YY/
```

### Leakage rules

- Pre-move features only before transaction date in backtest.
- Train seasons strictly before predict season in holdout.
- No future-season stats in current-season rankings.

### Ops

- Remove `--max-games` cap for production validation runs.
- Cache API responses; rate-limit in CI with mocked client (`pytest -m "not network"`).

---

## 6. Modeling & validation

### RAPM / lineup (Option C)

**Current:** 30 games → 549 players, all low-sample, zero RAPM variance.

**Target:**

- Full-season possessions (≈200 possessions/game × 82 games).
- Tune `RAPM_RIDGE_ALPHA` against public RAPTOR/LEBRON correlation on overlapping players.
- Add opponent and home context columns to stint matrix.
- Report projected NR delta in **points/100** on fit card alongside 0–1 `lineup_impact_fit`.

### Movement backtest

**Current:** 15 synthetic moves; mean calibrated fit 0.43 ([option_d/metrics.json](../../reports/validation/option_d/metrics.json)).

**Target:**

- ≥50 real moves from transactions fetcher.
- Metrics: decile calibration, Spearman on post-move performance, replacement-player benchmark.
- Selection bias note in model card.

### Uncertainty

Replace fixed band width with bootstrap over ensemble components or conformal prediction on movement residuals.

### Ablation study (2024-25 holdout)

| Config | Measure |
|--------|---------|
| A only | Profile submetrics ranking stability |
| A + B | Incremental movement calibration |
| A + B + C | Lineup delta correlation |
| Full D | Calibrated percentile + uncertainty coverage |

---

## 7. Bugs & engineering

### Known issues (from validation logs)

| Priority | Issue | Repro | Fix |
|----------|-------|-------|-----|
| P1 | sklearn GMM convergence warning on synthetic data | `test_archetypes.py` | Increase synthetic variance or reduce k in tests |
| P1 | SVD divide-by-zero warning | synthetic feature matrix | Add ridge to covariance / drop constant cols |
| P2 | `fit_index` neutral 0.5 fallbacks | missing artifacts | Log explicit fallback reason on fit card |
| P2 | Naming drift (`role_fit` vs `team_need_fit`) | fit card vs submetrics | Document in `fit_card_schema.py` |
| P1 | Dashboard probe 36 FAIL | extended endpoints | Separate essential vs extended health |

### Engineering deliverables (done / planned)

- ✅ CI workflow `.github/workflows/ci.yml`
- ✅ `contracts/fit_card_schema.py`
- ✅ `docs/model_cards/option_d.md`
- Planned: golden fit-card JSON for player 2544 / LAL
- Planned: mock ingest smoke in CI without network marker

---

## 8. Phased roadmap

### Phase 1 — Credibility (weeks 1–2)

- Real movement labels pipeline (P0-1)
- Full-season impact ingest (P0-2)
- Document baseline metrics in `reports/validation/`

**Exit:** Backtest on ≥50 real moves; RAPM std > 0 on full season.

### Phase 2 — Calibration (weeks 3–4)

- Learn ensemble weights on movement OUTCOMES (P0-3)
- Ablation A→D on holdout
- Bootstrap uncertainty (P1-4)

**Exit:** Beats equal-weight ensemble on movement decile calibration.

### Phase 3 — Features & roles (weeks 5–6)

- Hustle / passing endpoints (P1-3)
- GMM soft industry roles (P1-1)
- CTG-group team need radar

**Exit:** Feature coverage report; mapping audit on 30-player gold set.

### Phase 4 — Product polish (weeks 7–8)

- Dashboard parity with fit card contract
- Export rankings
- Portfolio notebook + updated model card

**Exit:** All P0/P1 backlog closed; model card reflects learned weights.

---

## 9. Definition of done (SOTA)

The project reaches **research-grade SOTA** when all of the following hold:

### Data

- [ ] Multi-season feature store with documented schema
- [ ] Full-season possessions for impact tier (no validation game cap)
- [ ] Real transaction table linked to player-team-date

### Models

- [ ] RAPM coefficients with documented correlation to a public benchmark (RAPTOR or LEBRON) on ≥100 players
- [ ] Industry roles on all fit outputs; soft roles where GMM posteriors available
- [ ] Ensemble weights learned or selected via documented search — not static priors

### Validation

- [ ] Movement backtest on ≥50 **real** moves with decile calibration plot
- [ ] Held-out season evaluation (train ≤2023-24, predict 2024-25) with reported rank metrics
- [ ] Ablation table (A, A+B, A+B+C, D) archived under `reports/validation/`
- [ ] CI green: `pytest -m "not network"` on Python 3.10 and 3.11

### Product

- [ ] Fit card matches `FitCard` contract including `industry_role`
- [ ] Model card `docs/model_cards/option_d.md` updated with learned weights and validation dates
- [ ] Dashboard essential endpoint health 100% on supported season

### Documentation

- [ ] [BIBLIOGRAPHY.md](../BIBLIOGRAPHY.md) cited for every major modeling choice
- [ ] [NON_GOALS.md](../NON_GOALS.md) honored — no paywalled data claims
- [ ] This plan archived with completion dates per phase

---

## Appendix — Validation figure index

| Figure | Path | Layer |
|--------|------|-------|
| Archetype map | `reports/validation/option_b/figures/07_archetype_map.png` | B |
| Team need radar | `reports/validation/option_b/figures/08_team_need_radar.png` | B |
| RAPM distribution | `reports/validation/option_c/figures/10_rapm_distribution.png` | C |
| Lineup delta | `reports/validation/option_c/figures/option_c/11_lineup_delta_bar.png` | C |
| Calibration curve | `reports/validation/option_d/figures/12_calibration_curve.png` | D |
| Ensemble weights | `reports/validation/option_d/figures/13_ensemble_weights.png` | D |
| Data health | `reports/validation/option_d/figures/14_dashboard_data_health.png` | D |

---

*Generated as part of SOTA completion tranche. Update this document when closing P0/P1 backlog items.*
