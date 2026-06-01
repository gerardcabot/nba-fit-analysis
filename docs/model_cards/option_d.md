# Model Card — Option D (Integrated Ensemble)

**Model ID:** `option_d`  
**Version:** 0.1.0 (MVP integration)  
**Last validated:** 2024-25 season ([RUN_LOG](../reports/validation/option_d/RUN_LOG.md))

---

## Intended use

- Rank player destinations and team acquisition targets from **public NBA stats**.
- Produce **interpretable fit cards** with submetrics, role fit, lineup synergy, and uncertainty bands.
- Support **movement backtest smoke tests** and dashboard exploration — not operational front-office decisions.

**Out of scope:** See [NON_GOALS.md](../NON_GOALS.md).

---

## Model overview

Option D combines Options A–C into a calibrated ensemble:

| Component | Weight (current) | Source module |
|-----------|------------------|---------------|
| Profile fit | 0.28 | Option A submetrics |
| Role fit | 0.12 | Option B team need |
| Team need fit | 0.18 | Option B archetype gaps |
| Projected impact | 0.22 | Option C lineup / RAPM proxy |
| Replacement upgrade | 0.20 | Option A replacement signal |

Weights are **analyst priors** in `scoring/constants.py` — not yet learned from movement labels ([BENCHMARKS.md](../BENCHMARKS.md)).

**Calibration:** Isotonic mapping from raw score to percentile (`models/calibration.py`).

---

## Training data

| Artifact | Season | Notes |
|----------|--------|-------|
| League dash player/team stats | 2024-25 | 569 players, 30 teams |
| Role embeddings + archetypes | 2024-25 | 8 GMM clusters → 6 heuristic labels |
| RAPM / possessions | 2024-25 | **30 games** impact ingest cap |
| Movement backtest | 2024-25 | 15 movements, **synthetic fallback** |

---

## Validation structure

SOTA closure uses a dedicated bundle under `reports/validation/sota/` (see [SOTA_FINAL_EXECUTION.md](../plans/SOTA_FINAL_EXECUTION.md)).

| Artifact | Field / content |
|----------|-----------------|
| `metrics.json` | `p0_gates_passed`, `p0_gate_failures`, `partitions`, `backtest`, `ablation` |
| `metrics.json` → `learned_ensemble_weights` | **Placeholder** — populated when `fit-weights` runs on real movement rows; prior remains `ensemble_component_weights_prior` |
| `metrics.json` → `rapm_benchmark_correlation` | **Placeholder** — vs RAPTOR or LEBRON on ≥100 overlapping players (not yet computed) |
| `metrics.json` → `backtest.n_movements` | **Placeholder** — target ≥50 real moves; P0 gate fails below threshold |
| `RUN_LOG.md` | Human-readable pipeline trace |
| `figures/` | `12_calibration_curve`, `13_ensemble_weights`, `14_dashboard_data_health` |
| `ablation_summary.csv` | Variants A, AB, ABC, D |

**Entrypoint:** `python scripts/run_sota_validation.py` (exits 1 on P0 gate failure).

Layer-specific reports remain under `reports/validation/option_{a,b,c,d}/`.

---

## Metrics (validation evidence)

From `reports/validation/option_d/metrics.json` (smoke, 2024-25):

- **overall_fit_percentile** (demo: player 2544 → LAL): 41.7
- **Backtest:** mean calibrated fit 0.43; pre-move fit 81.8 (synthetic moves)
- **Uncertainty:** mean CI [0.46, 0.58], disagreement 0.09
- **Pytest:** 40 passed (Option D subset)
- **Figures:** calibration curve, ensemble weights, dashboard data health

Re-run `scripts/run_sota_validation.py` after real movement labels and full-season impact ingest for authoritative SOTA metrics in `reports/validation/sota/metrics.json`.

---

## Role & archetype outputs

- **Heuristic label:** GMM centroid rules (`models/archetypes.py`)
- **Industry role:** BBall Index-style mapping via `models/role_taxonomy.py` and `data/reference/archetype_industry_map.csv`
- **CTG group:** Guard / Wing / Big coarse bucket for team need

Example (Option B validation): LeBron fit card archetype `high_usage_creator` → industry role **Primary Ball Handler**.

---

## Limitations

1. RAPM validated on low sample; coefficients collapsed to zero in Option C run.
2. Movement backtest lacks sufficient real transaction labels.
3. Missing proxies documented in `features/MISSING_PROXIES.md` (gravity, C&S, touches).
4. Ensemble weights unproven vs external benchmarks (RAPTOR, LEBRON, EPM).

---

## Ethical considerations

- Public stats only; no private player data.
- Rankings can reinforce existing bias in usage/role allocation; interpret with context.
- Do not use as sole input for employment or contract decisions.

---

## Maintenance

| Action | Command |
|--------|---------|
| Re-train roles | `python -m nba_fit train-roles --season 2024-25` |
| Re-train impact | `python -m nba_fit train-impact --season 2024-25` |
| Full Option D validation | `python scripts/run_option_d_validation.py` |
| SOTA validation bundle | `python scripts/run_sota_validation.py` |
| CI unit tests | `pytest -m "not network"` |

---

## References

See [BIBLIOGRAPHY.md](../BIBLIOGRAPHY.md). Primary external benchmarks: RAPTOR, LEBRON, DARKO, EPM, BBall Index roles, Wharton acquisition research.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-06 | SOTA validation bundle structure; `reports/validation/sota/` placeholders |
| 2026-06 | Added `industry_role` on fit cards; model card template created |
| 2026-05 | Option D MVP validation on 2024-25 |
