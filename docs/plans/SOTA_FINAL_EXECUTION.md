# SOTA Final Execution Plan

**Branch:** `stage/06-sota-final`  
**Parent plan:** [SOTA_COMPLETION_PLAN.md](SOTA_COMPLETION_PLAN.md)  
**Validation entrypoint:** `python scripts/run_sota_validation.py`  
**Artifacts:** `reports/validation/sota/`

This document operationalizes phases **A–F** for research-grade closure. Phase **E** documents external data-root layout (`NBA_FIT_DATA_ROOT`). Section **9** mirrors the definition-of-done checklist from the parent plan.

---

## Phase A — Data foundation & partitions

**Goal:** Durable interim Parquet for MVP + role + impact tiers on the target season.

| Step | Command / action |
|------|------------------|
| Set data root (recommended) | `export NBA_FIT_DATA_ROOT=/path/to/nba-fit-data` |
| Ensure layout | `python -c "from nba_fit.config.settings import ensure_data_root_layout; ensure_data_root_layout()"` |
| Ingest MVP | `python -m nba_fit ingest --tier mvp --season 2024-25` |
| Ingest role | `python -m nba_fit ingest --tier role --season 2024-25` |
| Ingest impact (full season) | `python -m nba_fit ingest --tier impact --season 2024-25 --full-season` |

**Required partitions** (under `{data_interim}/`):

| Table | Hive path |
|-------|-----------|
| players | `players/season=2024-25/` |
| teams | `teams/season=2024-25/` |
| lineup_units | `lineup_units/season=2024-25/` |
| onoff | `onoff/season=2024-25/` |
| possessions | `possessions/season=2024-25/` |

**Exit:** All five partitions contain at least one `.parquet` file; `run_sota_validation.py` reports `partitions.all_ready: true`.

---

## Phase B — Roles & archetypes (Option B)

**Goal:** Embeddings, GMM archetypes, industry-role mapping on fit outputs.

| Step | Command |
|------|---------|
| Train roles | `python -m nba_fit train-roles --season 2024-25` |
| Option B validation (figures) | `python scripts/run_option_b_validation.py` |

**Evidence paths:**

- `models/role_embedding/season=2024-25/`
- `reports/validation/option_b/metrics.json`
- `reports/validation/option_b/figures/07_archetype_map.png`

**Exit:** `train_roles_ok`; archetype board includes `industry_role` (see `test_role_taxonomy.py`).

---

## Phase C — Impact & RAPM (Option C)

**Goal:** Non-degenerate net RAPM spread on full-season possessions.

| Step | Command |
|------|---------|
| Train impact | `python -m nba_fit train-impact --season 2024-25` |
| Option C validation | `python scripts/run_option_c_validation.py` |

**P0 gate:** `net_rapm_std > 0` in `reports/validation/sota/metrics.json`.

**Evidence paths:**

- `models/rapm/season=2024-25/player_rapm.parquet`
- `reports/validation/option_c/metrics.json`

**Placeholder (model card):** `rapm_benchmark_correlation` vs RAPTOR/LEBRON — fill when overlap export exists.

---

## Phase D — Ensemble, calibration & movement (Option D)

**Goal:** Real movement labels, isotonic calibration, learned-weight stub on labeled rows.

| Step | Command |
|------|---------|
| Curate movements | Copy `data/raw/transactions/movements_template.csv` → `movements.csv` or build cache via gamelog inference |
| Backtest (no synthetic) | `python -m nba_fit backtest-movement --season 2024-25 --no-synthetic` |
| Learn weights (stub) | `python -m nba_fit fit-weights --season 2024-25` |
| Option D validation | `python scripts/run_option_d_validation.py` |

**P0 gates:**

- `movements_synthetic: false`
- `n_movements >= 50`

**Evidence:** `reports/validation/sota/backtest_results.csv`, calibration figures under `reports/validation/sota/figures/`.

---

## Phase E — External data root layout

When `NBA_FIT_DATA_ROOT` is set, raw/interim/features/models live **outside** the git checkout; reports stay in-repo.

```
$NBA_FIT_DATA_ROOT/
  raw/          # nba_api cache (hive-partitioned)
  interim/      # players, teams, lineup_units, onoff, possessions
  features/     # player_vectors, role_embeddings
  models/       # rapm, role_embedding, calibration
```

Repo-relative (default when env unset):

```
nba-fit-analysis/
  data/raw/
  data/interim/
  data/features/
  models/
  reports/validation/sota/   # always under repo root
```

See [STORAGE.md](../STORAGE.md) and `src/nba_fit/config/settings.py` (`ENV_NBA_FIT_DATA_ROOT`, `ensure_data_root_layout`).

---

## Phase F — SOTA validation bundle & closure

**Goal:** Single command produces metrics, run log, figures, and fails loudly on P0 regressions.

```bash
export NBA_FIT_DATA_ROOT=/path/to/nba-fit-data   # recommended
python scripts/run_sota_validation.py
```

| Output | Purpose |
|--------|---------|
| `reports/validation/sota/metrics.json` | Machine-readable gates + ablation + weights |
| `reports/validation/sota/RUN_LOG.md` | Human-readable step trace |
| `reports/validation/sota/figures/` | Calibration, ensemble weights, data health |
| `reports/validation/sota/ablation_summary.csv` | A / AB / ABC / D comparison |

**P0 exit codes (script returns 1):**

| Gate | Condition |
|------|-----------|
| `missing_required_partitions` | Any required interim table missing/empty |
| `movements_synthetic` | Synthetic fallback used or empty real set treated as failure |
| `n_movements_below_threshold` | `n_movements < 50` |
| `net_rapm_degenerate` | `net_rapm_std == 0` |

**CI (non-network):**

```bash
pytest -m "not network" -q
```

---

## §9 Definition of done (checklist)

Status tracked in [SOTA_COMPLETION_PLAN.md §9](SOTA_COMPLETION_PLAN.md#9-definition-of-done-sota). Re-run `run_sota_validation.py` after each phase closure and update checkboxes only when evidence exists.

### Data

- [ ] Multi-season feature store with documented schema
- [ ] Full-season possessions for impact tier (no validation game cap)
- [ ] Real transaction table linked to player-team-date

### Models

- [ ] RAPM coefficients with documented correlation to a public benchmark (RAPTOR or LEBRON) on ≥100 players
- [x] Industry roles on all fit outputs; soft roles where GMM posteriors available
- [ ] Ensemble weights learned or selected via documented search — not static priors

### Validation

- [ ] Movement backtest on ≥50 **real** moves with decile calibration plot
- [ ] Held-out season evaluation (train ≤2023-24, predict 2024-25) with reported rank metrics
- [ ] Ablation table (A, A+B, A+B+C, D) archived under `reports/validation/`
- [x] CI configured: `pytest -m "not network"` on Python 3.10 and 3.11 (`.github/workflows/ci.yml`)

### Product

- [x] Fit card matches `FitCard` contract including `industry_role` (`tests/test_fit_index.py`)
- [ ] Model card `docs/model_cards/option_d.md` updated with learned weights and validation dates
- [ ] Dashboard essential endpoint health 100% on supported season

### Documentation

- [x] [BIBLIOGRAPHY.md](../BIBLIOGRAPHY.md) present for modeling citations
- [x] [NON_GOALS.md](../NON_GOALS.md) present — no paywalled data claims
- [x] SOTA execution plan archived (`docs/plans/SOTA_FINAL_EXECUTION.md`)

---

## Quick reference — all validation scripts

| Script | Report dir |
|--------|------------|
| `scripts/run_foundation_validation.py` | `reports/validation/foundation/` |
| `scripts/run_option_a_validation.py` | `reports/validation/option_a/` |
| `scripts/run_option_b_validation.py` | `reports/validation/option_b/` |
| `scripts/run_option_c_validation.py` | `reports/validation/option_c/` |
| `scripts/run_option_d_validation.py` | `reports/validation/option_d/` |
| `scripts/run_sota_validation.py` | `reports/validation/sota/` |

---

*Update completion dates in this file when closing each phase.*
