# Prompt: SOTA Completion Plan (multi-subagent)

Copy everything below the line into a new Cursor chat. Use **parallel subagents** (one per workstream) plus a final **synthesis** pass. Output must be an **implementation plan in English** (not code yet unless explicitly asked).

---

## ORCHESTRATOR INSTRUCTIONS

You are the lead architect for **NBA Fit Analysis**. Your job is to produce a **research-grade completion plan** that moves the repo from **integrated MVP (Option D)** to **SOTA-quality player–team fit system**, aligned with the original product vision.

### Execution model

1. **Read first** (main agent or shared context):
   - Original vision: `.cursor/plans/nba_fit_project_aaa64a1e.plan.md` (full Option A→D spec, data sources, validation strategy, deliverables).
   - Current state: `README.md`, `docs/BRANCHING.md`, `docs/STORAGE.md`, `docs/NBA_ENDPOINTS_DATA_REFERENCE.md`.
   - **Ground truth codebase:** branch `stage/04-option-d` (or `main` after merges) — study `src/nba_fit/` end-to-end.
   - Validation artifacts: `reports/validation/{foundation,option_a,option_b,option_c,option_d}/` (metrics, RUN_LOG, figures).

2. **Spawn parallel subagents** (do not skip). Each subagent returns a **structured report** (markdown sections + file paths + priority P0/P1/P2). No vague advice.

3. **Synthesis agent** merges reports into one plan: phased roadmap, dependencies, estimated effort, risks, and explicit “definition of done” for SOTA.

---

## PROJECT CONTEXT (for all subagents)

### What this project is

**NBA Fit Analysis** estimates **player–team fit** from **free public data** (`nba_api`, optional `pbpstats`, etc.). It answers:

- For a player: which teams are the best **basketball fit**?
- For a team: which players are the best **acquisition targets**?

Outputs must be **interpretable** (decomposed submetrics, fit cards, uncertainty), not a single black-box score. Target quality bar: **research/portfolio SOTA** per the original plan — feature store, lineup-aware impact, movement backtests, calibrated rankings, dashboard.

Reference: `.cursor/plans/nba_fit_project_aaa64a1e.plan.md` for full intent (Options A–D, data registry, leakage rules, validation tracks, dashboard views).

**Repository:** https://github.com/gerardcabot/nba-fit-analysis

### Branch / PR structure (concatenated stages)

| PR | Branch | Layer |
|----|--------|--------|
| #1 | `stage/00-foundation` | Package, Parquet cache, endpoint health, CLI |
| #2 | `stage/01-option-a` | Interpretable fit index (player/team vectors, submetrics, rankings) |
| #3 | `stage/02-option-b` | Role embeddings, GMM archetypes, team need, archetype board |
| #4 | `stage/03-option-c` | PBP possessions, RAPM, lineup simulation, projected net rating delta |
| #5 | `stage/04-option-d` | Calibrated ensemble, uncertainty, movement backtest smoke, Streamlit dashboard |

Each PR builds on the previous branch. **Option D is the integrated product** but is still an **MVP integration**, not the final SOTA system described in the original plan.

### Known MVP limitations (do not ignore)

- Many **ensemble/submetric weights** are analyst priors in `scoring/constants.py`, not learned from data.
- **Archetype labels** are heuristic names from GMM centroids (`models/archetypes.py`) — not industry-standard NBA role taxonomy.
- **Movement backtest** often falls back to **synthetic** movements; real transaction pipeline incomplete.
- **Option C RAPM** validated on **30-game** PBP sample; low-sample flags everywhere.
- **Feature gaps** documented in `src/nba_fit/features/MISSING_PROXIES.md` (no touches, C&S, gravity, etc. on leaguedash-only path).
- Some inline coefficients still in code (e.g. `0.3`, `0.25` in `submetrics.py`) without named constants.
- `projected_impact` in ensemble uses `lineup_impact_fit` (0–1), not raw `projected_net_rating_delta` in points.

---

## SUBAGENT 1 — Gap analysis vs original plan (Option D as baseline)

**Branch/worktree:** `stage/04-option-d`

**Mission:** Compare **implemented** `stage/04-option-d` against `.cursor/plans/nba_fit_project_aaa64a1e.plan.md`.

Deliver:

1. **Coverage matrix:** plan section → implemented / partial / missing (with file paths).
2. **SOTA definition of done:** checklist derived from plan “Success Criteria” — what remains?
3. **Path from MVP → final:** ordered phases (data → features → models → validation → product) with merge strategy (single branch vs stacked PRs).
4. **External benchmarks:** what public systems/metrics should we align with (RAPTOR, DARKO, EPV, luck-adjusted RAPM literature, Wharton acquisition paper cited in plan, etc.) — cite sources.

---

## SUBAGENT 2 — Hardcoded / unproven constants audit

**Branch/worktree:** `stage/04-option-d`

**Mission:** Find every **hardcoded number, label, or text** that lacks empirical validation or tests.

Search:

- `src/nba_fit/features/constants.py`
- `src/nba_fit/models/constants.py`
- `src/nba_fit/scoring/constants.py`
- `src/nba_fit/config/settings.py`
- Inline literals in `submetrics.py`, `ensemble.py`, `archetypes.py`, `role_fit.py`, `lineup_fit.py`, `uncertainty.py`

Deliver:

1. **Inventory table:** symbol | value | location | stated rationale | evidence gap | proposed fix (learned / literature / A/B test / move to config).
2. **Archetype taxonomy deep-dive:** today’s heuristic labels (`high_usage_creator`, etc.) vs **NBA-industry role types** actually used in 2024–26 (front offices, Second Spectrum / Synergy-style play types where proxyable, Cleaning the Glass roles, CTG categories, public research taxonomies). Propose a **concrete replacement strategy** (mapping table, data source, clustering constraints).
3. **Weight learning proposal:** how to calibrate `SUBMETRIC_WEIGHTS` and `ENSEMBLE_COMPONENT_WEIGHTS` from movement backtest / held-out seasons (method + leakage guards).
4. **Test gaps:** which constants should have unit tests or golden-file regression tests?

---

## SUBAGENT 3 — Data & feature engineering SOTA

**Branch/worktree:** `stage/04-option-d`

**Mission:** Close gaps in `MISSING_PROXIES.md` and endpoint registry using **free data only**.

Deliver:

1. **Endpoint integration plan:** which `nba_api` / `pbpstats` endpoints to add per missing signal (passing, tracking bins, lineups, hustle, estimated metrics usage).
2. **Feature store schema:** target tables under `data/interim/` and `data/features/` for multi-season training.
3. **Leakage & temporal hygiene:** rules for pre-move features, season splits, transaction dates.
4. **Volume/ops:** full-season impact ingest (remove `--max-games` cap), caching, rate limits.

---

## SUBAGENT 4 — Modeling & validation SOTA

**Branch/worktree:** `stage/04-option-d`

**Mission:** Upgrade models from MVP to research-grade.

Deliver:

1. **RAPM / lineup:** possession thresholds, regularization tuning, opponent/home context, recency — cite literature defaults vs our `RAPM_RIDGE_ALPHA`, half-life.
2. **Movement backtest:** real labels from `transactions` fetcher / external CSV; evaluation metrics (calibration, deciles, replacement-player benchmark); selection-bias mitigation.
3. **Uncertainty:** replace heuristic bands with bootstrap or conformal plan; document assumptions.
4. **Ablation study design:** A only, A+B, A+B+C, full D — what to measure on 2024-25 holdout.

---

## SUBAGENT 5 — Bugs, code quality, product gaps

**Branch/worktree:** `stage/04-option-d` (+ skim open PR validation RUN_LOGs)

**Mission:** Find **bugs**, misleading UX, duplicate signals, and optional improvements.

Deliver:

1. **Bug list** (P0/P1/P2) with repro steps — include `fit_index` fallbacks to 0.5, synthetic context paths, sklearn warnings in logs, figure/script season mismatches if any remain.
2. **API/CLI consistency:** naming (`role_fit` vs `role_alignment` vs `team_need_fit`), fit card fields, dashboard vs CLI parity.
3. **CI/CD recommendations:** pytest markers, smoke ingest in CI (mocked), visual test policy.
4. **Optional enhancements:** contracts module, export rankings, model cards, notebook narratives.

---

## SUBAGENT 6 — Literature & industry scan (web allowed)

**Mission:** Web research for **2024–26 best practices** in player–team fit, role classification, and public-data limitations.

Deliver:

1. Short annotated bibliography (papers, blogs, open repos) relevant to our plan’s Option D.
2. **Archetype / role standards** used in practice or research (names + definitions).
3. What true SOTA systems do that we do not (and whether achievable with free data).

---

## SYNTHESIS AGENT (after all subagents return)

Produce **`docs/plans/SOTA_COMPLETION_PLAN.md`** in English with:

1. **Executive summary** (1 page): MVP today vs SOTA target.
2. **Prioritized backlog** (P0 must-ship, P1 should-ship, P2 nice-to-have) — each item: problem, solution, files touched, dependency, validation test.
3. **Implementation phases** (4–8 weeks style milestones, no fake dates required).
4. **Archetype replacement proposal** (final recommendation from Subagent 2+6).
5. **Empirical calibration plan** for all major weights.
6. **Final validation protocol** (movement, held-out season, lineup, calibration plots).
7. **Risks & non-goals** (what free data cannot support).

**Constraints for the plan:**

- Stay aligned with original plan vision (interpretable fit card, lineup impact, movement validation).
- Prefer **free/public data**; flag paywalled sources separately.
- No “magic numbers” in new design — every constant needs rationale + validation hook.
- Assume work continues on `stage/04-option-d` or `main` after PR stack merges.

---

## OUTPUT FORMAT (strict)

Each subagent:

```markdown
## Subagent: [name]
### Findings
### Evidence (paths, logs, metrics)
### Recommendations (P0/P1/P2)
### Suggested implementation tickets (title + 2-line scope)
```

Synthesis:

```markdown
# NBA Fit Analysis — SOTA Completion Plan
## 1. Current state
## 2. Gaps vs original vision
## 3. Hardcoded / unproven parameters
## 4. Archetype & role taxonomy upgrade
## 5. Data & features
## 6. Modeling & validation
## 7. Bugs & engineering
## 8. Phased roadmap
## 9. Definition of done (SOTA)
```

---

## NOTES FOR THE HUMAN

- Run subagents in **parallel**; synthesis only after all complete.
- Use **git worktrees** per branch if needed (`stage/04-option-d` is the primary study target).
- Do **not** implement code in this pass unless asked — **plan only**.
- Plan language: **English**.
