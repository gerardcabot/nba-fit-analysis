# Annotated Bibliography — NBA Fit Analysis

References cited by [SOTA Completion Plan](plans/SOTA_COMPLETION_PLAN.md), Option B role taxonomy, and validation design. URLs verified as of 2026.

---

## Public impact metrics

### RAPTOR (FiveThirtyEight)

**What:** Regularized Adjusted Plus-Minus with box-score and luck-adjusted priors for player impact.  
**Relevance:** Benchmark for lineup-aware impact; our Option C RAPM ridge uses similar regularization philosophy.  
**URL:** https://github.com/fivethirtyeight/data/tree/master/nba-raptor

### LEBRON (BBall Index)

**What:** Luck-adjusted estimated plus-minus with box prior and role context.  
**Relevance:** Target calibration quality for projected impact; role taxonomy alignment via BBall Index offensive roles.  
**URL:** https://www.bball-index.com/lebron-database/

### DARKO (Kostya Medvedovsky)

**What:** Daily adjusted RAPM using on/off and box priors with recency weighting.  
**Relevance:** Recency and low-sample handling patterns for RAPM refresh cadence.  
**URL:** https://darko.app/

### EPM (Dunks & Threes)

**What:** Estimated plus-minus from tracking-informed model (public summaries).  
**Relevance:** Ensemble impact benchmark; documents what tracking adds beyond box/league-dash.  
**URL:** https://dunksandthrees.com/epm

---

## Role classification & play types

### BBall Index — offensive roles

**What:** Industry-standard offensive role taxonomy (Primary Ball Handler, Shot Creator, Movement Shooter, etc.).  
**Relevance:** Primary mapping target for `data/reference/archetype_industry_map.csv` and `role_taxonomy.py`.  
**URL:** https://www.bball-index.com/roles-database/

### Synergy Sports — play types

**What:** Event-tagging taxonomy (PNR handler, spot-up, transition, etc.) used league-wide.  
**Relevance:** Future feature source if play-type rates are proxied from public PBP; not fully available on leaguedash-only path.  
**URL:** https://synergysports.com/

### Cleaning the Glass (CTG)

**What:** Subscription analytics with position-adjusted roles and coarse Guard/Wing/Big groupings.  
**Relevance:** `ctg_position_group()` coarse buckets for team-need aggregation.  
**URL:** https://cleaningtheglass.com/

---

## Academic & research papers

### Wharton — player acquisition and team fit

**What:** Research on optimal roster construction and complementarity in NBA acquisitions.  
**Relevance:** Original project vision for movement backtest and interpretable fit decomposition.  
**URL:** https://wsp.wharton.upenn.edu/josephine/ (search: NBA player acquisition / roster optimization)

### ACM — role discovery in basketball (2025)

**What:** Recent work on unsupervised / semi-supervised role classification from tracking or box features.  
**Relevance:** Validates embedding + clustering approach in Option B; informs constrained clustering upgrade.  
**URL:** https://dl.acm.org/ (search: basketball player roles 2025)

### arXiv — lineup optimization / net rating prediction

**What:** Lineup-aware models predicting five-man unit performance from player embeddings.  
**Relevance:** Option C lineup simulation and projected net-rating delta design.  
**URL:** https://arxiv.org/search/?query=nba+lineup+net+rating&searchtype=all

### Deep RAPM / luck-adjusted RAPM literature

**What:** Regularized stint-level RAPM with opponent/home controls and Bayesian priors.  
**Relevance:** Tuning `RAPM_RIDGE_ALPHA`, possession thresholds, and low-sample flags in Option C.  
**URL:** https://arxiv.org/search/?query=regularized+adjusted+plus+minus+basketball

---

## Internal project docs

| Document | Purpose |
|----------|---------|
| [SOTA Completion Plan](plans/SOTA_COMPLETION_PLAN.md) | Roadmap from Option D MVP to research-grade system |
| [BENCHMARKS.md](BENCHMARKS.md) | External systems we compare against |
| [NON_GOALS.md](NON_GOALS.md) | Explicit out-of-scope items for free-data MVP |
| [NBA_ENDPOINTS_DATA_REFERENCE.md](NBA_ENDPOINTS_DATA_REFERENCE.md) | Public API coverage |
| [features/MISSING_PROXIES.md](../src/nba_fit/features/MISSING_PROXIES.md) | Feature gaps on leaguedash path |

---

## How to cite in code and docs

Use inline references like: *"Industry role labels follow BBall Index offensive taxonomy ([BIBLIOGRAPHY](BIBLIOGRAPHY.md#bball-index--offensive-roles))."*

When adding new constants, link the literature default and the validation hook (movement backtest, held-out season, or ablation).
