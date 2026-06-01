# Non-Goals

Explicit boundaries for NBA Fit Analysis. Keeps scope honest for a **free public-data** portfolio project.

---

## Data & infrastructure

- **Proprietary tracking** (Second Spectrum, SportVU) — no access; do not pretend to replicate EPM/EPV tracking features.
- **Paywalled CTG/Synergy tables** — CTG coarse groups are referenced for taxonomy only; no scraping or redistribution.
- **Real-time in-game feeds** — batch season ingest only.
- **Multi-sport expansion** — NBA only.

---

## Modeling

- **Single black-box fit score** — product requires decomposed submetrics, fit cards, and uncertainty bands.
- **Contract / cap-space optimization** — fit only; no CBA or salary matching.
- **Injury prediction or availability modeling** — out of scope unless added as explicit future phase.
- **Draft prospect evaluation** — requires different data and labels.

---

## Validation

- **Claiming FO-grade acquisition decisions** — research/portfolio quality, not operational front-office tooling.
- **Publishing player trade recommendations as financial advice** — rankings are analytical artifacts.
- **100% reproducibility of live API pulls** — cached interim Parquet is the source of truth for CI and reports.

---

## Product

- **Mobile-native app** — Streamlit dashboard + CLI suffice for MVP/SOTA.
- **User accounts / multi-tenant SaaS** — not planned.
- **Automated bot posting rankings** — human-reviewed exports only.

---

## When to revisit

Items move off this list only with new data access, explicit user request, and an updated [SOTA Completion Plan](plans/SOTA_COMPLETION_PLAN.md) ticket.
