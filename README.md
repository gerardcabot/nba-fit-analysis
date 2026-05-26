# NBA Fit Analysis

Player–team fit modeling from public NBA stats (`nba_api`) and related free sources.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## CLI

### Foundation

```bash
python -m nba_fit health
python -m nba_fit fetch-sample --endpoint leaguedashplayerstats
python -m nba_fit ingest --season 2025-26
```

### Option A — interpretable fit index

```bash
python -m nba_fit rank-player 2544 --season 2025-26
python -m nba_fit rank-team 1610612747 --season 2025-26
python -m nba_fit rank-player 2544 --fit-card-team 1610612747
```

Add `--synthetic` for offline demos without cached ingest data.

### Option B — role embeddings and team need

Train player embeddings, GMM archetypes, team-need vectors, and nearest-neighbor comps:

```bash
python -m nba_fit train-roles --season 2025-26
```

Rank acquisition targets by **team need × role fit** (embedding alignment with roster gaps):

```bash
python -m nba_fit archetype-board 1610612747 --season 2025-26
```

Fit cards include `archetype`, `role_fit`, and `comps` (sklearn `NearestNeighbors` in embedding space; `k` from `nba_fit.models.constants.NEAREST_NEIGHBOR_K`) after `train-roles` has been run for that season.

### Option C — lineup-aware impact projection

Train ridge RAPM and a lineup net-rating model (requires Option B `train-roles` and `ingest --tier role` for live data):

```bash
python -m nba_fit train-impact --season 2025-26
```

Project top five-man units and a minutes-weighted `projected_net_rating_delta` for a player on a destination team:

```bash
python -m nba_fit lineup-sim 2544 1610612747 --season 2025-26
python -m nba_fit lineup-sim --synthetic
python -m nba_fit rank-player 2544 --fit-card-team 1610612747
```

Fit cards add `lineup_synergy` (top units + per-unit deltas) and top-level `projected_net_rating_delta` after Option A/B fields.

## Tests

```bash
pytest
python visual_tests/00_cache_smoke.py
python visual_tests/08_team_need_radar.py
python visual_tests/11_lineup_delta_bar.py
```
