# NBA Fit Analysis

Player–team fit modeling from public NBA stats (`nba_api`) and related free sources.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## CLI

```bash
python -m nba_fit health
python -m nba_fit fetch-sample --endpoint leaguedashplayerstats
```

## Tests

```bash
pytest
python visual_tests/00_cache_smoke.py
```
