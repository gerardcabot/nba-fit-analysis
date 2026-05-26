# Visual validation tests

Charts that sanity-check probe output and live `nba_api` play-by-play schema. Figures are written to `reports/figures/` (gitignored; regenerate anytime).

## Prerequisites

```powershell
cd <repo-root>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For `01_endpoint_health.py`, ensure `probe_all_results.json` exists at the repo root (from `python probe_all_nba_endpoints.py`).

For `02_pbp_schema.py`, `nba_api` must be installed and stats.nba.com must be reachable.

## Run all tests

**Windows (PowerShell):**

```powershell
.\scripts\run_visual_tests.ps1
```

**Make (Git Bash / WSL / macOS / Linux):**

```bash
make visual-tests
```

## Run individually

From the repo root (so `visual_tests` imports resolve):

```powershell
$env:PYTHONPATH = (Get-Location).Path
python visual_tests/01_endpoint_health.py
python visual_tests/02_pbp_schema.py
```

## Outputs

| Script | Figure |
|--------|--------|
| `01_endpoint_health.py` | `reports/figures/01_endpoint_health.png` |
| `02_pbp_schema.py` | `reports/figures/02_pbp_action_type_distribution.png` |

Shared styling lives in `_plot_utils.py`; probe-aligned IDs and paths are in `_constants.py`.
