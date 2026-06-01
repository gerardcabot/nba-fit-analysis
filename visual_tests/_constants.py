"""Shared constants for visual validation tests (aligned with probe scripts)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PROBE_RESULTS_JSON = REPO_ROOT / "probe_all_results.json"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

# Probe context (see probe_all_nba_endpoints.py and probe_all_results.json meta)
DEFAULT_SEASON = "2025-26"
# Option C / SOTA validation bundle when interim tables exist
VALIDATION_SEASON = "2024-25"
SOTA_VALIDATION_SEASON = "2024-25"
DEFAULT_PROBE_GAME_ID = "0022501198"
NBA_API_REQUEST_TIMEOUT_SEC = 30
NBA_API_PACING_DELAY_SEC = 0.6

# PlayByPlayV3 schema (probe_all_results.json → PlayByPlay dataset)
PBP_ACTION_TYPE_COLUMN = "actionType"
PBP_DATASET_INDEX = 0

# Plot display
TOP_ACTION_TYPES_TO_LABEL = 30
