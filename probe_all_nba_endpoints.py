#!/usr/bin/env python3
"""
Probe every nba_api endpoint class and emit JSON + Markdown reference.

Covers:
  - nba_api.stats.endpoints (138 exported)
  - nba_api.stats.endpoints.videoeventsasset (module exists, not in __all__)
  - nba_api.live.nba.endpoints (4 live CDN endpoints)

Run: python probe_all_nba_endpoints.py
Outputs:
  - probe_all_results.json
  - docs/NBA_ENDPOINTS_DATA_REFERENCE.md
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Shared context for parameter injection
# ---------------------------------------------------------------------------

SEASON = "2025-26"
PLAYER_ID = 1629029  # Luka Doncic
TEAM_ID = 1610612747  # LAL (2025-26)
GAME_ID = "0022501198"
LEAGUE_ID = "00"
VS_PLAYER_ID = 2544  # LeBron James
COLLEGE = "University of Kentucky"
GAME_DATE = "2025-10-21"
LOCATION = "Los Angeles"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

_backoff = 0.7

# Manual overrides for endpoints that need non-obvious parameter combos
ENDPOINT_KWARGS: dict[str, dict[str, Any]] = {
    "shotchartdetail": {
        "team_id": TEAM_ID,
        "player_id": PLAYER_ID,
        "season_nullable": SEASON,
        "context_measure_simple": "FGA",
    },
    "shotchartlineupdetail": {
        "team_id": TEAM_ID,
        "player_id": PLAYER_ID,
        "season_nullable": SEASON,
        "game_id_nullable": GAME_ID,
    },
    "synergyplaytypes": {
        "season": SEASON,
        "player_or_team_abbreviation": "P",
    },
    "playervsplayer": {
        "player_id": PLAYER_ID,
        "vs_player_id": VS_PLAYER_ID,
        "season": SEASON,
    },
    "teamandplayersvsplayers": {
        "team_id": TEAM_ID,
        "player_id": PLAYER_ID,
        "vs_player_id": VS_PLAYER_ID,
        "season": SEASON,
    },
    "teamvsplayer": {
        "team_id": TEAM_ID,
        "vs_player_id": VS_PLAYER_ID,
        "season": SEASON,
    },
    "playercompare": {
        "player_id_list": str(PLAYER_ID),
        "vs_player_id_list": str(VS_PLAYER_ID),
        "season": SEASON,
    },
    "cumestatsplayer": {
        "player_id": PLAYER_ID,
        "season": SEASON,
        "game_ids": GAME_ID,
    },
    "cumestatsplayergames": {
        "player_id": PLAYER_ID,
        "season": SEASON,
        "game_ids": GAME_ID,
    },
    "cumestatsteam": {
        "team_id": TEAM_ID,
        "season": SEASON,
        "game_ids": GAME_ID,
    },
    "cumestatsteamgames": {
        "team_id": TEAM_ID,
        "season": SEASON,
        "game_ids": GAME_ID,
    },
    "glalumboxscoresimilarityscore": {
        "person_id": PLAYER_ID,
        "season": SEASON,
        "game_id": GAME_ID,
    },
    "infographicfanduelplayer": {"player_id": PLAYER_ID},
    "playerfantasyprofilebargraph": {"player_id": PLAYER_ID},
    "videodetails": {
        "game_event_id": 1,
        "game_id": GAME_ID,
        "season": SEASON,
        "team_id": TEAM_ID,
    },
    "videodetailsasset": {
        "game_event_id": 1,
        "game_id": GAME_ID,
        "season": SEASON,
        "team_id": TEAM_ID,
    },
    "videoevents": {
        "game_event_id": 1,
        "game_id": GAME_ID,
        "season": SEASON,
        "team_id": TEAM_ID,
    },
    "videoeventsasset": {
        "game_event_id": 1,
        "game_id": GAME_ID,
        "season": SEASON,
        "team_id": TEAM_ID,
    },
    "videostatus": {"game_date": GAME_DATE, "league_id": LEAGUE_ID},
    "winprobabilitypbp": {"game_id": GAME_ID},
    "gamerotation": {"game_id": GAME_ID},
    "hustlestatsboxscore": {"game_id": GAME_ID},
    "playbyplayv3": {"game_id": GAME_ID},
    "playbyplayv2": {"game_id": GAME_ID},
    "playbyplay": {"game_id": GAME_ID},
    "scoreboardv3": {"game_date": GAME_DATE, "league_id": LEAGUE_ID},
    "scoreboardv2": {"game_date": GAME_DATE, "day_offset": 0, "league_id": LEAGUE_ID},
    "commonteamroster": {"team_id": TEAM_ID, "season": SEASON},
    "commonteamyears": {"league_id": LEAGUE_ID},
    "commonplayoffseries": {"season": SEASON},
    "playercompare": {
        "player_id_list": str(PLAYER_ID),
        "vs_player_id_list": str(VS_PLAYER_ID),
        "season": SEASON,
    },
    "playergamestreakfinder": {"league_id_nullable": LEAGUE_ID},
    "teamgamestreakfinder": {"league_id_nullable": LEAGUE_ID},
    "teaminfocommon": {"team_id": TEAM_ID, "season_nullable": SEASON},
    "teamyearbyyearstats": {"team_id": TEAM_ID, "season_type": "Regular Season", "league_id": LEAGUE_ID},
    "playoffpicture": {"league_id": LEAGUE_ID},
    "leaguelineupviz": {"season": SEASON, "team_id": TEAM_ID, "league_id": LEAGUE_ID},
    "leagueplayerondetails": {"player_id": PLAYER_ID, "season": SEASON, "team_id": TEAM_ID, "league_id": LEAGUE_ID},
    "leagueseasonmatchups": {"season": SEASON, "def_player_id": VS_PLAYER_ID, "off_player_id": PLAYER_ID, "league_id": LEAGUE_ID},
    "matchupsrollup": {"season": SEASON, "player_id": PLAYER_ID, "league_id": LEAGUE_ID},
    "teamandplayersvsplayers": {"team_id": TEAM_ID, "player_id": PLAYER_ID, "vs_player_id": VS_PLAYER_ID, "season": SEASON, "league_id": LEAGUE_ID},
    "playercareerbycollegerollup": {"college": COLLEGE, "league_id": LEAGUE_ID},
    "draftcombinestats": {"league_id": LEAGUE_ID, "season_year": "2024-25"},
    "draftcombinedrillresults": {"league_id": LEAGUE_ID, "season_year": "2024-25"},
    "draftcombineplayeranthro": {"league_id": LEAGUE_ID, "season_year": "2024-25"},
    "draftcombinespotshooting": {"league_id": LEAGUE_ID, "season_year": "2024-25"},
    "draftcombinenonstationaryshooting": {"league_id": LEAGUE_ID, "season_year": "2024-25"},
    "cumestatsplayergames": {"player_id": PLAYER_ID, "season": SEASON, "game_ids": GAME_ID, "league_id": LEAGUE_ID},
    "cumestatsteamgames": {"team_id": TEAM_ID, "season": SEASON, "game_ids": GAME_ID, "league_id": LEAGUE_ID},
    "glalumboxscoresimilarityscore": {"person_id": PLAYER_ID, "season": SEASON, "game_id": GAME_ID, "league_id": LEAGUE_ID},
    "infographicfanduelplayer": {"player_id": PLAYER_ID, "league_id": LEAGUE_ID},
    "videoevents": {"game_event_id": 1, "game_id": GAME_ID, "team_id": TEAM_ID, "league_id": LEAGUE_ID},
    "videoeventsasset": {"game_id": GAME_ID, "game_event_id": 1},
    "scheduleleaguev2int": {"season": SEASON, "league_id": LEAGUE_ID},
    "assisttracker": {"league_id_nullable": LEAGUE_ID},
    "draftboard": {"league_id": LEAGUE_ID, "season": SEASON},
    "shotchartlineupdetail": {
        "team_id": TEAM_ID,
        "player_id": PLAYER_ID,
        "game_id_nullable": GAME_ID,
        "season_nullable": SEASON,
        "context_measure_simple": "FGA",
    },
    "commonplayerinfo": {"player_id": PLAYER_ID},
    "franchisehistory": {"league_id": LEAGUE_ID},
    "franchiseleaders": {"team_id": TEAM_ID},
    "franchiseplayers": {"team_id": TEAM_ID, "league_id": LEAGUE_ID},
    "teaminfocommon": {"team_id": TEAM_ID, "season": SEASON},
    "teamdetails": {"team_id": TEAM_ID},
    "teamhistoricalleaders": {"team_id": TEAM_ID},
    "teamyearbyyearstats": {"team_id": TEAM_ID, "season_type": "Regular Season"},
    "playercareerbycollege": {"college": COLLEGE},
    "playercareerbycollegerollup": {"college": COLLEGE},
    "draftboard": {"season": SEASON, "league_id": LEAGUE_ID},
    "draftcombinestats": {"season": "2024-25"},
    "draftcombinedrillresults": {"season": "2024-25"},
    "draftcombineplayeranthro": {"season": "2024-25"},
    "draftcombinespotshooting": {"season": "2024-25"},
    "draftcombinenonstationaryshooting": {"season": "2024-25"},
    "drafthistory": {"league_id": LEAGUE_ID},
    "playernextngames": {"player_id": PLAYER_ID},
    "playerawards": {"player_id": PLAYER_ID},
    "playercareerstats": {"player_id": PLAYER_ID},
    "playerprofilev2": {"player_id": PLAYER_ID},
    "playergamelog": {
        "player_id": PLAYER_ID,
        "season": SEASON,
        "season_type_all_star": "Regular Season",
    },
    "teamgamelog": {
        "team_id": TEAM_ID,
        "season": SEASON,
        "season_type_all_star": "Regular Season",
    },
    "playergamestreakfinder": {"player_id": PLAYER_ID, "season": SEASON},
    "teamgamestreakfinder": {"team_id": TEAM_ID, "season": SEASON},
    "leagueplayerondetails": {
        "player_id": PLAYER_ID,
        "season": SEASON,
        "team_id": TEAM_ID,
    },
    "leagueseasonmatchups": {
        "season": SEASON,
        "def_player_id": VS_PLAYER_ID,
        "off_player_id": PLAYER_ID,
    },
    "matchupsrollup": {"season": SEASON, "player_id": PLAYER_ID},
    "leaguelineupviz": {"season": SEASON, "team_id": TEAM_ID},
    "leaguedashlineups": {"season": SEASON, "team_id_nullable": str(TEAM_ID)},
    "teamdashlineups": {"team_id": TEAM_ID, "season": SEASON},
    "scheduleleaguev2int": {"season": SEASON, "league_id": "00"},
    "scheduleleaguev2": {"season": SEASON, "league_id": "00"},
    "iststandings": {"season": SEASON, "league_id": LEAGUE_ID},
    "playoffpicture": {"league_id": LEAGUE_ID, "season": SEASON},
    "defensehub": {"season": SEASON},
    "homepagev2": {},
    "homepageleaders": {},
    "leaderstiles": {},
    "alltimeleadersgrids": {"league_id": LEAGUE_ID},
    "assistleaders": {"season": SEASON, "league_id": LEAGUE_ID},
    "assisttracker": {"league_id": LEAGUE_ID, "season": SEASON},
    "commonteamyears": {"league_id": LEAGUE_ID, "season": SEASON},
    "commonplayoffseries": {"season": SEASON},
    "draftboard": {"league_id": LEAGUE_ID, "season": SEASON},
    "draftcombinestats": {"league_id": LEAGUE_ID, "season_year": "2024-25"},
    "draftcombinedrillresults": {"league_id": LEAGUE_ID, "season_year": "2024-25"},
    "draftcombineplayeranthro": {"league_id": LEAGUE_ID, "season_year": "2024-25"},
    "draftcombinespotshooting": {"league_id": LEAGUE_ID, "season_year": "2024-25"},
    "draftcombinenonstationaryshooting": {"league_id": LEAGUE_ID, "season_year": "2024-25"},
    "teaminfocommon": {"team_id": TEAM_ID, "season_nullable": SEASON},
    "teamyearbyyearstats": {"team_id": TEAM_ID, "season_type": "Regular Season", "league_id": LEAGUE_ID},
    "playoffpicture": {"league_id": LEAGUE_ID},
    "leaguelineupviz": {"season": SEASON, "team_id": TEAM_ID, "league_id": LEAGUE_ID},
    "leagueplayerondetails": {"player_id": PLAYER_ID, "season": SEASON, "team_id": TEAM_ID, "league_id": LEAGUE_ID},
    "leagueseasonmatchups": {"season": SEASON, "def_player_id": VS_PLAYER_ID, "off_player_id": PLAYER_ID, "league_id": LEAGUE_ID},
    "matchupsrollup": {"season": SEASON, "player_id": PLAYER_ID, "league_id": LEAGUE_ID},
    "playercompare": {"vs_player_id_list": VS_PLAYER_ID, "player_id_list": PLAYER_ID, "season": SEASON},
    "playergamestreakfinder": {"league_id_nullable": LEAGUE_ID},
    "teamgamestreakfinder": {"league_id_nullable": LEAGUE_ID},
    "teamandplayersvsplayers": {"team_id": TEAM_ID, "player_id": PLAYER_ID, "vs_player_id": VS_PLAYER_ID, "season": SEASON, "league_id": LEAGUE_ID},
    "playercareerbycollegerollup": {"college": COLLEGE, "league_id": LEAGUE_ID},
    "shotchartlineupdetail": {"team_id": TEAM_ID, "player_id": PLAYER_ID, "game_id_nullable": GAME_ID, "season_nullable": SEASON, "context_measure_simple": "FGA"},
    "cumestatsplayergames": {"player_id": PLAYER_ID, "season": SEASON, "game_ids": GAME_ID, "league_id": LEAGUE_ID},
    "cumestatsteamgames": {"team_id": TEAM_ID, "season": SEASON, "game_ids": GAME_ID, "league_id": LEAGUE_ID},
    "glalumboxscoresimilarityscore": {"person_id": PLAYER_ID, "season": SEASON, "game_id": GAME_ID, "league_id": LEAGUE_ID},
    "infographicfanduelplayer": {"player_id": PLAYER_ID, "league_id": LEAGUE_ID},
    "videodetails": {"game_event_id": 1, "game_id": GAME_ID, "season": SEASON, "team_id": TEAM_ID, "league_id": LEAGUE_ID},
    "videodetailsasset": {"game_event_id": 1, "game_id": GAME_ID, "season": SEASON, "team_id": TEAM_ID, "league_id": LEAGUE_ID},
    "videoevents": {"game_event_id": 1, "game_id": GAME_ID, "team_id": TEAM_ID, "league_id": LEAGUE_ID},
    "fantasywidget": {"season": SEASON},
    "dunkscoreleaders": {"season": SEASON},
    "gravityleaders": {"season": SEASON},
}

# Param name -> context value resolver
PARAM_VALUES: dict[str, Any] = {
    "game_id": GAME_ID,
    "game_id_nullable": GAME_ID,
    "player_id": PLAYER_ID,
    "person_id": PLAYER_ID,
    "player_id_nullable": PLAYER_ID,
    "team_id": TEAM_ID,
    "team_id_nullable": str(TEAM_ID),
    "vs_player_id": VS_PLAYER_ID,
    "def_player_id": VS_PLAYER_ID,
    "off_player_id": PLAYER_ID,
    "season": SEASON,
    "season_nullable": SEASON,
    "season_year": SEASON,
    "league_id": LEAGUE_ID,
    "league_id_nullable": LEAGUE_ID,
    "game_date": GAME_DATE,
    "game_event_id": 1,
    "college": COLLEGE,
    "location": LOCATION,
    "player_or_team_abbreviation": "P",
    "player_or_team": "P",
    "is_only_current_season": 1,
    "season_type_all_star": "Regular Season",
    "season_type": "Regular Season",
    "per_mode_simple": "PerGame",
    "per_mode_detailed": "PerGame",
    "measure_type_detailed_defense": "Base",
    "measure_type_simple": "Base",
    "context_measure_simple": "FGA",
    "month": 0,
    "day_offset": 0,
    "last_n_games": 0,
    "period": 0,
    "series_id": "0042200101",
    "game_ids": GAME_ID,
    "player_id_list": f"{PLAYER_ID},{VS_PLAYER_ID}",
}


def _patch_headers() -> None:
    from nba_api.stats.library.http import NBAStatsHTTP

    NBAStatsHTTP.headers = {**NBAStatsHTTP.headers, **BROWSER_HEADERS}


def _sleep() -> None:
    global _backoff
    time.sleep(_backoff)
    _backoff = min(_backoff * 1.08, 3.5)


def _flatten_columns(cols: Any) -> list[str]:
    if isinstance(cols, pd.MultiIndex):
        return [
            " | ".join(str(p) for p in tup if str(p) not in ("", "nan"))
            for tup in cols
        ]
    return [str(c) for c in cols]


def _resolve_kwargs(endpoint_name: str, cls: type) -> dict[str, Any]:
    if endpoint_name in ENDPOINT_KWARGS:
        return {**ENDPOINT_KWARGS[endpoint_name], "timeout": 45}

    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {"timeout": 45}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "proxy", "headers", "get_request"):
            continue
        if param_name in PARAM_VALUES:
            kwargs[param_name] = PARAM_VALUES[param_name]
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        # required without default — best-effort guess
        lower = param_name.lower()
        if "game" in lower and "id" in lower:
            kwargs[param_name] = GAME_ID
        elif "player" in lower and "id" in lower:
            kwargs[param_name] = PLAYER_ID
        elif "team" in lower and "id" in lower:
            kwargs[param_name] = TEAM_ID
        elif "season" in lower:
            kwargs[param_name] = SEASON
        elif "league" in lower:
            kwargs[param_name] = LEAGUE_ID
    return kwargs


def _summarize_endpoint(ep: Any) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    total_rows = 0
    frames = ep.get_data_frames()

    ds_names: list[str] = []
    if ep.nba_response is not None:
        try:
            raw = ep.nba_response.get_data_sets(getattr(ep, "endpoint", None))
            ds_names = list(raw.keys())
        except Exception:  # noqa: BLE001
            try:
                ds_names = list(ep.nba_response.get_data_sets().keys())
            except Exception:  # noqa: BLE001
                ds_names = []

    if ds_names and len(ds_names) == len(frames):
        pairs = list(zip(ds_names, frames))
    else:
        pairs = [(f"dataset_{i}", df) for i, df in enumerate(frames)]

    for name, df in pairs:
        cols = _flatten_columns(df.columns)
        rows = len(df)
        total_rows += rows
        sample = df.head(1).to_dict(orient="records") if rows else []
        clean_sample = []
        for row in sample:
            clean_sample.append({str(k): json.loads(json.dumps(v, default=str)) for k, v in row.items()})
        datasets[name] = {"rows": rows, "columns": cols, "sample": clean_sample}

    return {"total_rows": total_rows, "datasets": datasets}


def _expected_schema(cls: type) -> dict[str, list[str]]:
    expected = getattr(cls, "expected_data", None)
    if not expected:
        return {}
    out: dict[str, list[str]] = {}
    for ds_name, cols in expected.items():
        if isinstance(cols, list):
            flat = []
            for c in cols:
                if isinstance(c, str):
                    flat.append(c)
                elif isinstance(c, dict) and "columnNames" in c:
                    flat.extend(c["columnNames"])
            out[ds_name] = flat
        else:
            out[ds_name] = []
    return out


def _get_endpoint_class(endpoint_name: str, *, api_family: str = "stats") -> type | None:
    from nba_api.stats.endpoints._base import Endpoint as StatsEndpoint

    if api_family == "live":
        from nba_api.live.nba.endpoints._base import Endpoint as LiveEndpoint

        live_map = {
            "playbyplay": "nba_api.live.nba.endpoints.playbyplay",
            "boxscore": "nba_api.live.nba.endpoints.boxscore",
            "scoreboard": "nba_api.live.nba.endpoints.scoreboard",
            "odds": "nba_api.live.nba.endpoints.odds",
        }
        mod_path = live_map.get(endpoint_name)
        if not mod_path:
            return None
        module = importlib.import_module(mod_path)
        base = LiveEndpoint
    else:
        try:
            module = importlib.import_module(f"nba_api.stats.endpoints.{endpoint_name}")
        except ModuleNotFoundError:
            return None
        base = StatsEndpoint

    if module is None:
        return None
    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, base) and obj is not base
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        # Prefer class whose endpoint attr matches module name
        for c in candidates:
            if getattr(c, "endpoint", None) == endpoint_name:
                return c
        return candidates[0]
    return None


EMPTY_REASONS: dict[str, str] = {
    "synergyplaytypes": "Synergy play-type feed returned zero rows (NBA no longer populates this table for recent seasons).",
    "playernextngames": "No upcoming scheduled games in the probe window for this player.",
    "playerdashptshotdefend": "No defended-shot tracking rows for player/season filter (often sparse or gated).",
    "playerfantasyprofilebargraph": "Fantasy bar-graph widget empty for this player ID.",
    "cumestatsplayer": "Cume stats require specific game_id list; empty when subset has no qualifying games.",
}


def infer_failure_reason(result: dict[str, Any]) -> str | None:
    """One-line human explanation for FAIL or OK-empty outcomes."""
    ep = result["endpoint"]
    if result.get("success") and result.get("has_data"):
        return None
    if result.get("success") and not result.get("has_data"):
        return EMPTY_REASONS.get(
            ep,
            "HTTP succeeded but every dataset was empty for the probe parameters used.",
        )

    err = (result.get("error") or "").lower()
    api = result.get("api_family", "stats")

    if ep == "scheduleleaguev2int" and "could not resolve" in err:
        return "In `__all__` but missing from package `__init__` imports (use `importlib` submodule load)."
    if "could not resolve endpoint class" in err:
        return "Python module exists but no Endpoint subclass could be resolved."
    if "keyerror" in err and "resultset" in err:
        return "Deprecated V1/V2 stats endpoint — NBA returns JSON without `resultSet` (use V3 replacement)."
    if "jsondecodeerror" in err:
        return "Empty or non-JSON HTTP body (endpoint deprecated, blocked, or wrong URL)."
    if "unexpected keyword argument" in err:
        return "Auto-probe used wrong constructor parameters; endpoint needs manual kwargs."
    if "missing" in err and "required" in err:
        return "Required constructor argument missing from auto-probe mapping."
    if "timeout" in err or "timed out" in err:
        return "Read timeout — stats.nba.com slow or throttling the client."
    if "connection" in err or "remotedisconnected" in err or "connection aborted" in err:
        return "TCP connection dropped by NBA servers (rate limit or transient network)."
    if "403" in err or "forbidden" in err:
        return "HTTP 403 — blocked without valid browser headers or IP throttled."
    if "404" in err or "not found" in err:
        return "HTTP 404 — resource not on CDN/API for the game_id or date used."
    if api == "live":
        return "Live CDN JSON unavailable for this game_id (only recent/today games are hosted on cdn.nba.com)."
    if "typeerror" in err and "keys must be str" in err:
        return "MultiIndex columns broke JSON serialization (endpoint works; nested column headers)."
    return f"Unhandled error: {result.get('error', 'unknown')[:120]}"


def _summarize_live_endpoint(ep: Any) -> dict[str, Any]:
    """Summarize live CDN endpoints that expose get_dict() instead of DataFrames."""
    data = ep.get_dict()
    datasets: dict[str, Any] = {}

    def _walk(obj: Any, prefix: str = "", depth: int = 0) -> None:
        if depth > 2:
            return
        if isinstance(obj, dict):
            for key, val in obj.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    cols = list(val[0].keys())
                    datasets[path] = {
                        "rows": len(val),
                        "columns": cols,
                        "sample": [val[0]],
                    }
                elif isinstance(val, dict) and val:
                    _walk(val, path, depth + 1)

    _walk(data)
    total = sum(d.get("rows", 0) for d in datasets.values())
    if not datasets and data:
        datasets["root"] = {
            "rows": 1,
            "columns": list(data.keys()) if isinstance(data, dict) else [],
            "sample": [data if isinstance(data, dict) else {"raw": str(data)[:200]}],
        }
        total = 1
    return {"total_rows": total, "datasets": datasets}


def probe_one(
    endpoint_name: str,
    *,
    api_family: str = "stats",
) -> dict[str, Any]:
    cls = _get_endpoint_class(endpoint_name, api_family=api_family)
    if cls is None:
        result = {
            "endpoint": endpoint_name,
            "api_family": api_family,
            "success": False,
            "error": f"Could not resolve Endpoint class for {endpoint_name}",
        }
        result["failure_reason"] = infer_failure_reason(result)
        return result

    kwargs = _resolve_kwargs(endpoint_name, cls) if api_family == "stats" else _resolve_live_kwargs(endpoint_name, cls)
    schema = _expected_schema(cls) if api_family == "stats" else {}

    if api_family == "stats":
        _patch_headers()
    _sleep()
    start = time.time()
    try:
        ep = cls(**kwargs)
        elapsed = time.time() - start
        if api_family == "live":
            summary = _summarize_live_endpoint(ep)
            api_path = getattr(cls, "endpoint_url", endpoint_name)
        else:
            summary = _summarize_endpoint(ep)
            api_path = getattr(cls, "endpoint", endpoint_name)
        result = {
            "endpoint": endpoint_name,
            "api_family": api_family,
            "class": cls.__name__,
            "api_endpoint": api_path,
            "success": True,
            "elapsed_sec": round(elapsed, 2),
            "kwargs_used": {k: v for k, v in kwargs.items() if k != "timeout"},
            "total_rows": summary["total_rows"],
            "datasets": summary["datasets"],
            "expected_schema": schema,
            "has_data": summary["total_rows"] > 0,
        }
        result["failure_reason"] = infer_failure_reason(result)
        return result
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - start
        result = {
            "endpoint": endpoint_name,
            "api_family": api_family,
            "class": cls.__name__ if cls else None,
            "api_endpoint": getattr(cls, "endpoint", endpoint_name) if cls else endpoint_name,
            "success": False,
            "elapsed_sec": round(elapsed, 2),
            "kwargs_used": {k: v for k, v in kwargs.items() if k != "timeout"},
            "expected_schema": schema,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-500:],
        }
        result["failure_reason"] = infer_failure_reason(result)
        return result


def _resolve_live_kwargs(endpoint_name: str, cls: type) -> dict[str, Any]:
    if endpoint_name in ("playbyplay", "boxscore"):
        return {"game_id": GAME_ID, "timeout": 45}
    return {"timeout": 45}


def _categorize(name: str, api_family: str = "stats") -> str:
    if api_family == "live":
        return "Live CDN (cdn.nba.com)"
    n = name.lower()
    # Specific names before broad substring rules
    if n in ("iststandings",):
        return "In-Season Tournament"
    if n.startswith("assist") or n in ("leaderstiles", "homepageleaders", "alltimeleadersgrids"):
        return "Leaders & Records"
    rules = [
        ("boxscore", "Box Score (game-level)"),
        ("playbyplay", "Play-by-Play"),
        ("shotchart", "Shot Charts"),
        ("playerdashpt", "Player Tracking Dashboards"),
        ("playerdashboard", "Player Dashboards"),
        ("teamdashpt", "Team Tracking Dashboards"),
        ("teamdashlineups", "Team Lineups"),
        ("teamdashboard", "Team Dashboards"),
        ("teamplayeronoff", "On/Off & Lineup Impact"),
        ("leaguedash", "League Dash (season aggregates)"),
        ("leaguegame", "League & Schedule"),
        ("leaguegamelog", "League & Schedule"),
        ("leaguestandings", "League & Schedule"),
        ("leaguehustle", "League & Schedule"),
        ("leagueleaders", "League & Schedule"),
        ("leaguelineup", "League & Schedule"),
        ("leagueplayer", "League & Schedule"),
        ("leagueseason", "League & Schedule"),
        ("common", "Common / Metadata"),
        ("playercareer", "Player Profiles & Logs"),
        ("playerprofile", "Player Profiles & Logs"),
        ("playergamelog", "Player Profiles & Logs"),
        ("playergamestreak", "Player Profiles & Logs"),
        ("playercompare", "Player Profiles & Logs"),
        ("playervsplayer", "Player Profiles & Logs"),
        ("playerindex", "Player Profiles & Logs"),
        ("playerawards", "Player Profiles & Logs"),
        ("playernext", "Player Profiles & Logs"),
        ("playerestimated", "Player Profiles & Logs"),
        ("playerfantasy", "Player Profiles & Logs"),
        ("teamgamelog", "Team Profiles & Logs"),
        ("teamgamestreak", "Team Profiles & Logs"),
        ("teamhistorical", "Team Profiles & Logs"),
        ("teamyearbyyear", "Team Profiles & Logs"),
        ("teaminfocommon", "Team Profiles & Logs"),
        ("teamdetails", "Team Profiles & Logs"),
        ("teamplayerdashboard", "Team Profiles & Logs"),
        ("teamvsplayer", "Team Profiles & Logs"),
        ("draft", "Draft & Combine"),
        ("video", "Video"),
        ("scoreboard", "Scoreboard"),
        ("schedule", "Schedule"),
        ("synergy", "Synergy Play Types"),
        ("cumestats", "Cume Stats (game subsets)"),
        ("homepage", "Homepage / Widgets"),
        ("fantasy", "Fantasy"),
        ("playoff", "Playoffs"),
        ("defensehub", "Defense"),
        ("hustle", "Hustle"),
        ("gravity", "Gravity / Advanced"),
        ("dunk", "Dunk Score"),
        ("matchupsrollup", "Matchups"),
        ("winprobability", "Win Probability"),
        ("gamerotation", "Game Rotation"),
        ("franchise", "Franchise History"),
    ]
    for key, cat in rules:
        if key in n:
            return cat
    return "Other"


def _describe_columns(cols: list[str]) -> str:
    if not cols:
        return "_No columns returned._"
    if len(cols) <= 12:
        return ", ".join(f"`{c}`" for c in cols)
    head = ", ".join(f"`{c}`" for c in cols[:10])
    return f"{head}, ... (+{len(cols) - 10} more)"


def generate_markdown(results: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines: list[str] = [
        "# NBA API Endpoints — Data Reference",
        "",
        f"> Auto-generated by `probe_all_nba_endpoints.py` on {meta['timestamp']}.",
        f"> Season context: **{meta['season']}** | Sample player: **Luka Dončić ({meta['player_id']})** | "
        f"Team: **LAL ({meta['team_id']})** | Game: **{meta['game_id']}**",
        "",
        "## What Was Tested",
        "",
        f"This document covers **all {meta['total']} endpoints** in `nba_api`:",
        "- **138** exported by `nba_api.stats.endpoints` ([docs](https://github.com/swar/nba_api/tree/master/docs/nba_api/stats/endpoints))",
        "- **1** extra stats module (`videoeventsasset`, not in `__all__`)",
        "- **4** live CDN endpoints (`nba_api.live.nba.endpoints`)",
        "",
        "Each endpoint was called with browser-like headers and exponential backoff.",
        "",
        "**Prior work:** 11 tactical endpoints were probed separately in `probe_nba_tactical_endpoints.py`.",
        "This reference supersedes that subset with full coverage.",
        "",
        "## Data Type Taxonomy",
        "",
        "Endpoints fall into six conceptual layers relevant to `nba-fit-analysis`:",
        "",
        "| Layer | What you get | Example endpoints | Continuous tracking? |",
        "|-------|----------------|-------------------|----------------------|",
        "| **Event stream** | Discrete play-by-play events with clock, actors, descriptions | `playbyplayv3`, `winprobabilitypbp` | No — events only |",
        "| **Game box slices** | Single-game traditional/advanced/usage/hustle/tracking stats | `boxscoretraditionalv3`, `boxscoreplayertrackv3`, `boxscorematchupsv3` | Aggregated per game |",
        "| **Season league dash** | Full-season tables for all players/teams with filters | `leaguedashplayerstats`, `leaguedash*shotlocations`, `leaguedashptstats` | Zone/tracking aggregates |",
        "| **Player/team dashboards** | Deep split trees (clutch, last N, shooting, splits) | `playerdashboardby*`, `teamdashboardby*` | Split aggregates |",
        "| **Tracking micro-behavior** | Dribbles-before-shot, touch time, pass pairs, contested reb | `playerdashptshots`, `playerdashptpass`, `playerdashptreb` | Derived from tracking, not frame data |",
        "| **Lineup & impact** | On/off ratings, 5-man lineups, matchups, rotations | `teamplayeronoffsummary`, `leaguedashlineups`, `gamerotation` | Stint-level, not coordinates |",
        "",
        "**Important:** None of these endpoints provide frame-by-frame `(x, y)` positions for all 10 players.",
        "Spatial data is limited to **shot coordinates** (`shotchartdetail`: `LOC_X`, `LOC_Y`) and **zone buckets**",
        "(`leaguedash*shotlocations`). Tracking endpoints expose **aggregated** speed, distance, touches, passes.",
        "",
        "## Executive Summary",
        "",
        f"- **Endpoints probed:** {meta['total']}",
        f"- **HTTP/API success:** {meta['success_count']} ({meta['success_pct']:.0f}%)",
        f"- **Returned ≥1 row:** {meta['data_count']} ({meta['data_pct']:.0f}%)",
        f"- **Failed:** {meta['fail_count']}",
        "",
        "Legend per endpoint:",
        "- **Status `OK+data`**: API responded and at least one dataset has rows.",
        "- **Status `OK empty`**: API responded but all datasets empty (deprecated, off-season, or wrong params).",
        "- **Status `FAIL`**: Exception or HTTP error.",
        "",
        "## Failed / Empty Endpoints (at a glance)",
        "",
        "| Endpoint | API | Status | Why |",
        "|----------|-----|--------|-----|",
    ]

    for r in sorted(results, key=lambda x: x["endpoint"]):
        if r.get("success") and r.get("has_data"):
            continue
        status = "OK empty" if r.get("success") else "FAIL"
        api = r.get("api_family", "stats")
        reason = r.get("failure_reason") or infer_failure_reason(r) or "—"
        lines.append(f"| `{r['endpoint']}` | {api} | {status} | {reason} |")

    lines.extend([
        "",
        "## Quick Index",
        "",
        "| Endpoint | API | Category | Status | Latency | Rows | Note |",
        "|----------|-----|----------|--------|---------|------|------|",
    ])

    for r in sorted(results, key=lambda x: x["endpoint"]):
        if r.get("success") and r.get("has_data"):
            status = "OK+data"
            note = "—"
        elif r.get("success"):
            status = "OK empty"
            note = r.get("failure_reason") or "—"
        else:
            status = "FAIL"
            note = r.get("failure_reason") or "—"
        api = r.get("api_family", "stats")
        lines.append(
            f"| `{r['endpoint']}` | {api} | {_categorize(r['endpoint'], api)} | {status} | "
            f"{r.get('elapsed_sec', '—')}s | {r.get('total_rows', 0)} | {note} |"
        )

    # Group by category
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_cat.setdefault(_categorize(r["endpoint"], r.get("api_family", "stats")), []).append(r)

    lines.extend(["", "---", ""])

    for cat in sorted(by_cat.keys()):
        lines.append(f"## {cat}")
        lines.append("")
        for r in sorted(by_cat[cat], key=lambda x: x["endpoint"]):
            lines.append(f"### `{r['endpoint']}`")
            lines.append("")
            lines.append(f"- **API family:** `{r.get('api_family', 'stats')}`")
            if r.get("class"):
                lines.append(f"- **Python class:** `{r['class']}`")
            host = "cdn.nba.com (live)" if r.get("api_family") == "live" else "stats.nba.com"
            lines.append(f"- **URL / path:** `{r.get('api_endpoint', r['endpoint'])}` ({host})")
            if r.get("success"):
                lines.append(f"- **Probe status:** {'OK with data' if r.get('has_data') else 'OK but empty'} ({r.get('elapsed_sec')}s)")
            else:
                lines.append(f"- **Probe status:** FAIL ({r.get('elapsed_sec', '—')}s)")
                lines.append(f"- **Error:** `{r.get('error', 'unknown')}`")
            if r.get("failure_reason"):
                lines.append(f"- **Why:** {r['failure_reason']}")

            if r.get("kwargs_used"):
                kw = ", ".join(f"{k}={v!r}" for k, v in r["kwargs_used"].items())
                lines.append(f"- **Parameters used:** {kw}")

            lines.append("")
            lines.append("**What this endpoint provides:**")
            lines.append("")
            desc = _endpoint_purpose(r["endpoint"])
            lines.append(desc)
            lines.append("")

            # Live datasets
            if r.get("datasets"):
                lines.append("**Live probe — datasets:**")
                lines.append("")
                for ds_name, info in r["datasets"].items():
                    lines.append(f"- **`{ds_name}`** — {info['rows']} rows")
                    lines.append(f"  - Columns: {_describe_columns(info['columns'])}")
                    if info.get("sample"):
                        lines.append(f"  - Sample row keys: {list(info['sample'][0].keys())[:8]}...")
                lines.append("")

            # Expected schema from nba_api
            if r.get("expected_schema"):
                lines.append("**Documented schema (nba_api `expected_data`):**")
                lines.append("")
                for ds_name, cols in r["expected_schema"].items():
                    if cols:
                        lines.append(f"- **`{ds_name}`**: {_describe_columns(cols[:20])}")
                lines.append("")

            lines.append("---")
            lines.append("")

    lines.extend([
        "## Notes for nba-fit-analysis",
        "",
        "1. **Prefer V3 endpoints** (`boxscore* v3`, `playbyplayv3`, `boxscoresummaryv3`) — V2 endpoints are deprecated or empty for 2025-26.",
        "2. **Game-level vs season-level:** Box score endpoints need `game_id`; league dash / dashboards need `season` + filters.",
        "3. **Tracking endpoints** (`playerdashpt*`, `boxscoreplayertrackv3`) provide aggregated Second Spectrum-style stats, not frame-by-frame coordinates.",
        "4. **Spatial data:** `shotchartdetail` gives `LOC_X`/`LOC_Y` per shot; `leaguedash*shotlocations` gives zone aggregates.",
        "5. **On/off impact:** `teamplayeronoffsummary` exposes pre-computed `NET_RATING` without building stints.",
        "6. **Empty OK responses** often mean Synergy deprecation, player DNP in sample game, or parameter mismatch — not necessarily a broken endpoint.",
        "",
    ])
    return "\n".join(lines)


def _endpoint_purpose(name: str) -> str:
    """Short human description per endpoint slug."""
    purposes = {
        "playbyplayv3": "Timestamped event stream for a game: substitutions, shots, fouls, clock, primary actor IDs. Foundation for stint construction and L-RAPM.",
        "shotchartdetail": "Every field goal attempt with court coordinates (`LOC_X`, `LOC_Y` in inches), shot zones, distance, make/miss, and game clock.",
        "leaguedashplayershotlocations": "Season shooting splits by court zone (Restricted Area, Paint, Mid-Range, Corner 3, Above Break 3) per player.",
        "leaguedashteamshotlocations": "Team-level zone shooting volume and efficiency — used to find shot-profile vacancies.",
        "boxscoreplayertrackv3": "Game-level tracking aggregates: distance, speed, touches, passes, contested/uncontested FG, rim protection.",
        "playerdashptshots": "Micro shooting behavior: dribbles before shot, touch time, shot clock, closest defender distance buckets.",
        "playerdashptpass": "Passing network: passes made/received by teammate, assist conversion, pass type frequencies.",
        "playerdashptreb": "Rebounding behavior: contested vs uncontested, rebound distance, shot-distance rebounding context.",
        "teamplayeronoffsummary": "Team performance with player on court vs off court; includes OFF/DEF/NET ratings.",
        "teamplayeronoffdetails": "Full box-score splits for on/off court lineup contexts.",
        "boxscorematchupsv3": "Pairwise offensive vs defensive matchup minutes, points, FG allowed, switches for one game.",
        "synergyplaytypes": "Synergy play-type taxonomy (PNR handler, spot-up, iso, etc.) with possession share and PPP.",
        "leaguedashlineups": "5-man lineup combinations with minutes and plus/minus across the league/team filter.",
        "leaguedashplayerstats": "Primary season stat line for all players (traditional + advanced measure types).",
        "playerestimatedmetrics": "NBA estimated net rating, EPM-style metrics per player season.",
        "gamerotation": "Stint/rotation log: who was on court by time segment for a game.",
        "winprobabilitypbp": "Win probability change per play event.",
    }
    if name in purposes:
        return purposes[name]

    cat = _categorize(name)
    generic = {
        "Box Score (game-level)": "Single-game box score slice (traditional, advanced, usage, hustle, etc.) for all players/teams in that game.",
        "Play-by-Play": "Play-by-play event data for a specific game ID.",
        "Shot Charts": "Shot location data at various granularities (player, lineup, league-wide).",
        "Player Tracking Dashboards": "Season or game aggregated tracking-derived micro stats for a player.",
        "Team Tracking Dashboards": "Team-level passing, rebounding, and shooting tracking aggregates.",
        "On/Off & Lineup Impact": "Lineup impact, on/off court performance, or chemistry splits.",
        "League Dash (season aggregates)": "Season-long league-wide stat tables with rich filtering (clutch, shot locations, PT stats).",
        "Common / Metadata": "Roster, player info, team years, and reference metadata.",
        "Player Profiles & Logs": "Career profiles, game logs, splits, comparisons, and streak finders.",
        "Team Profiles & Logs": "Team game logs, dashboards, and team-level analytics.",
        "Draft & Combine": "Draft history and combine measurement/shooting/drill results.",
        "Video": "Video asset metadata tied to game events.",
        "Scoreboard": "Games on a specific date with scores and status.",
        "Schedule": "Full season schedule with game IDs and broadcast info.",
    }
    return generic.get(cat, f"NBA Stats API endpoint (`{name}`). See documented schema below for column details.")


def _all_endpoint_names() -> list[tuple[str, str]]:
    """Return (endpoint_name, api_family) for every probe target."""
    import nba_api.live.nba.endpoints as live_mod
    import nba_api.stats.endpoints as ep_mod

    stats_names = list(ep_mod.__all__)
    if "videoeventsasset" not in stats_names:
        stats_names.append("videoeventsasset")
    targets: list[tuple[str, str]] = [(n, "stats") for n in stats_names]
    for name in live_mod.__all__:
        targets.append((name, "live"))
    return targets


def main() -> int:
    targets = _all_endpoint_names()
    print(f"Probing {len(targets)} nba_api endpoints (stats + live + extras) ...")
    print(f"Context: season={SEASON}, game={GAME_ID}, player={PLAYER_ID}, team={TEAM_ID}")

    results: list[dict[str, Any]] = []
    for i, (name, family) in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {family}:{name} ...", flush=True)
        results.append(probe_one(name, api_family=family))

    success = [r for r in results if r.get("success")]
    with_data = [r for r in success if r.get("has_data")]
    failed = [r for r in results if not r.get("success")]

    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "season": SEASON,
        "game_id": GAME_ID,
        "player_id": PLAYER_ID,
        "team_id": TEAM_ID,
        "total": len(results),
        "success_count": len(success),
        "data_count": len(with_data),
        "fail_count": len(failed),
        "success_pct": 100 * len(success) / len(results) if results else 0,
        "data_pct": 100 * len(with_data) / len(results) if results else 0,
    }

    out_dir = Path(__file__).parent
    json_path = out_dir / "probe_all_results.json"
    json_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")

    md_path = out_dir / "docs" / "NBA_ENDPOINTS_DATA_REFERENCE.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(generate_markdown(results, meta), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(f"Success: {meta['success_count']}/{meta['total']}")
    print(f"With data: {meta['data_count']}/{meta['total']}")
    print(f"Failed: {meta['fail_count']}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0 if meta["fail_count"] == 0 else 1


def retry_failures_and_regenerate() -> None:
    """Re-probe failed endpoints and refresh JSON + Markdown."""
    json_path = Path(__file__).parent / "probe_all_results.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    results = payload["results"]
    by_name = {r["endpoint"]: r for r in results}
    failed = [r["endpoint"] for r in results if not r.get("success")]
    print(f"Retrying {len(failed)} failed endpoints...")
    for name in failed:
        print(f"  retry {name}...")
        family = by_name[name].get("api_family", "stats")
        by_name[name] = probe_one(name, api_family=family)
    new_results = [by_name[r["endpoint"]] for r in results]
    success = [r for r in new_results if r.get("success")]
    with_data = [r for r in success if r.get("has_data")]
    failed_n = [r for r in new_results if not r.get("success")]
    meta = {**payload["meta"], "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    meta.update(
        {
            "success_count": len(success),
            "data_count": len(with_data),
            "fail_count": len(failed_n),
            "success_pct": 100 * len(success) / len(new_results),
            "data_pct": 100 * len(with_data) / len(new_results),
            "retried_failures": len(failed),
        }
    )
    json_path.write_text(json.dumps({"meta": meta, "results": new_results}, indent=2), encoding="utf-8")
    md_path = Path(__file__).parent / "docs" / "NBA_ENDPOINTS_DATA_REFERENCE.md"
    md_path.write_text(generate_markdown(new_results, meta), encoding="utf-8")
    print(f"Updated: success={meta['success_count']}, with_data={meta['data_count']}, fail={meta['fail_count']}")


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1 and _sys.argv[1] == "--retry-failures":
        retry_failures_and_regenerate()
    else:
        raise SystemExit(main())
