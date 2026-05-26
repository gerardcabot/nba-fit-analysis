#!/usr/bin/env python3
"""
Probe nba_api tactical endpoints for nba-fit-analysis.

Audits spatial, tracking, lineup, and synergy endpoints for schema delivery,
latency, and key tactical columns.
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEASON = "2025-26"
FALLBACK_SEASON = "2024-25"
LUKA_PLAYER_ID = 1629029
LUKA_TEAM_ID = 1610612742  # default DAL; resolved dynamically in main()
FALLBACK_GAME_ID = "0022501193"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

# Tactical columns we expect per endpoint (subset for pass/fail audit)
TACTICAL_KEYS: dict[str, list[str]] = {
    "shotchartdetail": ["LOC_X", "LOC_Y", "SHOT_DISTANCE", "SHOT_TYPE", "ACTION_TYPE"],
    "leaguedashplayershotlocations": ["PLAYER_ID", "PLAYER_NAME", "Restricted Area"],
    "leaguedashteamshotlocations": ["TEAM_ID", "TEAM_NAME", "Restricted Area"],
    "boxscoreplayertrackv3": ["distance", "speed", "touches", "passes"],
    "playerdashptshots": ["DRIBBLE_RANGE", "TOUCH_TIME_RANGE", "FGA_FREQUENCY"],
    "playerdashptpass": ["PASS_TYPE", "PASS", "AST", "FREQUENCY"],
    "playerdashptreb": ["REB_FREQUENCY", "C_REB", "UC_REB"],
    "teamplayeronoffdetails": ["PLUS_MINUS", "COURT_STATUS", "VS_PLAYER_NAME"],
    "teamplayeronoffsummary": ["NET_RATING", "OFF_RATING", "DEF_RATING", "PLUS_MINUS"],
    "boxscorematchupsv3": ["matchupMinutes", "playerPoints", "personIdOff", "personIdDef"],
    "synergyplaytypes": ["PLAY_TYPE", "POSS_PCT", "PPP", "TYPE_GROUPING"],
}

_backoff_sec = 0.8


def _patch_headers() -> None:
    from nba_api.stats.library.http import NBAStatsHTTP

    NBAStatsHTTP.headers = {**NBAStatsHTTP.headers, **BROWSER_HEADERS}


def _sleep_backoff() -> None:
    global _backoff_sec
    time.sleep(_backoff_sec)
    _backoff_sec = min(_backoff_sec * 1.15, 4.0)


def _resolve_luka_team_id(season: str) -> int:
    """Resolve Luka's current team for the target season."""
    from nba_api.stats.endpoints import commonallplayers

    _patch_headers()
    _sleep_backoff()
    players = commonallplayers.CommonAllPlayers(
        is_only_current_season=1,
        season=season,
        league_id="00",
        timeout=45,
    ).get_data_frames()[0]
    row = players[players["PERSON_ID"] == LUKA_PLAYER_ID]
    if not row.empty and "TEAM_ID" in row.columns:
        tid = row.iloc[0]["TEAM_ID"]
        if tid and int(tid) > 0:
            return int(tid)
    return 1610612742


def _safe_print(text: str) -> None:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(enc, errors="replace").decode(enc))


def _resolve_game_id(team_id: int) -> str:
    from nba_api.stats.endpoints import leaguegamefinder

    _patch_headers()
    _sleep_backoff()
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=SEASON,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
        team_id_nullable=str(team_id),
    )
    df = finder.get_data_frames()[0]
    if df.empty:
        return FALLBACK_GAME_ID
    df = df.sort_values("GAME_DATE", ascending=False)
    return str(df.iloc[0]["GAME_ID"])


def _flatten_columns(cols: Any) -> list[str]:
    if isinstance(cols, pd.MultiIndex):
        return [" | ".join(str(p) for p in tup if str(p) not in ("", "nan")) for tup in cols]
    return [str(c) for c in cols]


def _columns_match(df: pd.DataFrame, keys: list[str]) -> list[str]:
    flat = _flatten_columns(df.columns)
    flat_upper = {c.upper(): c for c in flat}
    found: list[str] = []
    for key in keys:
        if key in df.columns:
            found.append(key)
            continue
        if key.upper() in flat_upper:
            found.append(flat_upper[key.upper()])
            continue
        for col in flat:
            if key.lower() in col.lower():
                found.append(col)
                break
    return found


@dataclass
class ProbeResult:
    endpoint: str
    category: str
    success: bool
    elapsed_sec: float
    row_count: int = 0
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)
    key_columns_found: list[str] = field(default_factory=list)
    error: str | None = None


def _probe(name: str, category: str, fn: Callable[[], dict[str, Any]]) -> ProbeResult:
    start = time.time()
    try:
        payload = fn()
        elapsed = time.time() - start
        return ProbeResult(
            endpoint=name,
            category=category,
            success=True,
            elapsed_sec=elapsed,
            row_count=payload.get("row_count", 0),
            datasets=payload.get("datasets", {}),
            key_columns_found=payload.get("key_columns_found", []),
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - start
        return ProbeResult(
            endpoint=name,
            category=category,
            success=False,
            elapsed_sec=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )


def _dataset_summary(endpoint_obj: Any) -> dict[str, Any]:
    """Summarize all data frames on an nba_api endpoint instance."""
    out: dict[str, dict[str, Any]] = {}
    total_rows = 0
    all_keys: list[str] = []

    frames = endpoint_obj.get_data_frames()

    ds_names: list[str] = []
    if endpoint_obj.nba_response is not None:
        try:
            raw = endpoint_obj.nba_response.get_data_sets(getattr(endpoint_obj, "endpoint", None))
            ds_names = list(raw.keys())
        except Exception:  # noqa: BLE001
            try:
                ds_names = list(endpoint_obj.nba_response.get_data_sets().keys())
            except Exception:  # noqa: BLE001
                ds_names = []

    if ds_names and len(ds_names) == len(frames):
        pairs = zip(ds_names, frames)
    else:
        pairs = [(f"dataset_{i}", df) for i, df in enumerate(frames)]

    for ds_name, df in pairs:
        cols = _flatten_columns(df.columns)
        rows = len(df)
        total_rows += rows
        sample = df.head(1).to_dict(orient="records") if rows else []
        out[str(ds_name)] = {
            "rows": rows,
            "columns": cols,
            "sample": sample,
        }

    expected = TACTICAL_KEYS.get(endpoint_obj.endpoint, [])
    for info in out.values():
        dummy = pd.DataFrame(columns=info["columns"])
        all_keys.extend(_columns_match(dummy, expected))

    return {
        "datasets": out,
        "row_count": total_rows,
        "key_columns_found": sorted(set(all_keys)),
    }


# ---------------------------------------------------------------------------
# Endpoint probes
# ---------------------------------------------------------------------------

def probe_shotchartdetail(game_id: str, season: str) -> dict[str, Any]:
    from nba_api.stats.endpoints import shotchartdetail

    _patch_headers()
    _sleep_backoff()
    ep = shotchartdetail.ShotChartDetail(
        team_id=LUKA_TEAM_ID,
        player_id=LUKA_PLAYER_ID,
        season_nullable=season,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA",
        game_id_nullable=game_id,
        timeout=45,
    )
    return _dataset_summary(ep)


def probe_leaguedashplayershotlocations(season: str) -> dict[str, Any]:
    from nba_api.stats.endpoints import leaguedashplayershotlocations

    _patch_headers()
    _sleep_backoff()
    ep = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
        season=season,
        season_type_all_star="Regular Season",
        team_id_nullable=str(LUKA_TEAM_ID),
        timeout=45,
    )
    summary = _dataset_summary(ep)
    df = ep.shot_locations.get_data_frame()
    luka = df[df["PLAYER_ID"] == LUKA_PLAYER_ID] if "PLAYER_ID" in df.columns else df.head(1)
    summary["datasets"]["ShotLocations"]["luka_rows"] = len(luka)
    summary["datasets"]["ShotLocations"]["sample"] = luka.head(1).to_dict(orient="records")
    summary["row_count"] = len(luka)
    return summary


def probe_leaguedashteamshotlocations(season: str) -> dict[str, Any]:
    from nba_api.stats.endpoints import leaguedashteamshotlocations

    _patch_headers()
    _sleep_backoff()
    ep = leaguedashteamshotlocations.LeagueDashTeamShotLocations(
        season=season,
        season_type_all_star="Regular Season",
        team_id_nullable=str(LUKA_TEAM_ID),
        timeout=45,
    )
    summary = _dataset_summary(ep)
    df = ep.shot_locations.get_data_frame()
    dal = df[df["TEAM_ID"] == LUKA_TEAM_ID] if "TEAM_ID" in df.columns else df.head(1)
    summary["datasets"]["ShotLocations"]["dal_rows"] = len(dal)
    summary["datasets"]["ShotLocations"]["sample"] = dal.head(1).to_dict(orient="records")
    summary["row_count"] = len(dal)
    return summary


def probe_boxscoreplayertrackv3(game_id: str) -> dict[str, Any]:
    from nba_api.stats.endpoints import boxscoreplayertrackv3

    _patch_headers()
    _sleep_backoff()
    ep = boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=game_id, timeout=45)
    summary = _dataset_summary(ep)
    df = ep.player_stats.get_data_frame()
    luka = df[df["personId"] == LUKA_PLAYER_ID] if "personId" in df.columns else df.head(1)
    summary["datasets"]["PlayerStats"]["luka_rows"] = len(luka)
    summary["datasets"]["PlayerStats"]["sample"] = luka.head(1).to_dict(orient="records")
    summary["row_count"] = len(luka)
    return summary


def probe_playerdashptshots(season: str) -> dict[str, Any]:
    from nba_api.stats.endpoints import playerdashptshots

    _patch_headers()
    _sleep_backoff()
    ep = playerdashptshots.PlayerDashPtShots(
        team_id=LUKA_TEAM_ID,
        player_id=LUKA_PLAYER_ID,
        season=season,
        season_type_all_star="Regular Season",
        timeout=45,
    )
    return _dataset_summary(ep)


def probe_playerdashptpass(season: str) -> dict[str, Any]:
    from nba_api.stats.endpoints import playerdashptpass

    _patch_headers()
    _sleep_backoff()
    ep = playerdashptpass.PlayerDashPtPass(
        team_id=LUKA_TEAM_ID,
        player_id=LUKA_PLAYER_ID,
        season=season,
        season_type_all_star="Regular Season",
        timeout=45,
    )
    return _dataset_summary(ep)


def probe_playerdashptreb(season: str) -> dict[str, Any]:
    from nba_api.stats.endpoints import playerdashptreb

    _patch_headers()
    _sleep_backoff()
    ep = playerdashptreb.PlayerDashPtReb(
        team_id=LUKA_TEAM_ID,
        player_id=LUKA_PLAYER_ID,
        season=season,
        season_type_all_star="Regular Season",
        timeout=45,
    )
    return _dataset_summary(ep)


def probe_teamplayeronoffdetails(season: str) -> dict[str, Any]:
    from nba_api.stats.endpoints import teamplayeronoffdetails

    _patch_headers()
    _sleep_backoff()
    ep = teamplayeronoffdetails.TeamPlayerOnOffDetails(
        team_id=LUKA_TEAM_ID,
        season=season,
        season_type_all_star="Regular Season",
        timeout=45,
    )
    summary = _dataset_summary(ep)
    df = ep.players_off_court_team_player_on_off_details.get_data_frame()
    luka = df[df["VS_PLAYER_ID"] == LUKA_PLAYER_ID] if "VS_PLAYER_ID" in df.columns else df.head(3)
    summary["datasets"]["PlayersOffCourtTeamPlayerOnOffDetails"]["luka_sample"] = (
        luka.head(2).to_dict(orient="records")
    )
    return summary


def probe_teamplayeronoffsummary(season: str) -> dict[str, Any]:
    from nba_api.stats.endpoints import teamplayeronoffsummary

    _patch_headers()
    _sleep_backoff()
    ep = teamplayeronoffsummary.TeamPlayerOnOffSummary(
        team_id=LUKA_TEAM_ID,
        season=season,
        season_type_all_star="Regular Season",
        timeout=45,
    )
    summary = _dataset_summary(ep)
    df = ep.players_off_court_team_player_on_off_summary.get_data_frame()
    luka = df[df["VS_PLAYER_ID"] == LUKA_PLAYER_ID] if "VS_PLAYER_ID" in df.columns else df.head(3)
    summary["datasets"]["PlayersOffCourtTeamPlayerOnOffSummary"]["luka_sample"] = (
        luka.head(2).to_dict(orient="records")
    )
    return summary


def probe_boxscorematchupsv3(game_id: str) -> dict[str, Any]:
    from nba_api.stats.endpoints import boxscorematchupsv3

    _patch_headers()
    _sleep_backoff()
    ep = boxscorematchupsv3.BoxScoreMatchupsV3(game_id=game_id, timeout=45)
    summary = _dataset_summary(ep)
    df = ep.player_stats.get_data_frame()
    luka = df[df["personIdOff"] == LUKA_PLAYER_ID] if "personIdOff" in df.columns else df.head(3)
    summary["datasets"]["PlayerStats"]["luka_matchup_rows"] = len(luka)
    summary["datasets"]["PlayerStats"]["sample"] = luka.head(2).to_dict(orient="records")
    summary["row_count"] = len(luka)
    return summary


def probe_synergyplaytypes(season: str) -> dict[str, Any]:
    from nba_api.stats.endpoints import synergyplaytypes

    _patch_headers()
    _sleep_backoff()
    ep = synergyplaytypes.SynergyPlayTypes(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="P",
        timeout=45,
    )
    summary = _dataset_summary(ep)
    df = ep.synergy_play_type.get_data_frame()
    luka = df[df["TEAM_ID"] == LUKA_TEAM_ID] if "TEAM_ID" in df.columns else df
    # Synergy player rows use TEAM_ID as player's team; filter PLAY_TYPE for sample
    if "PLAYER_ID" in df.columns:
        luka = df[df["PLAYER_ID"] == LUKA_PLAYER_ID]
    elif "PLAYER_NAME" in df.columns:
        luka = df[df["PLAYER_NAME"].str.contains("Don", case=False, na=False)]
    else:
        luka = df.head(5)
    summary["datasets"]["SynergyPlayType"]["luka_rows"] = len(luka)
    summary["datasets"]["SynergyPlayType"]["sample"] = luka.head(3).to_dict(orient="records")
    summary["row_count"] = len(luka)
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_probe(result: ProbeResult) -> None:
    status = "PASS" if result.success else "FAIL"
    print(f"\n[{status}] {result.endpoint}  ({result.elapsed_sec:.2f}s)  [{result.category}]")
    if result.success:
        print(f"  rows (primary): {result.row_count}")
        print(f"  key tactical columns found: {result.key_columns_found or '—'}")
        for ds_name, info in result.datasets.items():
            cols = info.get("columns", [])
            print(f"  dataset '{ds_name}': {info.get('rows', '?')} rows, {len(cols)} cols")
            if len(cols) <= 20:
                print(f"    columns: {cols}")
            else:
                print(f"    columns (first 15): {cols[:15]} ...")
            if info.get("sample"):
                _safe_print(f"    sample: {info['sample'][:1]}")
    else:
        print(f"  error: {result.error}")


def _run_all_probes(game_id: str, season: str) -> list[ProbeResult]:
    probes: list[tuple[str, str, Callable[[], dict[str, Any]]]] = [
        ("shotchartdetail", "Spatial & Shot Selection", lambda: probe_shotchartdetail(game_id, season)),
        ("leaguedashplayershotlocations", "Spatial & Shot Selection", lambda: probe_leaguedashplayershotlocations(season)),
        ("leaguedashteamshotlocations", "Spatial & Shot Selection", lambda: probe_leaguedashteamshotlocations(season)),
        ("boxscoreplayertrackv3", "Tracking & Rhythm", lambda: probe_boxscoreplayertrackv3(game_id)),
        ("playerdashptshots", "Tracking & Rhythm", lambda: probe_playerdashptshots(season)),
        ("playerdashptpass", "Tracking & Rhythm", lambda: probe_playerdashptpass(season)),
        ("playerdashptreb", "Tracking & Rhythm", lambda: probe_playerdashptreb(season)),
        ("teamplayeronoffdetails", "Lineup & Matchup", lambda: probe_teamplayeronoffdetails(season)),
        ("teamplayeronoffsummary", "Lineup & Matchup", lambda: probe_teamplayeronoffsummary(season)),
        ("boxscorematchupsv3", "Lineup & Matchup", lambda: probe_boxscorematchupsv3(game_id)),
        ("synergyplaytypes", "Lineup & Matchup", lambda: probe_synergyplaytypes(season)),
    ]

    results: list[ProbeResult] = []
    for name, category, fn in probes:
        print("\n" + "-" * 72)
        print(f"Probing {name} ...")
        result = _probe(name, category, fn)
        _print_probe(result)
        results.append(result)
    return results


def main() -> int:
    global LUKA_TEAM_ID

    print("NBA Tactical Endpoint Probe — nba-fit-analysis")
    print(f"Season: {SEASON}  |  Player: Luka Doncic ({LUKA_PLAYER_ID})")
    print(f"Python: {sys.version.split()[0]}")

    season = SEASON
    try:
        LUKA_TEAM_ID = _resolve_luka_team_id(season)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] team resolve failed ({exc}); defaulting to DAL")
    print(f"Resolved team_id: {LUKA_TEAM_ID}")

    try:
        game_id = _resolve_game_id(LUKA_TEAM_ID)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[WARN] game_id resolve failed ({exc}); using fallback {FALLBACK_GAME_ID}")
        game_id = FALLBACK_GAME_ID

    print(f"Resolved game_id: {game_id}")

    results = _run_all_probes(game_id, season)

    # Retry empty synergy/tracking with fallback season if needed
    empty_endpoints = [r.endpoint for r in results if r.success and r.row_count == 0]
    if empty_endpoints and season == SEASON:
        print(f"\n[INFO] Retrying empty endpoints with fallback season {FALLBACK_SEASON}: {empty_endpoints}")
        retry_map = {
            "synergyplaytypes": lambda: probe_synergyplaytypes(FALLBACK_SEASON),
            "playerdashptshots": lambda: probe_playerdashptshots(FALLBACK_SEASON),
            "playerdashptpass": lambda: probe_playerdashptpass(FALLBACK_SEASON),
            "playerdashptreb": lambda: probe_playerdashptreb(FALLBACK_SEASON),
            "shotchartdetail": lambda: probe_shotchartdetail("", FALLBACK_SEASON),
        }
        for i, r in enumerate(results):
            if r.endpoint in retry_map and r.row_count == 0:
                print(f"\n--- retry {r.endpoint} ---")
                new = _probe(r.endpoint, r.category, retry_map[r.endpoint])
                _print_probe(new)
                results[i] = new

    print("\n" + "=" * 72)
    print("  PROBE SUMMARY")
    print("=" * 72)
    ok = sum(1 for r in results if r.success)
    print(f"  Passed: {ok}/{len(results)}")
    for r in results:
        mark = "OK" if r.success else "FAIL"
        keys = ", ".join(r.key_columns_found[:4]) if r.key_columns_found else "—"
        print(f"  [{mark}] {r.endpoint:35s} {r.elapsed_sec:6.2f}s  rows={r.row_count:4d}  keys={keys}")

    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
