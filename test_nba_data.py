#!/usr/bin/env python3
"""
NBA data source validation benchmark for nba-fit-analysis.

Tests three ingestion paths:
  1. nba_api  (stats.nba.com PlayByPlay endpoints)
  2. soccerdata (FBref scraper — basketball capability probe)
  3. Custom Basketball-Reference scraper (requests + BeautifulSoup)
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Shared reporting helpers
# ---------------------------------------------------------------------------

SEASON = "2025-26"
FALLBACK_GAME_ID = "0022500001"  # first regular-season game ID pattern for 2025-26


@dataclass
class TestResult:
    name: str
    success: bool
    elapsed_sec: float
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def run_timed(name: str, fn: Callable[[], dict[str, Any]]) -> TestResult:
    """Execute *fn* inside try/except and capture latency."""
    start = time.time()
    try:
        payload = fn()
        elapsed = time.time() - start
        return TestResult(
            name=name,
            success=True,
            elapsed_sec=elapsed,
            summary=payload.get("summary", "OK"),
            details=payload,
        )
    except Exception as exc:  # noqa: BLE001 — benchmark script; surface all failures
        elapsed = time.time() - start
        return TestResult(
            name=name,
            success=False,
            elapsed_sec=elapsed,
            summary=f"FAILED: {type(exc).__name__}",
            error=f"{exc}\n{traceback.format_exc()}",
        )


def print_result(result: TestResult) -> None:
    status = "PASS" if result.success else "FAIL"
    print(f"\n[{status}] {result.name}  ({result.elapsed_sec:.2f}s)")
    print(f"  {result.summary}")
    for key, value in result.details.items():
        if key == "summary":
            continue
        print(f"  {key}: {value}")
    if result.error:
        print("  --- traceback ---")
        for line in result.error.strip().splitlines()[-8:]:
            print(f"  {line}")


# ---------------------------------------------------------------------------
# Realistic browser headers (stats.nba.com is sensitive to bot traffic)
# ---------------------------------------------------------------------------

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

BBREF_HEADERS = {
    "User-Agent": BROWSER_HEADERS["User-Agent"],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.basketball-reference.com/",
}


# ---------------------------------------------------------------------------
# 1. nba_api
# ---------------------------------------------------------------------------

def _patch_nba_api_headers() -> None:
    """Inject browser-like headers into nba_api's HTTP client."""
    from nba_api.stats.library.http import NBAStatsHTTP

    NBAStatsHTTP.headers = {**NBAStatsHTTP.headers, **BROWSER_HEADERS}


def _resolve_game_id() -> tuple[str, str | None, str | None]:
    """Pick one recent 2025-26 regular-season game and BBRef URL hints."""
    from nba_api.stats.endpoints import leaguegamefinder

    _patch_nba_api_headers()
    time.sleep(0.6)  # gentle pacing
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=SEASON,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    df = finder.get_data_frames()[0]
    if df.empty:
        return FALLBACK_GAME_ID, None, None

    df = df.sort_values("GAME_DATE", ascending=False)
    game_id = str(df.iloc[0]["GAME_ID"])
    game_rows = df[df["GAME_ID"] == game_id]

    home_team: str | None = None
    game_date: str | None = None
    for _, row in game_rows.iterrows():
        matchup = str(row.get("MATCHUP", ""))
        if " vs. " in matchup:
            home_team = str(row["TEAM_ABBREVIATION"])
            raw_date = str(row["GAME_DATE"])
            game_date = raw_date[:10].replace("-", "")
            break

    return game_id, game_date, home_team


def test_nba_api_playbyplay_v2(game_id: str) -> dict[str, Any]:
    from nba_api.stats.endpoints import playbyplayv2

    _patch_nba_api_headers()
    time.sleep(0.6)
    try:
        endpoint = playbyplayv2.PlayByPlayV2(
            game_id=game_id,
            start_period=1,
            end_period=10,
            timeout=30,
        )
        df = endpoint.get_data_frames()[0]
    except KeyError as exc:
        # NBA.com now returns JSON without resultSet for deprecated V2 endpoint.
        return {
            "summary": "V2 deprecated — NBA API returned empty/malformed payload (expected)",
            "game_id": game_id,
            "endpoint": "PlayByPlayV2",
            "row_count": 0,
            "deprecated": True,
            "error_type": type(exc).__name__,
        }

    columns = list(df.columns)
    row_count = len(df)
    sample_cols = [
        c
        for c in (
            "GAME_ID",
            "EVENTNUM",
            "PERIOD",
            "PCTIMESTRING",
            "HOMEDESCRIPTION",
            "VISITORDESCRIPTION",
            "PLAYER1_ID",
            "PLAYER2_ID",
            "PLAYER3_ID",
        )
        if c in columns
    ]
    return {
        "summary": f"V2 returned {row_count} events; columns={len(columns)}",
        "game_id": game_id,
        "endpoint": "PlayByPlayV2",
        "row_count": row_count,
        "columns": columns,
        "key_columns_present": sample_cols,
        "sample_event": df.head(1).to_dict(orient="records") if row_count else [],
    }


def test_nba_api_playbyplay_v3(game_id: str) -> dict[str, Any]:
    from nba_api.stats.endpoints import playbyplayv3

    _patch_nba_api_headers()
    time.sleep(0.6)
    endpoint = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=30)
    df = endpoint.get_data_frames()[0]
    columns = list(df.columns)
    row_count = len(df)
    sample_cols = [
        c
        for c in (
            "gameId",
            "actionNumber",
            "period",
            "clock",
            "description",
            "personId",
            "teamId",
            "videoAvailable",
        )
        if c in columns
    ]
    return {
        "summary": f"V3 returned {row_count} events; columns={len(columns)}",
        "game_id": game_id,
        "endpoint": "PlayByPlayV3",
        "row_count": row_count,
        "columns": columns,
        "key_columns_present": sample_cols,
        "sample_event": df.head(1).to_dict(orient="records") if row_count else [],
    }


def _resolve_game_id_wrapped() -> dict[str, Any]:
    gid, game_date, home_team = _resolve_game_id()
    bbref_url = None
    if game_date and home_team:
        bbref_url = (
            f"https://www.basketball-reference.com/boxscores/pbp/"
            f"{game_date}0{home_team}.html"
        )
    return {
        "summary": f"Using game_id={gid}",
        "game_id": gid,
        "game_date": game_date,
        "home_team": home_team,
        "bbref_url": bbref_url,
    }


def run_nba_api_tests_fixed() -> list[TestResult]:
    banner("TEST 1 — nba_api (stats.nba.com PlayByPlay)")
    results: list[TestResult] = []

    resolve = run_timed("nba_api — resolve game_id", _resolve_game_id_wrapped)
    print_result(resolve)
    game_id = resolve.details.get("game_id", FALLBACK_GAME_ID) if resolve.success else FALLBACK_GAME_ID

    r_v2 = run_timed(
        "nba_api — PlayByPlayV2",
        lambda: test_nba_api_playbyplay_v2(game_id),
    )
    print_result(r_v2)
    results.extend([resolve, r_v2])

    r_v3 = run_timed(
        "nba_api — PlayByPlayV3",
        lambda: test_nba_api_playbyplay_v3(game_id),
    )
    print_result(r_v3)
    results.append(r_v3)
    return results


# ---------------------------------------------------------------------------
# 2. soccerdata (basketball / FBref probe)
# ---------------------------------------------------------------------------

def test_soccerdata_basketball_probe() -> dict[str, Any]:
    import soccerdata as sd

    findings: list[str] = []

    # Document supported leagues (soccer-only)
    fbref_leagues = sd.FBref.available_leagues()
    findings.append(f"FBref.available_leagues() -> {fbref_leagues}")

    nba_like = [lg for lg in fbref_leagues if "NBA" in lg.upper() or "BASKET" in lg.upper()]
    findings.append(f"NBA/Basketball leagues in FBref: {nba_like or 'NONE'}")

    # Attempt direct FBref NBA URL scrape via internal reader (expected to fail gracefully)
    fbref = sd.FBref(leagues=["ENG-Premier League"], seasons=["2526"], no_cache=True)
    try:
        team_stats = fbref.read_team_season_stats(stat_type="standard")
        findings.append(
            f"read_team_season_stats (soccer) OK: shape={team_stats.shape}, "
            f"cols={list(team_stats.columns[:6])}..."
        )
    except Exception as exc:  # noqa: BLE001
        findings.append(f"read_team_season_stats failed: {exc}")

    # Probe whether any basketball-specific API exists on the package
    basketball_attrs = [
        name
        for name in dir(sd)
        if any(k in name.lower() for k in ("nba", "basketball", "bbref", "bball"))
    ]
    findings.append(f"Basketball-related soccerdata exports: {basketball_attrs or 'NONE'}")

    # Try ESPN class (sometimes covers multiple sports)
    espn_leagues: list[str] = []
    if hasattr(sd, "ESPN") and hasattr(sd.ESPN, "available_leagues"):
        try:
            espn_leagues = sd.ESPN.available_leagues()
        except Exception as exc:  # noqa: BLE001
            findings.append(f"ESPN.available_leagues error: {exc}")
    findings.append(f"ESPN.available_leagues() -> {espn_leagues}")

    basketball_viable = bool(nba_like) or bool(basketball_attrs)
    return {
        "summary": (
            "soccerdata is soccer-focused; no native NBA PBP pipeline detected"
            if not basketball_viable
            else "Partial basketball support detected — inspect findings"
        ),
        "basketball_viable": basketball_viable,
        "findings": findings,
    }


def run_soccerdata_tests() -> list[TestResult]:
    banner("TEST 2 — soccerdata (FBref / basketball capability probe)")
    result = run_timed("soccerdata — basketball probe", test_soccerdata_basketball_probe)
    print_result(result)
    return [result]


# ---------------------------------------------------------------------------
# 3. Basketball-Reference custom scraper
# ---------------------------------------------------------------------------

def _fallback_bbref_pbp_url() -> str:
    """Opening-night 2025-26 game: HOU @ OKC (Oct 21, 2025)."""
    return "https://www.basketball-reference.com/boxscores/pbp/202510210OKC.html"


def _extract_pbp_table(soup: Any) -> Any:
    """Locate BBRef PBP table, including tables hidden in HTML comments."""
    from bs4 import BeautifulSoup

    pbp_table = soup.find("table", id="pbp")
    if pbp_table is not None:
        return pbp_table

    for node in soup.find_all(string=lambda t: isinstance(t, str) and 'id="pbp"' in t):
        inner = BeautifulSoup(str(node), "html.parser")
        pbp_table = inner.find("table", id="pbp")
        if pbp_table is not None:
            return pbp_table
    return None


def _table_headers_and_rows(table: Any) -> tuple[list[str], list[Any]]:
    """Parse BBRef tables that may omit <thead>."""
    thead = table.find("thead")
    if thead is not None:
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
    else:
        first_row = table.find("tr")
        headers = [
            cell.get_text(strip=True)
            for cell in (first_row.find_all("th") if first_row else [])
        ]

    body = table.find("tbody") or table
    rows = body.find_all("tr")
    if not thead and rows:
        rows = rows[1:]  # skip header row
    return headers, rows


def scrape_bbref_play_by_play(url: str) -> dict[str, Any]:
    import requests
    from bs4 import BeautifulSoup

    time.sleep(1.0)
    resp = requests.get(url, headers=BBREF_HEADERS, timeout=30)
    status = resp.status_code
    if status != 200:
        raise RuntimeError(f"HTTP {status} for {url}")

    soup = BeautifulSoup(resp.text, "html.parser")
    pbp_table = _extract_pbp_table(soup)
    if pbp_table is None:
        title = soup.title.string.strip() if soup.title and soup.title.string else "N/A"
        raise RuntimeError(
            f"PBP table not found (possible block or layout change). page_title={title!r}"
        )

    headers, rows = _table_headers_and_rows(pbp_table)
    events: list[dict[str, str]] = []
    for row in rows[:5]:
        cells = [td.get_text(" ", strip=True) for td in row.find_all(["th", "td"])]
        if cells:
            events.append(dict(zip(headers, cells)))

    raw_text_sample = " | ".join(
        td.get_text(" ", strip=True) for td in rows[0].find_all(["th", "td"])
    ) if rows else ""

    return {
        "summary": f"Scraped PBP table: {len(rows)} events, {len(headers)} columns",
        "url": url,
        "http_status": status,
        "columns": headers,
        "event_count": len(rows),
        "sample_events": events,
        "first_event_raw": raw_text_sample,
        "has_timestamps": any("Time" in h or "time" in h for h in headers),
        "has_player_refs": bool(soup.find("a", href=lambda h: h and "/players/" in h)),
    }


def test_bbref_scraper(
    game_id: str | None,
    bbref_url: str | None = None,
) -> dict[str, Any]:
    url = bbref_url or _fallback_bbref_pbp_url()
    url_source = "fallback" if bbref_url is None else "derived_from_nba_game_id"

    result = scrape_bbref_play_by_play(url)
    result["url_source"] = url_source
    result["game_id"] = game_id
    return result


def run_bbref_tests(
    game_id: str | None,
    bbref_url: str | None = None,
) -> list[TestResult]:
    banner("TEST 3 — Basketball-Reference scraper (requests + BeautifulSoup)")
    result = run_timed(
        "bbref — custom PBP scraper",
        lambda: test_bbref_scraper(game_id, bbref_url),
    )
    print_result(result)
    return [result]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_final_summary(all_results: list[TestResult]) -> None:
    banner("BENCHMARK SUMMARY")
    for r in all_results:
        status = "PASS" if r.success else "FAIL"
        print(f"  [{status}] {r.name:40s}  {r.elapsed_sec:6.2f}s  {r.summary}")

    slowest = max(all_results, key=lambda x: x.elapsed_sec)
    failed = [r for r in all_results if not r.success]
    print(f"\n  Slowest test : {slowest.name} ({slowest.elapsed_sec:.2f}s)")
    print(f"  Failures     : {len(failed)} / {len(all_results)}")


def main() -> int:
    print("NBA Data Source Validation Benchmark")
    print(f"Target season: {SEASON}")
    print(f"Python: {sys.version.split()[0]}")

    all_results: list[TestResult] = []

    nba_results = run_nba_api_tests_fixed()
    all_results.extend(nba_results)

    game_id = None
    bbref_url = None
    for r in nba_results:
        if r.details.get("game_id"):
            game_id = r.details["game_id"]
        if r.details.get("bbref_url"):
            bbref_url = r.details["bbref_url"]
        if game_id and bbref_url:
            break

    all_results.extend(run_soccerdata_tests())
    all_results.extend(run_bbref_tests(game_id, bbref_url))

    print_final_summary(all_results)
    return 0 if all(r.success for r in all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
