"""nba_api wrapper: browser headers, backoff, hybrid Parquet cache."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from nba_fit.config.endpoints import ENDPOINT_REGISTRY
from nba_fit.config.settings import (
    BROWSER_HEADERS,
    DEFAULT_GAME_ID,
    DEFAULT_LEAGUE_ID,
    DEFAULT_PLAYER_ID,
    DEFAULT_SEASON,
    DEFAULT_TEAM_ID,
    REQUEST_MAX_RETRIES,
    REQUEST_SLEEP_BACKOFF_FACTOR,
    REQUEST_SLEEP_INITIAL_SEC,
    REQUEST_SLEEP_MAX_SEC,
    REQUEST_TIMEOUT_SEC,
    Settings,
    get_settings,
)

_HEADERS_PATCHED = False

# Context resolvers (aligned with probe_all_nba_endpoints.py)
_PARAM_VALUES: dict[str, Any] = {
    "game_id": DEFAULT_GAME_ID,
    "game_id_nullable": DEFAULT_GAME_ID,
    "player_id": DEFAULT_PLAYER_ID,
    "person_id": DEFAULT_PLAYER_ID,
    "player_id_nullable": DEFAULT_PLAYER_ID,
    "team_id": DEFAULT_TEAM_ID,
    "team_id_nullable": str(DEFAULT_TEAM_ID),
    "season": DEFAULT_SEASON,
    "season_nullable": DEFAULT_SEASON,
    "season_year": DEFAULT_SEASON,
    "league_id": DEFAULT_LEAGUE_ID,
    "league_id_nullable": DEFAULT_LEAGUE_ID,
    "season_type_all_star": "Regular Season",
    "season_type": "Regular Season",
    "per_mode_simple": "PerGame",
    "per_mode_detailed": "PerGame",
    "measure_type_detailed_defense": "Base",
    "measure_type_simple": "Base",
    "context_measure_simple": "FGA",
    "player_or_team_abbreviation": "P",
    "is_only_current_season": 1,
}


@dataclass
class FetchResult:
    """In-memory result of a single endpoint fetch (probe / smoke)."""

    endpoint: str
    season: str
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    kwargs_used: dict[str, Any] = field(default_factory=dict)
    from_cache: bool = False
    cache_path: Path | None = None
    elapsed_sec: float = 0.0


def patch_nba_stats_headers() -> None:
    """Apply browser-like headers to nba_api (stats.nba.com WAF bypass)."""
    global _HEADERS_PATCHED
    if _HEADERS_PATCHED:
        return
    from nba_api.stats.library.http import NBAStatsHTTP

    NBAStatsHTTP.headers = {**NBAStatsHTTP.headers, **BROWSER_HEADERS}
    _HEADERS_PATCHED = True


class _BackoffSleeper:
    def __init__(self) -> None:
        self._delay = REQUEST_SLEEP_INITIAL_SEC

    def sleep(self) -> None:
        time.sleep(self._delay)
        self._delay = min(self._delay * REQUEST_SLEEP_BACKOFF_FACTOR, REQUEST_SLEEP_MAX_SEC)

    def reset(self) -> None:
        self._delay = REQUEST_SLEEP_INITIAL_SEC


def _get_endpoint_class(endpoint_name: str) -> type:
    from nba_api.stats.endpoints._base import Endpoint as StatsEndpoint

    module = importlib.import_module(f"nba_api.stats.endpoints.{endpoint_name}")
    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, StatsEndpoint) and obj is not StatsEndpoint
    ]
    if not candidates:
        raise ImportError(f"No Endpoint class for {endpoint_name}")
    for candidate in candidates:
        if getattr(candidate, "endpoint", None) == endpoint_name:
            return candidate
    return candidates[0]


def _resolve_kwargs(endpoint_name: str, cls: type, season: str, extra: dict[str, Any]) -> dict[str, Any]:
    spec = ENDPOINT_REGISTRY.get(endpoint_name)
    registry_defaults = dict(spec.default_kwargs) if spec else {}
    merged = {**registry_defaults, **extra}
    kwargs: dict[str, Any] = {"timeout": REQUEST_TIMEOUT_SEC, **merged}
    sig = inspect.signature(cls.__init__)
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "proxy", "headers", "get_request"):
            continue
        if param_name in kwargs:
            continue
        if param_name in ("season", "season_nullable", "season_year"):
            kwargs[param_name] = season
            continue
        if param_name in _PARAM_VALUES:
            kwargs[param_name] = _PARAM_VALUES[param_name]
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        lower = param_name.lower()
        if "game" in lower and "id" in lower:
            kwargs[param_name] = DEFAULT_GAME_ID
        elif "player" in lower and "id" in lower:
            kwargs[param_name] = DEFAULT_PLAYER_ID
        elif "team" in lower and "id" in lower:
            kwargs[param_name] = DEFAULT_TEAM_ID
        elif "season" in lower:
            kwargs[param_name] = season
        elif "league" in lower:
            kwargs[param_name] = DEFAULT_LEAGUE_ID
    return kwargs


def _flatten_columns(cols: Any) -> list[str]:
    if isinstance(cols, pd.MultiIndex):
        return [
            " | ".join(str(p) for p in tup if str(p) not in ("", "nan"))
            for tup in cols
        ]
    return [str(c) for c in cols]


def _frames_from_endpoint(ep: Any) -> dict[str, pd.DataFrame]:
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
    out: dict[str, pd.DataFrame] = {}
    for name, df in pairs:
        copy = df.copy()
        copy.columns = _flatten_columns(copy.columns)
        out[name] = copy
    return out


def _cache_key_suffix(kwargs: dict[str, Any]) -> str:
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


class NBAClient:
    """Fetch stats endpoints with retry, backoff, and Parquet cache."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._sleeper = _BackoffSleeper()
        self.source = "nba_api"

    def _cache_file(self, endpoint: str, season: str, kwargs: dict[str, Any]) -> Path:
        base = self.settings.raw_parquet_dir(self.source, endpoint, season)
        base.mkdir(parents=True, exist_ok=True)
        suffix = _cache_key_suffix(kwargs)
        return base / f"data_{suffix}.parquet"

    def _cache_meta_file(self, parquet_path: Path) -> Path:
        return parquet_path.with_suffix(".meta.json")

    def _is_cache_fresh(self, parquet_path: Path) -> bool:
        if not parquet_path.exists():
            return False
        age_hours = (time.time() - parquet_path.stat().st_mtime) / 3600.0
        return age_hours < self.settings.cache_ttl_hours

    def _write_cache(
        self,
        parquet_path: Path,
        frames: dict[str, pd.DataFrame],
        meta: dict[str, Any],
    ) -> None:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(
            [df.assign(_dataset=name) for name, df in frames.items()],
            ignore_index=True,
        )
        combined.to_parquet(parquet_path, index=False)
        meta_path = self._cache_meta_file(parquet_path)
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def _read_cache(self, parquet_path: Path) -> dict[str, pd.DataFrame]:
        combined = pd.read_parquet(parquet_path)
        if "_dataset" not in combined.columns:
            return {"data": combined}
        out: dict[str, pd.DataFrame] = {}
        for name, group in combined.groupby("_dataset", sort=False):
            out[str(name)] = group.drop(columns=["_dataset"]).reset_index(drop=True)
        return out

    def fetch(
        self,
        endpoint: str,
        *,
        season: str | None = None,
        use_cache: bool = True,
        persist_cache: bool = True,
        **kwargs: Any,
    ) -> FetchResult:
        """
        Fetch one endpoint into memory.

        For bulk PBP, pass game_id per call; do not loop inside this method.
        """
        season = season or self.settings.default_season
        cls = _get_endpoint_class(endpoint)
        call_kwargs = _resolve_kwargs(endpoint, cls, season, kwargs)
        cache_path = self._cache_file(endpoint, season, call_kwargs)

        if use_cache and self._is_cache_fresh(cache_path):
            frames = self._read_cache(cache_path)
            return FetchResult(
                endpoint=endpoint,
                season=season,
                frames=frames,
                kwargs_used=call_kwargs,
                from_cache=True,
                cache_path=cache_path,
            )

        patch_nba_stats_headers()
        last_exc: Exception | None = None
        start = time.time()
        for attempt in range(REQUEST_MAX_RETRIES):
            self._sleeper.sleep()
            try:
                ep = cls(**call_kwargs)
                frames = _frames_from_endpoint(ep)
                elapsed = time.time() - start
                self._sleeper.reset()
                if persist_cache and frames:
                    meta = {
                        "endpoint": endpoint,
                        "season": season,
                        "kwargs": {k: v for k, v in call_kwargs.items() if k != "timeout"},
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                        "datasets": list(frames.keys()),
                    }
                    self._write_cache(cache_path, frames, meta)
                return FetchResult(
                    endpoint=endpoint,
                    season=season,
                    frames=frames,
                    kwargs_used=call_kwargs,
                    from_cache=False,
                    cache_path=cache_path if persist_cache else None,
                    elapsed_sec=elapsed,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt + 1 >= REQUEST_MAX_RETRIES:
                    break
        raise RuntimeError(
            f"Failed to fetch {endpoint} after {REQUEST_MAX_RETRIES} attempts"
        ) from last_exc
