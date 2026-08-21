"""Movement labels from game-log team changes or manual transaction CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from nba_fit.config.settings import get_settings
from nba_fit.data.client import FetchResult, NBAClient

# Canonical columns for manual / derived movement tables
MOVEMENT_COLUMNS: tuple[str, ...] = (
    "player_id",
    "from_team_id",
    "to_team_id",
    "move_date",
    "season",
    "movement_type",
)

# Required columns when curating ``movements.csv`` (``season`` optional in file).
MANUAL_MOVEMENT_TEMPLATE_COLUMNS: tuple[str, ...] = (
    "player_id",
    "from_team_id",
    "to_team_id",
    "move_date",
    "movement_type",
)

_MANUAL_TEMPLATE_NAME = "movements_template.csv"
_README_NAME = "README.md"


@dataclass(frozen=True)
class MovementEvent:
    """One detected or curated player movement (trade, signing, waiver)."""

    player_id: int
    from_team_id: int
    to_team_id: int
    move_date: pd.Timestamp
    season: str
    movement_type: str = "team_change"

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "from_team_id": self.from_team_id,
            "to_team_id": self.to_team_id,
            "move_date": self.move_date,
            "season": self.season,
            "movement_type": self.movement_type,
        }


def transactions_dir() -> Path:
    """
    ``{NBA_FIT_DATA_ROOT or repo}/raw/transactions/`` — manual CSVs and caches.

    Override the data root with env ``NBA_FIT_DATA_ROOT`` (see ``get_settings``).
    Movement parquet caches: ``movements_{season}.parquet``.
    """
    path = get_settings().data_raw / "transactions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def movements_cache_path(season: str) -> Path:
    return transactions_dir() / f"movements_{season.replace('/', '-')}.parquet"


def manual_csv_path(name: str = "movements.csv") -> Path:
    return transactions_dir() / name


def ensure_stub_template() -> Path:
    """
    Write a CSV template and README if missing (no network).

    Copy ``movements_template.csv`` to ``movements.csv`` and fill rows for
    trades/signings when game-log inference is insufficient.
    """
    root = transactions_dir()
    template = root / _MANUAL_TEMPLATE_NAME
    if not template.exists():
        pd.DataFrame(
            [
                {
                    "player_id": 2544,
                    "from_team_id": 1610612747,
                    "to_team_id": 1610612748,
                    "move_date": "2024-07-06",
                    "season": "2024-25",
                    "movement_type": "free_agent",
                }
            ]
        ).to_csv(template, index=False)
    readme = root / _README_NAME
    if not readme.exists():
        readme.write_text(
            "# Player movement labels (Option D)\n\n"
            "Sources (in priority order when calling ``get_movements``):\n\n"
            "1. ``movements_{season}.parquet`` — cached derived labels\n"
            "2. ``movements.csv`` — curated manual file (copy from template)\n"
            "3. ``fetch_movements_from_gamelogs`` — infers team changes from "
            "``playergamelogs`` (requires API or warm cache)\n"
            "4. ``synthetic_movements`` — deterministic demo rows for tests\n\n"
            "Manual CSV template columns (``season`` optional in file): "
            + ", ".join(MANUAL_MOVEMENT_TEMPLATE_COLUMNS)
            + "\n\n"
            "Cached parquet includes: "
            + ", ".join(MOVEMENT_COLUMNS)
            + "\n",
            encoding="utf-8",
        )
    return template


def _normalize_movements(df: pd.DataFrame, *, season: str | None = None) -> pd.DataFrame:
    missing = [c for c in MOVEMENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Movement table missing columns: {missing}")
    out = df[list(MOVEMENT_COLUMNS)].copy()
    out["player_id"] = out["player_id"].astype(int)
    out["from_team_id"] = out["from_team_id"].astype(int)
    out["to_team_id"] = out["to_team_id"].astype(int)
    out["move_date"] = pd.to_datetime(out["move_date"])
    if season is not None:
        out["season"] = season
    out["movement_type"] = out["movement_type"].fillna("team_change").astype(str)
    return out.sort_values("move_date").reset_index(drop=True)


def load_manual_movements(path: Path | None = None, *, season: str | None = None) -> pd.DataFrame:
    """Load curated ``movements.csv`` if present; else empty frame with schema."""
    ensure_stub_template()
    path = path or manual_csv_path()
    if not path.exists():
        return pd.DataFrame(columns=list(MOVEMENT_COLUMNS))
    raw = pd.read_csv(path)
    if "season" not in raw.columns and season is not None:
        raw["season"] = season
    return _normalize_movements(raw, season=season)


def save_movements_cache(df: pd.DataFrame, season: str) -> Path:
    path = movements_cache_path(season)
    _normalize_movements(df, season=season).to_parquet(path, index=False)
    return path


def load_movements_cache(season: str) -> pd.DataFrame:
    path = movements_cache_path(season)
    if not path.exists():
        raise FileNotFoundError(path)
    return _normalize_movements(pd.read_parquet(path), season=season)


def _gamelog_primary_frame(result: FetchResult) -> pd.DataFrame:
    for key in ("PlayerGameLogs", "player_game_logs", "dataset_0"):
        if key in result.frames:
            return result.frames[key]
    if len(result.frames) == 1:
        return next(iter(result.frames.values()))
    raise KeyError(f"No game log frame in {result.endpoint}: {list(result.frames)}")


def _parse_game_minutes(min_val: Any) -> float:
    if pd.isna(min_val):
        return 0.0
    if isinstance(min_val, (int, float)):
        return float(min_val)
    text = str(min_val).strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            return float(parts[0]) + float(parts[1]) / 60.0
        if len(parts) == 3:
            return float(parts[0]) * 60 + float(parts[1]) + float(parts[2]) / 60.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def infer_movements_from_gamelogs(logs: pd.DataFrame, *, season: str) -> pd.DataFrame:
    """
    Detect team changes by ordering each player's games by date.

    Uses ``PLAYER_ID``, ``TEAM_ID``, and ``GAME_DATE`` (or ``GAME_ID`` proxy).
    """
    if logs.empty:
        return pd.DataFrame(columns=list(MOVEMENT_COLUMNS))

    col_map = {c.upper(): c for c in logs.columns}
    pid_col = col_map.get("PLAYER_ID", "PLAYER_ID")
    team_col = col_map.get("TEAM_ID", "TEAM_ID")
    date_col = col_map.get("GAME_DATE") or col_map.get("GAME_ID", "GAME_DATE")

    work = logs.copy()
    work["_player_id"] = work[pid_col].astype(int)
    work["_team_id"] = work[team_col].astype(int)
    work["_game_date"] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=["_game_date"]).sort_values(
        ["_player_id", "_game_date"]
    )

    rows: list[dict[str, Any]] = []
    for player_id, grp in work.groupby("_player_id", sort=False):
        prev_team: int | None = None
        for _, row in grp.iterrows():
            team_id = int(row["_team_id"])
            if prev_team is not None and team_id != prev_team:
                rows.append(
                    {
                        "player_id": int(player_id),
                        "from_team_id": prev_team,
                        "to_team_id": team_id,
                        "move_date": row["_game_date"],
                        "season": season,
                        "movement_type": "team_change",
                    }
                )
            prev_team = team_id

    if not rows:
        return pd.DataFrame(columns=list(MOVEMENT_COLUMNS))
    return _normalize_movements(pd.DataFrame(rows), season=season)


def fetch_movements_from_gamelogs(
    client: NBAClient,
    *,
    season: str,
    use_cache: bool = True,
    persist_cache: bool = True,
    extra_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Pull ``playergamelogs`` and infer roster team changes.

    For league-wide logs, pass ``player_id_nullable=""`` in *extra_kwargs*
    when the endpoint allows it; otherwise the client default probes one player.
    """
    kwargs = {"player_id_nullable": "", "team_id_nullable": ""}
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    result = client.fetch(
        "playergamelogs",
        season=season,
        use_cache=use_cache,
        persist_cache=persist_cache,
        **kwargs,
    )
    logs = _gamelog_primary_frame(result)
    movements = infer_movements_from_gamelogs(logs, season=season)
    if persist_cache and not movements.empty:
        save_movements_cache(movements, season)
    return movements


def synthetic_movements(
    season: str,
    *,
    player_ids: tuple[int, ...] = (2544, 201939, 203507),
    n_moves: int = 3,
) -> pd.DataFrame:
    """Deterministic movements for unit tests and offline backtests."""
    teams = (1610612747, 1610612748, 1610612752, 1610612738)
    rows: list[dict[str, Any]] = []
    base_date = pd.Timestamp(f"{season.split('-')[0]}-11-01")
    for i in range(n_moves):
        pid = player_ids[i % len(player_ids)]
        from_t = teams[i % len(teams)]
        to_t = teams[(i + 1) % len(teams)]
        rows.append(
            {
                "player_id": pid,
                "from_team_id": from_t,
                "to_team_id": to_t,
                "move_date": base_date + pd.Timedelta(days=14 * i),
                "season": season,
                "movement_type": "synthetic",
            }
        )
    return _normalize_movements(pd.DataFrame(rows), season=season)


def load_gamelogs_for_season(
    season: str,
    *,
    client: NBAClient | None = None,
    allow_api: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load league ``playergamelogs`` from cache or API when *allow_api* is True."""
    if allow_api or client is not None:
        client = client or NBAClient()
        result = client.fetch(
            "playergamelogs",
            season=season,
            use_cache=use_cache,
            persist_cache=True,
            player_id_nullable="",
        )
        return _gamelog_primary_frame(result)
    cache_dir = get_settings().raw_parquet_dir("nba_api", "playergamelogs", season)
    if not cache_dir.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(cache_dir.glob("data_*.parquet")):
        combined = pd.read_parquet(path)
        if "_dataset" in combined.columns:
            for _, group in combined.groupby("_dataset", sort=False):
                frames.append(group.drop(columns=["_dataset"], errors="ignore"))
        else:
            frames.append(combined)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def warm_movements_cache(
    seasons: tuple[str, ...] | list[str],
    *,
    allow_api: bool = True,
    client: NBAClient | None = None,
) -> dict[str, int]:
    """
    Fetch and persist ``movements_{season}.parquet`` for each season.

    Returns ``{season: n_movements}`` (0 when inference yields no rows).
    """
    client = client or (NBAClient() if allow_api else None)
    counts: dict[str, int] = {}
    for season in seasons:
        mv = get_movements(
            season,
            client=client,
            allow_api=allow_api,
            use_synthetic_fallback=False,
        )
        counts[season] = len(mv)
    return counts


def get_movements(
    season: str,
    *,
    client: NBAClient | None = None,
    prefer_manual: bool = True,
    allow_api: bool = False,
    use_synthetic_fallback: bool = True,
) -> pd.DataFrame:
    """
    Resolve movement labels for *season* without surprising network I/O.

    Order: parquet cache → manual CSV → API (if *allow_api*) → synthetic.

    When *allow_api* is True and *client* is omitted, an :class:`NBAClient` is
    created. Successful API inference writes
    ``raw/transactions/movements_{season}.parquet``.
    """
    cache_path = movements_cache_path(season)
    if cache_path.exists():
        return load_movements_cache(season)

    if prefer_manual:
        manual = load_manual_movements()
        manual_season = manual.loc[manual["season"] == season] if not manual.empty else manual
        if not manual_season.empty:
            save_movements_cache(manual_season, season)
            return manual_season.reset_index(drop=True)

    if allow_api:
        api_client = client or NBAClient()
        try:
            return fetch_movements_from_gamelogs(api_client, season=season)
        except Exception:  # noqa: BLE001
            pass

    if use_synthetic_fallback:
        return synthetic_movements(season)

    return pd.DataFrame(columns=list(MOVEMENT_COLUMNS))
