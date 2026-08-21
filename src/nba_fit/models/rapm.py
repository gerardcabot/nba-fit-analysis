"""Ridge RAPM-style offensive/defensive impact from lineup stint matrices."""

from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from nba_fit.config.settings import (
    MIN_POSSESSIONS_RAPM_FLAG,
    RAPM_RECENCY_HALF_LIFE_GAMES,
    get_settings,
)
from nba_fit.features.constants import COL_PLAYER_ID
from nba_fit.models.constants import MODEL_RANDOM_STATE, RAPM_RIDGE_ALPHA

_GROUP_ID_SPLIT = re.compile(r"[-,\s|]+")
_LEAGUE_AVG_OFF_RATING: float = 110.0
_LEAGUE_AVG_DEF_RATING: float = 110.0

logger = logging.getLogger(__name__)


@dataclass
class RapmArtifacts:
    """Season-level regularized APM coefficients (offense / defense split)."""

    season: str
    player_ids: np.ndarray
    orapm: np.ndarray
    drapm: np.ndarray
    net_rapm: np.ndarray
    stint_possessions: np.ndarray
    low_sample_flag: np.ndarray
    ridge_alpha: float
    recency_half_life_games: float
    degenerate: bool = False
    rapm_source: str = "lineup_units"
    metadata: dict[str, object] = field(default_factory=dict)

    def impact_for(self, player_id: int) -> tuple[float, float, float] | None:
        hits = np.where(self.player_ids == player_id)[0]
        if len(hits) == 0:
            return None
        i = int(hits[0])
        return float(self.orapm[i]), float(self.drapm[i]), float(self.net_rapm[i])


def parse_lineup_player_ids(group_id: object) -> list[int]:
    """Parse player IDs from NBA ``GROUP_ID`` (dash-separated integers)."""
    if group_id is None or (isinstance(group_id, float) and np.isnan(group_id)):
        return []
    parts = _GROUP_ID_SPLIT.split(str(group_id).strip())
    ids: list[int] = []
    for part in parts:
        if part.isdigit():
            pid = int(part)
            if pid > 0:
                ids.append(pid)
    return ids


def recency_weights(
    n_stints: int,
    half_life_games: float | None = None,
    *,
    newest_last: bool = True,
) -> np.ndarray:
    """
    Exponential recency weights for stint rows.

    With ``newest_last=True`` (default), the last row is the most recent stint and
    receives weight 1.0; earlier rows decay as ``0.5 ** (age / half_life_games)``.
    """
    if n_stints <= 0:
        return np.array([], dtype=float)
    hl = half_life_games if half_life_games is not None else RAPM_RECENCY_HALF_LIFE_GAMES
    ages = np.arange(n_stints, dtype=float)
    if newest_last:
        ages = ages[::-1]
    return np.power(0.5, ages / hl, dtype=float)


def build_player_dummy_matrix(
    stint_players: list[list[int]],
    *,
    player_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build stint × player indicator matrix (+1 when player is on court)."""
    if player_ids is None:
        all_ids = sorted({pid for stint in stint_players for pid in stint})
        player_ids = np.asarray(all_ids, dtype=int)
    id_to_idx = {int(pid): i for i, pid in enumerate(player_ids)}
    n_stints = len(stint_players)
    n_players = len(player_ids)
    matrix = np.zeros((n_stints, n_players), dtype=float)
    for row, stint in enumerate(stint_players):
        for pid in stint:
            col = id_to_idx.get(int(pid))
            if col is not None:
                matrix[row, col] = 1.0
    return matrix, player_ids


def fit_ridge_impact(
    design: np.ndarray,
    targets: np.ndarray,
    sample_weight: np.ndarray | None,
    *,
    alpha: float = RAPM_RIDGE_ALPHA,
) -> np.ndarray:
    """Weighted ridge regression returning player coefficients (no intercept)."""
    if design.size == 0:
        return np.array([], dtype=float)
    model = Ridge(alpha=alpha, fit_intercept=True, random_state=MODEL_RANDOM_STATE)
    model.fit(design, targets, sample_weight=sample_weight)
    return np.asarray(model.coef_, dtype=float)


def fit_rapm_from_stints(
    stint_players: list[list[int]],
    off_targets: np.ndarray,
    def_targets: np.ndarray,
    *,
    season: str,
    stint_weights: np.ndarray | None = None,
    alpha: float = RAPM_RIDGE_ALPHA,
    half_life_games: float | None = None,
    recency_weight_arr: np.ndarray | None = None,
) -> RapmArtifacts:
    """
    Fit offensive and defensive ridge APM on the same stint player matrix.

    * ``off_targets`` — points per 100 possessions scored (``OFF_RATING``).
    * ``def_targets`` — opponent points per 100 allowed (``DEF_RATING``); lower is better
      defense, so ``drapm`` coefficients are negated so positive = good defender.
    """
    design, player_ids = build_player_dummy_matrix(stint_players)
    n = design.shape[0]
    if recency_weight_arr is None and half_life_games is not None:
        recency_weight_arr = recency_weights(n, half_life_games)
    weights = stint_weights
    if weights is not None and recency_weight_arr is not None:
        weights = np.asarray(weights, dtype=float) * np.asarray(recency_weight_arr, dtype=float)
    elif recency_weight_arr is not None:
        weights = np.asarray(recency_weight_arr, dtype=float)

    orapm = fit_ridge_impact(design, off_targets, weights, alpha=alpha)
    drapm_raw = fit_ridge_impact(design, def_targets, weights, alpha=alpha)
    drapm = -drapm_raw
    net = orapm + drapm

    # Stint possessions proxy: weighted stint exposure per player
    stint_poss = np.zeros(len(player_ids), dtype=float)
    if weights is not None and len(weights) == n:
        for row, stint in enumerate(stint_players):
            w = float(weights[row])
            for pid in stint:
                hits = np.where(player_ids == pid)[0]
                if len(hits):
                    stint_poss[int(hits[0])] += w

    low_sample = stint_poss < MIN_POSSESSIONS_RAPM_FLAG

    return RapmArtifacts(
        season=season,
        player_ids=player_ids,
        orapm=orapm,
        drapm=drapm,
        net_rapm=net,
        stint_possessions=stint_poss,
        low_sample_flag=low_sample,
        ridge_alpha=alpha,
        recency_half_life_games=half_life_games or RAPM_RECENCY_HALF_LIFE_GAMES,
    )


def stints_from_lineup_units(lineups: pd.DataFrame) -> pd.DataFrame:
    """
    Build stint rows from interim ``lineup_units`` table.

    Expects ``GROUP_ID``, ``NET_RATING``, ``OFF_RATING``, ``DEF_RATING``, ``MIN``,
    and ``team_id`` or ``TEAM_ID``.
    """
    if lineups.empty:
        return pd.DataFrame(
            columns=[
                "player_ids",
                "team_id",
                "net_rating",
                "off_rating",
                "def_rating",
                "minutes",
                "stint_weight",
            ]
        )

    team_col = "team_id" if "team_id" in lineups.columns else "TEAM_ID"
    gid_col = "lineup_group_id" if "lineup_group_id" in lineups.columns else "GROUP_ID"

    rows: list[dict[str, object]] = []
    for _, row in lineups.iterrows():
        group_id = row.get(gid_col, row.get("GROUP_ID"))
        players = parse_lineup_player_ids(group_id)
        if len(players) < 2:
            continue
        minutes = float(row.get("MIN", 0.0) or 0.0)
        if minutes <= 0:
            continue
        rows.append(
            {
                "player_ids": players,
                "team_id": int(row[team_col]) if team_col in row and pd.notna(row[team_col]) else 0,
                "net_rating": float(row.get("NET_RATING", np.nan)),
                "off_rating": float(row.get("OFF_RATING", np.nan)),
                "def_rating": float(row.get("DEF_RATING", np.nan)),
                "minutes": minutes,
                "stint_weight": minutes,
            }
        )

    stints = pd.DataFrame(rows)
    if stints.empty:
        return stints

    for col in ("net_rating", "off_rating", "def_rating"):
        if stints[col].isna().all():
            if col == "net_rating" and not stints["off_rating"].isna().all():
                stints["net_rating"] = stints["off_rating"] - stints["def_rating"]
            else:
                stints[col] = stints[col].fillna(0.0)

    return stints


def _possession_points(row: pd.Series) -> tuple[float, float]:
    """Return (offense_pts, defense_pts) for one possession row when derivable."""
    if "points_scored" in row.index and pd.notna(row.get("points_scored")):
        off_pts = float(row["points_scored"])
        if "opponent_points_scored" in row.index and pd.notna(row.get("opponent_points_scored")):
            def_pts = float(row["opponent_points_scored"])
        else:
            def_pts = 0.0
        return off_pts, def_pts

    if "off_pts" in row.index and pd.notna(row.get("off_pts")):
        off_pts = float(row["off_pts"])
        def_pts = float(row.get("def_pts", 0.0) or 0.0)
        return off_pts, def_pts

    return 0.0, 0.0


def stints_from_possessions(possessions: pd.DataFrame) -> pd.DataFrame:
    """
    Build stint rows from interim ``possessions`` parquet.

    Groups by ``lineup_id`` + ``offense_team_id`` (or ``team_id``). Uses
    ``points_scored`` / ``opponent_points_scored`` when present; otherwise
    league-average per-100 targets with possession-count weights.
    """
    if possessions.empty:
        return pd.DataFrame(
            columns=[
                "player_ids",
                "team_id",
                "net_rating",
                "off_rating",
                "def_rating",
                "minutes",
                "stint_weight",
            ]
        )

    df = possessions.copy()
    lineup_col = "lineup_id" if "lineup_id" in df.columns else "GROUP_ID"
    team_col = "offense_team_id" if "offense_team_id" in df.columns else "team_id"
    if team_col not in df.columns:
        team_col = "TEAM_ID"

    df = df[df[lineup_col].notna()].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "player_ids",
                "team_id",
                "net_rating",
                "off_rating",
                "def_rating",
                "minutes",
                "stint_weight",
            ]
        )

    has_scoring = any(c in df.columns for c in ("points_scored", "off_pts"))
    rows: list[dict[str, object]] = []

    group_cols = [lineup_col]
    if team_col in df.columns:
        group_cols.append(team_col)

    for key, group in df.groupby(group_cols, sort=False):
        if isinstance(key, tuple):
            lineup_id, team_raw = key
        else:
            lineup_id, team_raw = key, group[team_col].iloc[0] if team_col in group.columns else 0

        players = parse_lineup_player_ids(lineup_id)
        if len(players) < 2:
            continue

        n_poss = len(group)
        if n_poss <= 0:
            continue

        min_game_id: str | None = None
        if "game_id" in group.columns:
            gids = group["game_id"].dropna().astype(str)
            if len(gids):
                min_game_id = str(sorted(gids)[0])

        off_pts_total = 0.0
        def_pts_total = 0.0
        for _, prow in group.iterrows():
            off_pts, def_pts = _possession_points(prow)
            off_pts_total += off_pts
            def_pts_total += def_pts

        if has_scoring and (off_pts_total > 0 or def_pts_total > 0):
            off_rating = 100.0 * off_pts_total / n_poss
            def_rating = 100.0 * def_pts_total / n_poss
        else:
            off_rating = _LEAGUE_AVG_OFF_RATING
            def_rating = _LEAGUE_AVG_DEF_RATING

        team_id = int(team_raw) if pd.notna(team_raw) else 0
        row: dict[str, object] = {
            "player_ids": players,
            "team_id": team_id,
            "net_rating": off_rating - def_rating,
            "off_rating": off_rating,
            "def_rating": def_rating,
            "minutes": float(n_poss),
            "stint_weight": float(n_poss),
        }
        if min_game_id is not None:
            row["min_game_id"] = min_game_id
        rows.append(row)

    stints = pd.DataFrame(rows)
    if stints.empty:
        return stints

    for col in ("net_rating", "off_rating", "def_rating"):
        if stints[col].isna().all():
            stints[col] = stints[col].fillna(0.0)

    return stints


def check_degenerate_rapm(artifacts: RapmArtifacts) -> RapmArtifacts:
    """
    Flag and warn when all ``net_rapm`` coefficients are identical (std == 0).

    Updates ``degenerate`` and ``metadata['net_rapm_std']`` on *artifacts*.
    """
    net_std = float(np.std(artifacts.net_rapm)) if len(artifacts.net_rapm) else 0.0
    degenerate = net_std == 0.0 and len(artifacts.net_rapm) > 0
    meta = dict(artifacts.metadata)
    meta["net_rapm_std"] = net_std
    meta["degenerate_rapm"] = degenerate
    if degenerate:
        warnings.warn(
            f"Degenerate RAPM for season {artifacts.season}: net_rapm std == 0 "
            f"(source={artifacts.rapm_source}, n_players={len(artifacts.player_ids)})",
            stacklevel=2,
        )
        logger.warning(
            "Degenerate RAPM season=%s source=%s n=%d",
            artifacts.season,
            artifacts.rapm_source,
            len(artifacts.player_ids),
        )
    return RapmArtifacts(
        season=artifacts.season,
        player_ids=artifacts.player_ids,
        orapm=artifacts.orapm,
        drapm=artifacts.drapm,
        net_rapm=artifacts.net_rapm,
        stint_possessions=artifacts.stint_possessions,
        low_sample_flag=artifacts.low_sample_flag,
        ridge_alpha=artifacts.ridge_alpha,
        recency_half_life_games=artifacts.recency_half_life_games,
        degenerate=degenerate,
        rapm_source=artifacts.rapm_source,
        metadata=meta,
    )


def fit_rapm_from_possessions(
    possessions: pd.DataFrame,
    *,
    season: str,
    alpha: float = RAPM_RIDGE_ALPHA,
    half_life_games: float | None = None,
    tune_hyperparameters: bool = False,
) -> RapmArtifacts:
    """Convenience: ``stints_from_possessions`` then ridge off/def split."""
    from nba_fit.config.settings import RAPM_RECENCY_HALF_LIFE_GAMES

    stints = stints_from_possessions(possessions)
    if stints.empty:
        raise ValueError("No valid possession stints with parseable lineup_id player lists")

    tune_meta: dict[str, object] = {}
    if tune_hyperparameters:
        from nba_fit.models.rapm_tuning import tune_rapm_hyperparameters

        tune = tune_rapm_hyperparameters(stints)
        alpha = tune.ridge_alpha
        half_life_games = tune.recency_half_life_games
        tune_meta = tune.metadata_fields()

    if half_life_games is None:
        half_life_games = RAPM_RECENCY_HALF_LIFE_GAMES

    stint_players = [list(map(int, s)) for s in stints["player_ids"]]
    recency = recency_weights(len(stint_players), half_life_games)
    artifacts = fit_rapm_from_stints(
        stint_players,
        stints["off_rating"].to_numpy(dtype=float),
        stints["def_rating"].to_numpy(dtype=float),
        season=season,
        stint_weights=stints["stint_weight"].to_numpy(dtype=float),
        alpha=alpha,
        recency_weight_arr=recency,
        half_life_games=half_life_games,
    )
    artifacts = RapmArtifacts(
        season=artifacts.season,
        player_ids=artifacts.player_ids,
        orapm=artifacts.orapm,
        drapm=artifacts.drapm,
        net_rapm=artifacts.net_rapm,
        stint_possessions=artifacts.stint_possessions,
        low_sample_flag=artifacts.low_sample_flag,
        ridge_alpha=artifacts.ridge_alpha,
        recency_half_life_games=artifacts.recency_half_life_games,
        rapm_source="possessions",
        metadata=tune_meta,
    )
    return check_degenerate_rapm(artifacts)


def fit_rapm_from_lineup_table(
    lineups: pd.DataFrame,
    *,
    season: str,
    alpha: float = RAPM_RIDGE_ALPHA,
    half_life_games: float | None = None,
    tune_hyperparameters: bool = False,
) -> RapmArtifacts:
    """Convenience: ``stints_from_lineup_units`` then ridge off/def split."""
    from nba_fit.config.settings import RAPM_RECENCY_HALF_LIFE_GAMES

    stints = stints_from_lineup_units(lineups)
    if stints.empty:
        raise ValueError("No valid lineup stints with parseable GROUP_ID player lists")

    tune_meta: dict[str, object] = {}
    if tune_hyperparameters:
        from nba_fit.models.rapm_tuning import tune_rapm_hyperparameters

        tune = tune_rapm_hyperparameters(stints)
        alpha = tune.ridge_alpha
        half_life_games = tune.recency_half_life_games
        tune_meta = tune.metadata_fields()

    if half_life_games is None:
        half_life_games = RAPM_RECENCY_HALF_LIFE_GAMES

    stint_players = [list(map(int, s)) for s in stints["player_ids"]]
    recency = recency_weights(len(stint_players), half_life_games)
    artifacts = fit_rapm_from_stints(
        stint_players,
        stints["off_rating"].to_numpy(dtype=float),
        stints["def_rating"].to_numpy(dtype=float),
        season=season,
        stint_weights=stints["stint_weight"].to_numpy(dtype=float),
        alpha=alpha,
        recency_weight_arr=recency,
        half_life_games=half_life_games,
    )
    artifacts = RapmArtifacts(
        season=artifacts.season,
        player_ids=artifacts.player_ids,
        orapm=artifacts.orapm,
        drapm=artifacts.drapm,
        net_rapm=artifacts.net_rapm,
        stint_possessions=artifacts.stint_possessions,
        low_sample_flag=artifacts.low_sample_flag,
        ridge_alpha=artifacts.ridge_alpha,
        recency_half_life_games=artifacts.recency_half_life_games,
        rapm_source="lineup_units",
        metadata=tune_meta,
    )
    return check_degenerate_rapm(artifacts)


def synthetic_stint_matrix(
    *,
    n_players: int = 12,
    n_stints: int = 80,
    rng: np.random.Generator | None = None,
) -> tuple[list[list[int]], np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic stint data with known ORAPM/DRAPM for tests."""
    rng = rng or np.random.default_rng(MODEL_RANDOM_STATE)
    true_orapm = rng.normal(0, 2, size=n_players)
    true_drapm = rng.normal(0, 1.5, size=n_players)
    stint_players: list[list[int]] = []
    off_targets: list[float] = []
    def_targets: list[float] = []

    for _ in range(n_stints):
        k = int(rng.integers(3, 6))
        players = sorted(rng.choice(n_players, size=k, replace=False).tolist())
        players = [int(p + 1) for p in players]
        stint_players.append(players)
        off_targets.append(float(true_orapm[np.array(players) - 1].sum() + rng.normal(0, 1)))
        def_targets.append(float(true_drapm[np.array(players) - 1].sum() + rng.normal(0, 1)))

    weights = rng.uniform(50, 200, size=n_stints)
    return stint_players, np.asarray(off_targets), np.asarray(def_targets), weights


def rapm_dir(season: str, *, root: Path | None = None) -> Path:
    settings = get_settings()
    base = root or settings.models_dir
    return base / "rapm" / f"season={season}"


def save_rapm(artifacts: RapmArtifacts, path: Path | None = None) -> Path:
    out_dir = path or rapm_dir(artifacts.season)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            COL_PLAYER_ID: artifacts.player_ids,
            "orapm": artifacts.orapm,
            "drapm": artifacts.drapm,
            "net_rapm": artifacts.net_rapm,
            "stint_possessions": artifacts.stint_possessions,
            "low_sample_flag": artifacts.low_sample_flag,
        }
    ).to_parquet(out_dir / "player_rapm.parquet", index=False)
    meta = {
        "season": artifacts.season,
        "ridge_alpha": artifacts.ridge_alpha,
        "recency_half_life_games": artifacts.recency_half_life_games,
        "n_players": int(len(artifacts.player_ids)),
        "rapm_source": artifacts.rapm_source,
        "degenerate_rapm": artifacts.degenerate,
        "net_rapm_std": float(artifacts.metadata.get("net_rapm_std", np.std(artifacts.net_rapm))),
        **{k: v for k, v in artifacts.metadata.items() if k not in ("net_rapm_std", "degenerate_rapm")},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def load_rapm(season: str, *, root: Path | None = None) -> RapmArtifacts:
    in_dir = rapm_dir(season, root=root)
    meta = json.loads((in_dir / "metadata.json").read_text(encoding="utf-8"))
    table = pd.read_parquet(in_dir / "player_rapm.parquet")
    return RapmArtifacts(
        season=meta["season"],
        player_ids=table[COL_PLAYER_ID].to_numpy(dtype=int),
        orapm=table["orapm"].to_numpy(dtype=float),
        drapm=table["drapm"].to_numpy(dtype=float),
        net_rapm=table["net_rapm"].to_numpy(dtype=float),
        stint_possessions=table["stint_possessions"].to_numpy(dtype=float),
        low_sample_flag=table["low_sample_flag"].to_numpy(dtype=bool),
        ridge_alpha=float(meta["ridge_alpha"]),
        recency_half_life_games=float(meta["recency_half_life_games"]),
        degenerate=bool(meta.get("degenerate_rapm", False)),
        rapm_source=str(meta.get("rapm_source", "lineup_units")),
        metadata={
            "net_rapm_std": float(meta.get("net_rapm_std", 0.0)),
            "degenerate_rapm": bool(meta.get("degenerate_rapm", False)),
        },
    )
