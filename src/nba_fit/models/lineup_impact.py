"""Lineup-aware impact projection (Option C interim proxy before full RAPM)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_fit.config.settings import ONOFF_STAT_NET_RATING
from nba_fit.features.constants import COL_PLAYER_ID, COL_TEAM_ID
from nba_fit.models.constants import LINEUP_REPLACEMENT_BLEND, VECTOR_NORM_EPSILON


@dataclass(frozen=True)
class LineupUnitProjection:
    """Projected change for one five-man unit after a minutes replacement."""

    lineup_key: str
    lineup_label: str
    player_ids: tuple[int, ...]
    baseline_net_rating: float
    projected_net_rating_delta: float
    minutes: float
    replaced_player_id: int | None


def player_net_rating_map(players_raw: pd.DataFrame) -> dict[int, float]:
    """Map player_id -> season net rating (E_NET_RATING fallback)."""
    if players_raw.empty:
        return {}
    work = players_raw.copy()
    if "player_id" in work.columns:
        pid_col = "player_id"
    elif COL_PLAYER_ID in work.columns:
        pid_col = COL_PLAYER_ID
    elif "PLAYER_ID" in work.columns:
        pid_col = "PLAYER_ID"
    else:
        return {}
    net_col = ONOFF_STAT_NET_RATING
    if net_col not in work.columns:
        for alt in ("E_NET_RATING", "PLUS_MINUS"):
            if alt in work.columns:
                net_col = alt
                break
        else:
            return {}
    out: dict[int, float] = {}
    for _, row in work.iterrows():
        pid = row[pid_col]
        if pd.isna(pid):
            continue
        val = row.get(net_col)
        if pd.isna(val):
            continue
        out[int(pid)] = float(val)
    return out


def baseline_net_rating_from_row(row: pd.Series) -> float:
    """Observed lineup net rating from a lineup_units row."""
    for col in (ONOFF_STAT_NET_RATING, "NET_RATING", "E_NET_RATING"):
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    if "PLUS_MINUS" in row.index and "MIN" in row.index:
        mins = float(row["MIN"]) if pd.notna(row["MIN"]) and float(row["MIN"]) > 0 else 1.0
        return float(row["PLUS_MINUS"]) / mins * 48.0
    return 0.0


def project_lineup_replacement(
    *,
    lineup_player_ids: tuple[int, ...],
    candidate_id: int,
    impact_by_player: dict[int, float],
    baseline_net_rating: float,
    minutes: float,
    blend: float = LINEUP_REPLACEMENT_BLEND,
) -> LineupUnitProjection | None:
    """
    Project net-rating delta when *candidate_id* replaces the weakest lineup member.

    Uses a partial substitution blend (not full additive RAPM) until stint-level
    models are wired in.
    """
    if candidate_id in lineup_player_ids or len(lineup_player_ids) < 5:
        return None
    candidate_net = impact_by_player.get(candidate_id)
    if candidate_net is None:
        return None

    lineup_members = [pid for pid in lineup_player_ids if pid in impact_by_player]
    if len(lineup_members) < 4:
        return None

    replaced_id = min(lineup_members, key=lambda pid: impact_by_player[pid])
    replaced_net = impact_by_player[replaced_id]
    delta = blend * (candidate_net - replaced_net)
    projected = baseline_net_rating + delta

    label_ids = list(lineup_player_ids)
    try:
        idx = label_ids.index(replaced_id)
        label_ids[idx] = candidate_id
    except ValueError:
        label_ids = label_ids[:4] + [candidate_id]

    return LineupUnitProjection(
        lineup_key="-".join(str(i) for i in sorted(label_ids)),
        lineup_label=f"unit:{','.join(str(i) for i in label_ids)}",
        player_ids=tuple(label_ids),
        baseline_net_rating=baseline_net_rating,
        projected_net_rating_delta=projected - baseline_net_rating,
        minutes=minutes,
        replaced_player_id=replaced_id,
    )


def aggregate_projected_delta(
    projections: list[LineupUnitProjection],
) -> float | None:
    """Minutes-weighted mean projected net-rating delta across lineup units."""
    if not projections:
        return None
    weights = np.array([max(p.minutes, 1.0) for p in projections], dtype=float)
    deltas = np.array([p.projected_net_rating_delta for p in projections], dtype=float)
    if weights.sum() <= VECTOR_NORM_EPSILON:
        return float(np.mean(deltas))
    return float(np.average(deltas, weights=weights))
