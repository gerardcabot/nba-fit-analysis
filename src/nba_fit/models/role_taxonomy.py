"""Industry role taxonomy: heuristic archetypes → BBall Index-style offensive roles."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from nba_fit.models.constants import ARCHETYPE_NOISE_LABEL

# BBall Index-style offensive role vocabulary (public taxonomy alignment).
INDUSTRY_OFFENSIVE_ROLES: tuple[str, ...] = (
    "Primary Ball Handler",
    "Secondary Ball Handler",
    "Movement Ball Handler",
    "Combo Guard",
    "Shot Creator",
    "Movement Shooter",
    "Spot Up Guard",
    "Spot Up Wing",
    "Connector",
    "Roll/Cut Big",
    "Spot Up Big",
    "Versatile Big",
    "Unknown",
)

# Cleaning the Glass coarse position groups used for lineup / need aggregation.
CTG_POSITION_GROUPS: tuple[str, ...] = ("Guard", "Wing", "Big", "Unknown")

_DEFAULT_INDUSTRY_ROLE = "Unknown"
_DEFAULT_CTG_GROUP = "Unknown"

# Fallback when CSV is unavailable (mirrors data/reference/archetype_industry_map.csv).
_HEURISTIC_TO_INDUSTRY: dict[str, str] = {
    "high_usage_creator": "Primary Ball Handler",
    "advantage_creator": "Shot Creator",
    "movement_shooter": "Movement Shooter",
    "low_usage_spacer": "Spot Up Wing",
    "connector_wing": "Connector",
    "rim_running_big": "Roll/Cut Big",
    "stretch_big": "Spot Up Big",
    "defensive_guard": "Combo Guard",
    "rim_protector": "Versatile Big",
    "rebound_finisher": "Roll/Cut Big",
    "versatile_forward": "Connector",
    ARCHETYPE_NOISE_LABEL: "Unknown",
}

_INDUSTRY_TO_CTG: dict[str, str] = {
    "Primary Ball Handler": "Guard",
    "Secondary Ball Handler": "Guard",
    "Movement Ball Handler": "Guard",
    "Combo Guard": "Guard",
    "Shot Creator": "Guard",
    "Spot Up Guard": "Guard",
    "Movement Shooter": "Wing",
    "Spot Up Wing": "Wing",
    "Connector": "Wing",
    "Roll/Cut Big": "Big",
    "Spot Up Big": "Big",
    "Versatile Big": "Big",
    "Unknown": "Unknown",
}


def _reference_map_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "reference" / "archetype_industry_map.csv"


@lru_cache(maxsize=1)
def load_archetype_industry_map(*, root: Path | None = None) -> pd.DataFrame:
    """Load heuristic → industry role mapping from ``data/reference/archetype_industry_map.csv``."""
    path = (root / "data" / "reference" / "archetype_industry_map.csv") if root else _reference_map_path()
    if path.is_file():
        df = pd.read_csv(path)
        required = {"heuristic_label", "industry_role", "ctg_position_group"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"archetype_industry_map.csv missing columns: {sorted(missing)}")
        return df
    return pd.DataFrame(
        {
            "heuristic_label": list(_HEURISTIC_TO_INDUSTRY.keys()),
            "industry_role": list(_HEURISTIC_TO_INDUSTRY.values()),
            "ctg_position_group": [
                _INDUSTRY_TO_CTG.get(role, _DEFAULT_CTG_GROUP) for role in _HEURISTIC_TO_INDUSTRY.values()
            ],
        }
    )


def map_heuristic_to_industry(label: str, *, root: Path | None = None) -> str:
    """Map a repo heuristic archetype label to an industry offensive role name."""
    key = str(label).strip()
    table = load_archetype_industry_map(root=root)
    hits = table.loc[table["heuristic_label"] == key, "industry_role"]
    if len(hits):
        return str(hits.iloc[0])
    return _HEURISTIC_TO_INDUSTRY.get(key, _DEFAULT_INDUSTRY_ROLE)


def ctg_position_group(role_bucket: str, *, root: Path | None = None) -> str:
    """Map an industry role (or heuristic label) to a CTG coarse position group."""
    bucket = str(role_bucket).strip()
    table = load_archetype_industry_map(root=root)

    industry_hits = table.loc[table["industry_role"] == bucket, "ctg_position_group"]
    if len(industry_hits):
        return str(industry_hits.iloc[0])

    heuristic_hits = table.loc[table["heuristic_label"] == bucket, "ctg_position_group"]
    if len(heuristic_hits):
        return str(heuristic_hits.iloc[0])

    industry = map_heuristic_to_industry(bucket, root=root)
    return _INDUSTRY_TO_CTG.get(industry, _DEFAULT_CTG_GROUP)


def soft_role_display(probs: Mapping[str, float] | Sequence[tuple[str, float]], *, top_n: int = 3) -> str:
    """Format soft role probabilities for fit cards and dashboards.

    Accepts either a ``{role: prob}`` mapping or an ordered sequence of
    ``(role, prob)`` pairs. Returns a comma-separated string of the top roles
    with percentage weights, e.g. ``"Shot Creator (62%), Connector (24%)"``.
    """
    if isinstance(probs, Mapping):
        items = [(str(k), float(v)) for k, v in probs.items()]
    else:
        items = [(str(k), float(v)) for k, v in probs]

    if not items:
        return _DEFAULT_INDUSTRY_ROLE

    total = sum(v for _, v in items)
    if total <= 0:
        return _DEFAULT_INDUSTRY_ROLE

    ranked = sorted(items, key=lambda kv: kv[1], reverse=True)[: max(1, top_n)]
    parts: list[str] = []
    for role, weight in ranked:
        pct = 100.0 * weight / total
        parts.append(f"{role} ({pct:.0f}%)")
    return ", ".join(parts)


def industry_roles_for_labels(labels: Sequence[str], *, root: Path | None = None) -> list[str]:
    """Vectorized helper: map heuristic labels to industry roles."""
    return [map_heuristic_to_industry(label, root=root) for label in labels]
