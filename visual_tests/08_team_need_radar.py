#!/usr/bin/env python3
"""Radar chart of team need dimensions (Option B archetype gaps + weaknesses)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

from nba_fit.features.season_context import SeasonFitContext
from nba_fit.models.role_context import RoleFitContext
from visual_tests._cli import resolve_season_team
from visual_tests._plot_utils import apply_plot_style, save_figure
from visual_tests._role_loader import load_trained_role_context


def main(
    season: str | None = None,
    team_id: int | None = None,
    *,
    role_context: RoleFitContext | None = None,
) -> int:
    season, team_id = resolve_season_team(season=season, team_id=team_id)

    role_ctx = role_context
    context: SeasonFitContext | None = None
    if role_ctx is None:
        role_ctx = load_trained_role_context(season)
        context = SeasonFitContext.build(season, prefer_interim=True, prefer_api=False)
    else:
        context = SeasonFitContext.build(season, prefer_interim=True, prefer_api=False)

    profile = role_ctx.team_needs.get(team_id)
    if profile is None:
        print(f"No team need profile for team {team_id}", file=sys.stderr)
        return 1

    arch_labels = list(profile.archetype_labels)
    arch_vals = profile.values[: profile.n_archetypes].tolist()
    weak_names = list(profile.weakness_feature_names)
    weak_vals = profile.values[profile.n_archetypes :].tolist()

    labels = arch_labels + [f"weak_{n}" for n in weak_names]
    values = arch_vals + weak_vals
    if not labels:
        print("Empty team need profile", file=sys.stderr)
        return 1

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_cycle = values + values[:1]
    angles_cycle = angles + angles[:1]

    apply_plot_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.plot(angles_cycle, values_cycle, "o-", linewidth=2, color="#4c72b0")
    ax.fill(angles_cycle, values_cycle, alpha=0.25, color="#4c72b0")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, size=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    team_name = context.teams.get(team_id)
    title_team = team_name.display_name if team_name else str(team_id)
    ax.set_title(
        f"Team need radar — {title_team} ({season})\n"
        "Higher = larger archetype gap or weakness proxy"
    )
    fig.tight_layout()

    path = save_figure(fig, "08_team_need_radar", subdir="option_b")
    print(
        f"Wrote {path} (season={season}, team_id={team_id}, "
        f"n_axes={len(labels)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
