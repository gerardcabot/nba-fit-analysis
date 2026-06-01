#!/usr/bin/env python3
"""2D archetype map: players colored by trained archetype labels."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from nba_fit.models.constants import (
    ARCHETYPE_MAP_METHOD,
    ARCHETYPE_MAP_N_COMPONENTS,
    MODEL_RANDOM_STATE,
)
from nba_fit.models.role_context import RoleFitContext
from visual_tests._cli import resolve_season_team
from visual_tests._plot_utils import apply_plot_style, save_figure
from visual_tests._role_loader import load_trained_role_context


def _project_2d(embeddings: np.ndarray) -> np.ndarray:
    if ARCHETYPE_MAP_METHOD == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_components=ARCHETYPE_MAP_N_COMPONENTS,
                random_state=MODEL_RANDOM_STATE,
            )
            return reducer.fit_transform(embeddings)
        except ImportError:
            pass
    pca = PCA(n_components=ARCHETYPE_MAP_N_COMPONENTS, random_state=MODEL_RANDOM_STATE)
    return pca.fit_transform(embeddings)


def main(
    season: str | None = None,
    team_id: int | None = None,
    *,
    role_context: RoleFitContext | None = None,
) -> int:
    season, team_id = resolve_season_team(season=season, team_id=team_id)
    _ = team_id  # season/team scope for validation runner parity

    role = role_context or load_trained_role_context(season)
    coords = _project_2d(role.embeddings.embeddings)

    labels = pd.Series(
        role.archetypes.archetype_labels,
        index=role.archetypes.player_ids,
        name="archetype",
    )
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1] if coords.shape[1] > 1 else 0.0,
            "archetype": labels.loc[role.embeddings.player_ids].to_numpy(),
        }
    )
    n_points = len(df)

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    for archetype, group in df.groupby("archetype"):
        ax.scatter(
            group["x"],
            group["y"],
            label=str(archetype),
            alpha=0.75,
            s=40,
        )
    ax.set_title(
        f"Player archetype map — {season} "
        f"({n_points} players, trained role embeddings)"
    )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()

    path = save_figure(fig, "07_archetype_map", subdir="roles")
    print(f"Wrote {path} (season={season}, n_points={n_points})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
