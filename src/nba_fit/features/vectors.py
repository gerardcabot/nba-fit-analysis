"""Typed player/team vectors passed into the scoring layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nba_fit.features.constants import PLAYER_FEATURE_PREFIX, TEAM_FEATURE_PREFIX


@dataclass(frozen=True)
class PlayerVector:
    """Scaled player feature vector (z_pf_* columns)."""

    player_id: int
    season: str
    values: np.ndarray
    feature_names: tuple[str, ...]

    def select_groups(self, *groups: str) -> np.ndarray:
        """Return values for feature groups (e.g. ``role_volume``)."""
        prefixes = tuple(f"z_{PLAYER_FEATURE_PREFIX}{g}__" for g in groups)
        idx = [i for i, n in enumerate(self.feature_names) if n.startswith(prefixes)]
        if not idx:
            return np.array([], dtype=float)
        return self.values[idx]


@dataclass(frozen=True)
class TeamVector:
    """Scaled team need/style vector (z_tf_* columns)."""

    team_id: int
    season: str
    values: np.ndarray
    feature_names: tuple[str, ...]
    display_name: str = ""

    def select_groups(self, *groups: str) -> np.ndarray:
        prefixes = tuple(f"z_{TEAM_FEATURE_PREFIX}{g}__" for g in groups)
        idx = [i for i, n in enumerate(self.feature_names) if n.startswith(prefixes)]
        if not idx:
            return np.array([], dtype=float)
        return self.values[idx]
