"""Shared similarity helpers for Option A/B submetrics (avoids circular imports)."""

from __future__ import annotations

import numpy as np

from nba_fit.models.constants import ROLE_FIT_SIGMOID_SCALE


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def role_fit_sigmoid(x: float) -> float:
    return sigmoid(x * ROLE_FIT_SIGMOID_SCALE)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.5
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.5
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
