"""SOTA exit checks for Option C (RAPM / lineup impact)."""

from __future__ import annotations

from typing import Any

from nba_fit.evaluation.rapm_benchmark import RapmBenchmarkResult, benchmark_rapm_vs_estimated_net
from nba_fit.models.rapm import RapmArtifacts, load_rapm


class SotaValidationError(Exception):
    """Raised when a SOTA gate fails."""


def assert_non_degenerate_rapm(rapm: RapmArtifacts) -> None:
    """Fail when all ``net_rapm`` coefficients are identical (std == 0)."""
    net_std = float(rapm.metadata.get("net_rapm_std", 0.0))
    if rapm.degenerate or (len(rapm.net_rapm) > 0 and net_std == 0.0):
        raise SotaValidationError(
            f"Degenerate RAPM for season {rapm.season}: net_rapm std == 0 "
            f"(source={rapm.rapm_source}, n_players={len(rapm.player_ids)})"
        )


def validate_option_c_rapm(
    season: str,
    *,
    require_benchmark: bool = True,
    min_benchmark_players: int = 100,
) -> dict[str, Any]:
    """
    Run SOTA gates: non-degenerate RAPM and optional E_NET_RATING Spearman benchmark.

    Returns a metrics dict suitable for ``reports/validation/sota/option_c_metrics.json``.
    """
    rapm = load_rapm(season)
    assert_non_degenerate_rapm(rapm)

    benchmark: RapmBenchmarkResult = benchmark_rapm_vs_estimated_net(
        season,
        rapm=rapm,
        min_players=min_benchmark_players,
    )

    if require_benchmark and not benchmark.passed:
        detail = benchmark.note or f"spearman={benchmark.spearman_rho}"
        raise SotaValidationError(
            f"RAPM benchmark failed for {season}: {detail} "
            f"(n_overlap={benchmark.n_overlap}, degenerate={benchmark.degenerate_rapm})"
        )

    return {
        "season": season,
        "rapm": {
            "n_players": int(len(rapm.player_ids)),
            "net_rapm_std": float(rapm.metadata.get("net_rapm_std", 0.0)),
            "degenerate_rapm": bool(rapm.degenerate),
            "rapm_source": rapm.rapm_source,
            "ridge_alpha": rapm.ridge_alpha,
            "recency_half_life_games": rapm.recency_half_life_games,
            **{
                k: v
                for k, v in rapm.metadata.items()
                if k.startswith("rapm_tune") or k.startswith("bibliography")
            },
        },
        "benchmark": benchmark.to_dict(),
        "sota_gates": {
            "non_degenerate_rapm": True,
            "benchmark_min_players": min_benchmark_players,
            "benchmark_passed": benchmark.passed,
        },
    }
