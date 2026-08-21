# Uncertainty bands

NBA Fit exposes uncertainty through `nba_fit.scoring.uncertainty` for fit cards, movement backtests, and validation.

## Model disagreement bootstrap

`bootstrap_disagreement_ci` treats each ensemble component (profile, role, need, impact, replacement) as an independent model draw. It resamples component indices with replacement and builds a percentile CI on the ensemble mean.

Use when you have a **vector of submodel scores** for the same player–team pair and want a quick spread-based band.

## Block bootstrap

`block_bootstrap_ci(values, block_ids)` resamples **whole blocks** (games, stints, players) instead of individual rows. This preserves within-block correlation—important for:

- RAPM stint residuals (blocks = `game_id` or stint id)
- Movement backtest rows (blocks = `player_id`)
- Season-level aggregates where observations are not i.i.d.

Returns `(point, ci_low, ci_high)` for the sample mean of `values`.

Example:

```python
from nba_fit.scoring.uncertainty import block_bootstrap_ci
import numpy as np

residuals = np.array([...])       # per-stint or per-pair errors
game_ids = np.array([...])        # same length as residuals
point, lo, hi = block_bootstrap_ci(residuals, game_ids, n_bootstrap=500)
```

## Sample-size widening

`sample_size_penalty` and `widen_ci_for_sample` inflate CI width when rotation minutes are low (aligned with RAPM `low_sample_flag` semantics). `fit_uncertainty` combines disagreement bootstrap, sample penalty, and an effective-*n* width scale.

## Empirical coverage (validation template)

Run after movement backtest or holdout scoring to check whether uncertainty bands achieve nominal coverage.

### Procedure

1. **Collect rows** — movement backtest or holdout eval frame with `post_move_outcome`, `calibrated_fit_percentile`, and `uncertainty_ci_low` / `uncertainty_ci_high` (or fit-card bands).
2. **Block bootstrap** — resample by `player_id` when reporting season-level coverage (see `block_bootstrap_ci`).
3. **Coverage metric** — fraction of rows where realized outcome (0–100 scale) falls inside `[ci_low, ci_high]`.
4. **Target** — 80% nominal (`alpha=0.10`) for disagreement bootstrap; document under-coverage when RAPM is degenerate.

### Report template (`reports/validation/sota/coverage.json`)

```json
{
  "generated_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "eval_season": "2024-25",
  "nominal_coverage": 0.80,
  "empirical_coverage": null,
  "n_eval_rows": 0,
  "block_bootstrap": {
    "block_id": "player_id",
    "n_bootstrap": 500,
    "alpha": 0.10,
    "coverage_ci_low": null,
    "coverage_ci_high": null
  },
  "conformal_hook": {
    "status": "planned",
    "note": "Apply split conformal on movement residuals once real labels >= 50"
  },
  "degenerate_rapm_flag": false
}
```

### Hooks in code

| Function | Use |
|----------|-----|
| `bootstrap_disagreement_ci` | Per-pair band from ensemble components |
| `block_bootstrap_ci` | Season-level coverage CI with player blocks |
| `fit_uncertainty` | Fit cards + movement backtest row bands |
| `evaluate_holdout_calibration` | Holdout isotonic metrics (not coverage) |

Fill `empirical_coverage` from `scripts/run_sota_validation.py` once holdout eval rows include uncertainty columns.

## Related metadata

Option C impact training sets `ImpactFitContext.degenerate_rapm` when all `net_rapm` coefficients are identical (`std == 0`). Treat uncertainty bands as unreliable in that regime and prefer lineup_units or more possession coverage.
