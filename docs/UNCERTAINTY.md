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

## Related metadata

Option C impact training sets `ImpactFitContext.degenerate_rapm` when all `net_rapm` coefficients are identical (`std == 0`). Treat uncertainty bands as unreliable in that regime and prefer lineup_units or more possession coverage.
