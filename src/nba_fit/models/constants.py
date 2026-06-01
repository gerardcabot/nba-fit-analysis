"""Option B role-embedding and archetype knobs (no magic numbers without rationale)."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Role embedding (PCA / truncated SVD on scaled player feature matrix)
# ---------------------------------------------------------------------------

# Embedding dimension: enough to capture play-style variance without overfitting
# small public samples; audit with explained-variance elbow on held-out seasons.
ROLE_EMBEDDING_N_COMPONENTS: int = 12

# ``pca`` uses full SVD; ``truncated_svd`` is faster when n_features >> n_components.
ROLE_EMBEDDING_METHOD: str = "truncated_svd"

# Minimum players before fitting season embeddings (below → skip / synthetic only).
ROLE_EMBEDDING_MIN_PLAYERS: int = 30

# ---------------------------------------------------------------------------
# Archetype clustering (GMM default; HDBSCAN for variable cluster count)
# ---------------------------------------------------------------------------

# GMM / mixture components — NOT final without elbow + manual basketball audit.
# Re-fit when adding tracking/lineup features; clusters must map to real roles.
ARCHETYPE_N_COMPONENTS: int = 8

# ``gmm`` or ``hdbscan`` (sklearn.cluster.HDBSCAN).
ARCHETYPE_CLUSTERER: str = "gmm"

# HDBSCAN: minimum cluster size ~ rotation-unit (8–10 NBA players).
HDBSCAN_MIN_CLUSTER_SIZE: int = 8

# HDBSCAN noise label (-1) mapped to this bucket name for team-need accounting.
ARCHETYPE_NOISE_LABEL: str = "unclassified_role"

# Random seed for reproducible GMM / PCA fits.
MODEL_RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Team need vector (archetype gaps + lineup weakness proxies)
# ---------------------------------------------------------------------------

# Weight archetype-gap block vs scaled team weakness proxies in the need vector.
TEAM_NEED_ARCHETYPE_BLOCK_WEIGHT: float = 0.55
TEAM_NEED_WEAKNESS_BLOCK_WEIGHT: float = 0.45

# League archetype share below this is treated as zero when computing roster gaps.
ARCHETYPE_LEAGUE_SHARE_FLOOR: float = 0.02

# ---------------------------------------------------------------------------
# Scoring integration
# ---------------------------------------------------------------------------

# Option B submetric weight; existing Option A weights scaled by (1 - B - C).
ROLE_FIT_WEIGHT: float = 0.10

# Option C lineup-impact submetric weight (``lineup_impact_fit`` from projected NR delta).
LINEUP_IMPACT_WEIGHT: float = 0.12

# Sigmoid scale for team_need_fit (matches submetrics._SIGMOID_SCALE family).
ROLE_FIT_SIGMOID_SCALE: float = 0.75

# Blend archetype-gap vs embedding-direction signals in team_need_fit (sum to 1.0).
ROLE_FIT_ARCHETYPE_BLEND_WEIGHT: float = 0.60
ROLE_FIT_EMBEDDING_BLEND_WEIGHT: float = 0.40

# Neutral submetric when Option B artifacts are unavailable.
ROLE_FIT_NEUTRAL_SCORE: float = 0.5

# Neutral lineup-impact submetric when Option C artifacts are unavailable.
LINEUP_IMPACT_NEUTRAL_SCORE: float = 0.5

# ---------------------------------------------------------------------------
# RAPM ridge regression (lineup stint matrix)
# ---------------------------------------------------------------------------

# Ridge L2 penalty on player coefficients; typical public RAPM uses 500–5000+.
RAPM_RIDGE_ALPHA: float = 2500.0

# Map projected net rating delta (pts/100) to 0–1 via sigmoid(delta * scale).
LINEUP_IMPACT_SIGMOID_SCALE: float = 0.35

# ---------------------------------------------------------------------------
# Lineup net-rating model (embedding + team context)
# ---------------------------------------------------------------------------

# Default regularizer: ``ridge`` or ``elasticnet``.
LINEUP_MODEL_PENALTY: str = "ridge"
LINEUP_MODEL_RIDGE_ALPHA: float = 1.0
LINEUP_MODEL_ELASTICNET_ALPHA: float = 0.1
LINEUP_MODEL_ELASTICNET_L1_RATIO: float = 0.5

# Vector norm below this treated as zero for cosine / gap alignment.
VECTOR_NORM_EPSILON: float = 1e-9

# ---------------------------------------------------------------------------
# Heuristic archetype centroid thresholds (z-scored feature space)
# ---------------------------------------------------------------------------
# High-usage creator: elite usage + playmaking vs league.
ARCHETYPE_THRESH_HIGH_USG: float = 1.0
ARCHETYPE_THRESH_HIGH_AST: float = 0.5
# Advantage creator: above-average usage + assist rate.
ARCHETYPE_THRESH_MED_USG: float = 0.5
ARCHETYPE_THRESH_MED_AST: float = 0.3
# Movement shooter: high 3PA rate, low usage.
ARCHETYPE_THRESH_SHOOTER_FG3: float = 0.8
ARCHETYPE_THRESH_LOW_USG: float = 0.3
ARCHETYPE_THRESH_SPACER_FG3: float = 0.5
ARCHETYPE_THRESH_VERY_LOW_USG: float = -0.2
# Connector wing: passing without primary usage.
ARCHETYPE_THRESH_CONNECTOR_AST: float = 0.4
ARCHETYPE_THRESH_CONNECTOR_USG_CAP: float = 0.5
# Bigs / defense: rim, blocks, boards.
ARCHETYPE_THRESH_RIM_FREQ: float = 0.6
ARCHETYPE_THRESH_BLK: float = 0.8
ARCHETYPE_THRESH_REB: float = 0.6
ARCHETYPE_THRESH_STRETCH_FG3: float = 0.4
ARCHETYPE_THRESH_STRETCH_REB: float = 0.2
ARCHETYPE_THRESH_STL: float = 0.5
ARCHETYPE_THRESH_DEF_GUARD_USG: float = 0.2

# 2D visualization: ``umap`` when umap-learn installed, else PCA projection.
ARCHETYPE_MAP_METHOD: str = "pca"
ARCHETYPE_MAP_N_COMPONENTS: int = 2

# Nearest-neighbor player comps in embedding space (sklearn NearestNeighbors).
NEAREST_NEIGHBOR_K: int = 5

# ---------------------------------------------------------------------------
# Option C — lineup impact projection (product / interim RAPM proxy)
# ---------------------------------------------------------------------------

# Default number of five-man units returned by lineup-sim and fit cards.
LINEUP_SIM_TOP_N: int = 5

# Ignore observed lineup rows below this minute threshold (noisy small samples).
LINEUP_SIM_MIN_MINUTES: float = 25.0

# Partial substitution blend: candidate does not fully replace replaced impact.
LINEUP_REPLACEMENT_BLEND: float = 0.40

# Synthetic roster lineups when interim lineup_units is unavailable.
LINEUP_SYNTHETIC_UNITS_PER_TEAM: int = 8

# ---------------------------------------------------------------------------
# Movement backtest — post-move outcome labeling (Option D validation)
# ---------------------------------------------------------------------------

# Minutes earned proxy: pre_min × (base + fit_scale × raw_fit_score).
POST_MOVE_MINUTES_BASE: float = 0.85
POST_MOVE_MINUTES_FIT_SCALE: float = 0.3
POST_MOVE_MINUTES_CAP: float = 3500.0
POST_MOVE_MINUTES_NORM_DIVISOR: float = 2000.0

# Weighted per-36 proxy delta (PTS, USG, TS, AST) vs pre-move profile.
POST_MOVE_METRIC_WEIGHTS: dict[str, float] = {
    "pts": 0.35,
    "usg": 0.25,
    "ts": 0.25,
    "ast": 0.15,
}

# Composite outcome: minutes_norm × w_min + sigmoid(metric_delta) × w_metric.
POST_MOVE_OUTCOME_MINUTES_WEIGHT: float = 0.6
POST_MOVE_OUTCOME_METRIC_WEIGHT: float = 0.4
POST_MOVE_OUTCOME_METRIC_SIGMOID: float = 5.0

# Default scale when synthetic post-move rates are generated.
POST_MOVE_SYNTHETIC_RATE_MEANS: dict[str, float] = {
    "pts": 0.45,
    "usg": 0.40,
    "ts": 0.52,
    "ast": 0.35,
}
POST_MOVE_SYNTHETIC_RATE_STD: float = 0.07

assert abs(sum(POST_MOVE_METRIC_WEIGHTS.values()) - 1.0) < 1e-9

# Ensemble weight learning (Option D calibration).
WEIGHT_LEARNING_DEFAULT_L2: float = 0.01
