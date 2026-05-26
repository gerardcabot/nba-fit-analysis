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

# Option B submetric weight; existing Option A weights scaled by (1 - this).
ROLE_FIT_WEIGHT: float = 0.10

# Sigmoid scale for team_need_fit (matches submetrics._SIGMOID_SCALE family).
ROLE_FIT_SIGMOID_SCALE: float = 0.75

# 2D visualization: ``umap`` when umap-learn installed, else PCA projection.
ARCHETYPE_MAP_METHOD: str = "pca"
ARCHETYPE_MAP_N_COMPONENTS: int = 2

# Nearest-neighbor player comps in embedding space (sklearn NearestNeighbors).
NEAREST_NEIGHBOR_K: int = 5
