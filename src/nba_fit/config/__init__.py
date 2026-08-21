"""Configuration: paths, seasons, thresholds, endpoint registry."""

from nba_fit.config.endpoints import ENDPOINT_REGISTRY, EndpointSpec
from nba_fit.config.settings import (
    ENV_NBA_FIT_DATA_ROOT,
    Settings,
    ensure_data_root_layout,
    get_settings,
)

__all__ = [
    "ENDPOINT_REGISTRY",
    "EndpointSpec",
    "ENV_NBA_FIT_DATA_ROOT",
    "Settings",
    "ensure_data_root_layout",
    "get_settings",
]
