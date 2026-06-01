"""Configuration: paths, seasons, thresholds, endpoint registry."""

from nba_fit.config.endpoints import ENDPOINT_REGISTRY, EndpointSpec
from nba_fit.config.settings import Settings, get_settings

__all__ = [
    "ENDPOINT_REGISTRY",
    "EndpointSpec",
    "Settings",
    "get_settings",
]
