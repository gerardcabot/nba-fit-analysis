"""Config and endpoint registry tests."""

from nba_fit.config.endpoints import ENDPOINT_REGISTRY, ESSENTIAL_ENDPOINTS
from nba_fit.config.settings import (
    INGEST_TIER_MVP,
    MIN_SEASON_MINUTES_STABLE_RATES,
    OPTION_A_MVP_ENDPOINTS,
    POSSESSIONS_PER_TEAM_PER_GAME_REGULATION,
    get_settings,
)
from nba_fit.normalize.ids import canonical_game_id, canonical_player_id, canonical_team_id


def test_settings_paths_exist_under_root():
    settings = get_settings()
    assert settings.root.is_dir()
    assert settings.data_raw.name == "raw"
    assert "season=" in str(settings.raw_parquet_dir("nba_api", "leaguedashplayerstats", "2024-25"))


def test_domain_constants_documented_values():
    assert POSSESSIONS_PER_TEAM_PER_GAME_REGULATION == 100.0
    assert MIN_SEASON_MINUTES_STABLE_RATES == 500


def test_essential_registry_includes_core_endpoints():
    assert "leaguedashplayerstats" in ESSENTIAL_ENDPOINTS
    assert "playbyplayv3" in ESSENTIAL_ENDPOINTS
    assert ENDPOINT_REGISTRY["synergyplaytypes"].reliability == "unreliable"


def test_option_a_mvp_ingest_bundle():
    assert INGEST_TIER_MVP == "mvp"
    assert "leaguedashplayerstats" in OPTION_A_MVP_ENDPOINTS
    assert "teamestimatedmetrics" in OPTION_A_MVP_ENDPOINTS
    assert len(OPTION_A_MVP_ENDPOINTS) == 7


def test_canonical_ids():
    assert canonical_player_id("1629029") == 1629029
    assert canonical_team_id(1610612747) == 1610612747
    assert canonical_game_id("22501198") == "0022501198"
