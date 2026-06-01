"""Config and endpoint registry tests."""

from nba_fit.config.endpoints import ENDPOINT_REGISTRY, ESSENTIAL_ENDPOINTS
from nba_fit.config.settings import (
    INGEST_IMPACT_MAX_GAMES_DEV,
    INGEST_TIER_IMPACT,
    INGEST_TIER_MVP,
    INGEST_TIER_ROLE,
    INGEST_TIER_TACTICAL,
    MIN_SEASON_MINUTES_STABLE_RATES,
    NBA_TEAM_IDS,
    OPTION_A_MVP_ENDPOINTS,
    OPTION_B_ROLE_ENDPOINTS,
    OPTION_C_IMPACT_ENDPOINTS,
    OPTION_D_TACTICAL_ENDPOINTS,
    POSSESSIONS_PER_TEAM_PER_GAME_REGULATION,
    REGULAR_SEASON_GAME_COUNT,
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


def test_option_b_role_ingest_bundle():
    assert INGEST_TIER_ROLE == "role"
    assert "teamplayeronoffsummary" in OPTION_B_ROLE_ENDPOINTS
    assert "leaguedashlineups" in OPTION_B_ROLE_ENDPOINTS
    assert len(OPTION_B_ROLE_ENDPOINTS) == 4
    assert len(NBA_TEAM_IDS) == 30


def test_option_c_impact_ingest_bundle():
    assert INGEST_TIER_IMPACT == "impact"
    assert "playbyplayv3" in OPTION_C_IMPACT_ENDPOINTS
    assert "gamerotation" in OPTION_C_IMPACT_ENDPOINTS
    assert "leaguegamefinder" in OPTION_C_IMPACT_ENDPOINTS
    assert len(OPTION_C_IMPACT_ENDPOINTS) == 3
    assert INGEST_IMPACT_MAX_GAMES_DEV > 0
    assert INGEST_IMPACT_MAX_GAMES_DEV < REGULAR_SEASON_GAME_COUNT


def test_option_d_tactical_ingest_bundle():
    assert INGEST_TIER_TACTICAL == "tactical"
    assert "leaguehustlestatsplayer" in OPTION_D_TACTICAL_ENDPOINTS
    assert "leaguedashptdefend" in OPTION_D_TACTICAL_ENDPOINTS
    assert "gravityleaders" in OPTION_D_TACTICAL_ENDPOINTS
    assert len(OPTION_D_TACTICAL_ENDPOINTS) == 4


def test_canonical_ids():
    assert canonical_player_id("1629029") == 1629029
    assert canonical_team_id(1610612747) == 1610612747
    assert canonical_game_id("22501198") == "0022501198"
