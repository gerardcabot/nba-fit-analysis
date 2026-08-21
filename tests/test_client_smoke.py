"""Client smoke tests (mocked offline; optional live network)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from nba_fit.data.client import NBAClient, patch_nba_stats_headers
from nba_fit.data.registry import ProbeRegistry


def test_probe_registry_loads():
    settings = __import__("nba_fit.config.settings", fromlist=["get_settings"]).get_settings()
    if not settings.probe_results_path.exists():
        pytest.skip("probe_all_results.json not present")
    reg = ProbeRegistry.load()
    assert reg.endpoints
    assert sum(reg.summary().values()) == len(reg.endpoints)


@patch("nba_fit.data.client._get_endpoint_class")
def test_fetch_uses_cache_after_first_write(mock_cls, tmp_path):
    from nba_fit.config.settings import Settings

    settings = Settings(
        root=tmp_path,
        data_root=None,
        data_raw=tmp_path / "data" / "raw",
        data_interim=tmp_path / "data" / "interim",
        data_features=tmp_path / "data" / "features",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        probe_results_path=tmp_path / "probe_all_results.json",
        seasons=("2025-26",),
        cache_ttl_hours=24,
        default_season="2025-26",
    )

    fake_ep = MagicMock()
    df = pd.DataFrame({"PLAYER_ID": [1], "PTS": [10.0]})
    fake_ep.get_data_frames.return_value = [df]
    fake_ep.nba_response = None
    mock_cls.return_value = MagicMock(return_value=fake_ep)

    client = NBAClient(settings=settings)
    with patch("nba_fit.data.client.patch_nba_stats_headers"):
        r1 = client.fetch("leaguedashplayerstats", season="2025-26", use_cache=False)
        assert not r1.from_cache
        r2 = client.fetch("leaguedashplayerstats", season="2025-26", use_cache=True)
        assert r2.from_cache
    assert mock_cls.return_value.call_count == 1


@pytest.mark.network
def test_live_fetch_sample_optional():
    """Run with: pytest -m network tests/test_client_smoke.py::test_live_fetch_sample_optional"""
    patch_nba_stats_headers()
    client = NBAClient()
    result = client.fetch("commonallplayers", season="2025-26", use_cache=True)
    assert sum(len(df) for df in result.frames.values()) > 0
