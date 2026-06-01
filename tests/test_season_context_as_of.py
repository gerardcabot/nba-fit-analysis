"""SeasonFitContext temporal as_of tests."""

from datetime import date

import pandas as pd

from nba_fit.features.season_context import SeasonFitContext, filter_gamelogs_as_of


def test_as_of_sets_cutoff_date():
    ctx = SeasonFitContext.from_synthetic("2025-26", n_players=5)
    move = date(2026, 2, 6)
    scoped = ctx.as_of(move)
    assert scoped.as_of_date == move
    assert scoped.season == ctx.season
    assert scoped.as_of(None).as_of_date is None


def test_filter_gamelogs_as_of():
    logs = pd.DataFrame(
        {
            "GAME_DATE": ["2026-01-01", "2026-02-10", "2026-03-01"],
            "PTS": [20, 25, 30],
        }
    )
    filtered = filter_gamelogs_as_of(logs, date(2026, 2, 6))
    assert len(filtered) == 1
    assert filtered.iloc[0]["PTS"] == 20
