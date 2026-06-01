"""
NBA Fit Streamlit dashboard (Phase 6).

Run: streamlit run src/nba_fit/app/dashboard.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from nba_fit.app.data_health import (
    build_data_health_figure,
    essential_health_table,
    probe_summary,
)
from nba_fit.app.sota_validation import (
    calibration_summary,
    figure_path,
    learned_weights_frame,
    load_degradation_warnings,
    load_metrics,
    movement_source_label,
    sota_dir,
)
from nba_fit.config.endpoints import ENDPOINT_REGISTRY
from nba_fit.config.settings import get_settings
from nba_fit.data.registry import ProbeRegistry
from nba_fit.features.season_context import DEMO_PLAYER_ID, DEMO_TEAM_ID, SeasonFitContext
from nba_fit.models.constants import (
    ARCHETYPE_MAP_METHOD,
    ARCHETYPE_MAP_N_COMPONENTS,
    MODEL_RANDOM_STATE,
)
from nba_fit.models.role_context import RoleFitContext
from nba_fit.scoring.archetype_board import archetype_board_for_team
from nba_fit.scoring.constants import SUBMETRIC_NAMES, SUBMETRIC_WEIGHTS
from nba_fit.scoring.fit_card import build_fit_card, fit_card_to_json
from nba_fit.scoring.lineup_sim import run_lineup_sim
from nba_fit.scoring.ranker import FitRanker

VIEWS: tuple[str, ...] = (
    "Player Destination Explorer",
    "Team Target Board",
    "Fit Card",
    "Lineup Simulator",
    "Archetype Map",
    "Backtest Report",
    "Data Health",
)


def _project_archetype_2d(embeddings: np.ndarray) -> np.ndarray:
    if ARCHETYPE_MAP_METHOD == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_components=ARCHETYPE_MAP_N_COMPONENTS,
                random_state=MODEL_RANDOM_STATE,
            )
            return reducer.fit_transform(embeddings)
        except ImportError:
            pass
    pca = PCA(n_components=ARCHETYPE_MAP_N_COMPONENTS, random_state=MODEL_RANDOM_STATE)
    return pca.fit_transform(embeddings)


@st.cache_data(show_spinner="Loading fit rankings…")
def _load_ranker(season: str, synthetic: bool) -> FitRanker:
    return FitRanker.from_season(
        season,
        prefer_interim=not synthetic,
        prefer_api=not synthetic,
        synthetic=synthetic,
    )


@st.cache_data(show_spinner="Loading role model…")
def _load_role_context(season: str, synthetic: bool) -> RoleFitContext:
    if synthetic:
        ctx = SeasonFitContext.from_synthetic(season)
        return RoleFitContext.from_synthetic(ctx)
    return RoleFitContext.from_season(season, persist=False)


@st.cache_data(show_spinner="Loading probe registry…")
def _load_registry() -> ProbeRegistry:
    return ProbeRegistry.load(settings=get_settings())


def _sidebar() -> tuple[str, str, bool, int]:
    st.sidebar.title("NBA Fit")
    view = st.sidebar.selectbox("View", VIEWS)
    settings = get_settings()
    season = st.sidebar.text_input("Season", value=settings.default_season)
    synthetic = st.sidebar.checkbox(
        "Synthetic demo data",
        value=False,
        help="Offline mode without cached ingest or live API",
    )
    top_n = st.sidebar.slider("Top N rows", min_value=5, max_value=30, value=10)
    st.sidebar.caption(
        "Install: `pip install -e \".[dashboard]\"` · "
        "CLI: `python -m nba_fit health`"
    )
    return view, season, synthetic, top_n


def _view_player_destinations(season: str, synthetic: bool, top_n: int) -> None:
    st.header("Player Destination Explorer")
    player_id = st.number_input(
        "Player ID",
        min_value=1,
        value=DEMO_PLAYER_ID,
        step=1,
    )
    ranker = _load_ranker(season, synthetic)
    st.caption(f"Data source: **{ranker.context.source}**")
    rankings = ranker.rank_destinations_for_player(int(player_id), top_n=top_n)
    if rankings.empty:
        st.warning(f"No rankings for player {player_id} in {season}.")
        return
    st.dataframe(rankings, use_container_width=True, hide_index=True)
    st.bar_chart(
        rankings.set_index("team")["overall_fit_percentile"],
        horizontal=True,
    )


def _view_team_targets(season: str, synthetic: bool, top_n: int) -> None:
    st.header("Team Target Board")
    team_id = st.number_input(
        "Team ID",
        min_value=1,
        value=DEMO_TEAM_ID,
        step=1,
    )
    ranker = _load_ranker(season, synthetic)
    st.caption(f"Data source: **{ranker.context.source}**")
    rankings = ranker.rank_players_for_team(int(team_id), top_n=top_n)
    if rankings.empty:
        st.warning(f"No rankings for team {team_id} in {season}.")
        return
    st.dataframe(rankings, use_container_width=True, hide_index=True)


def _view_fit_card(season: str, synthetic: bool) -> None:
    st.header("Fit Card")
    col1, col2 = st.columns(2)
    with col1:
        player_id = st.number_input(
            "Player ID",
            min_value=1,
            value=DEMO_PLAYER_ID,
            key="fit_card_player",
        )
    with col2:
        team_id = st.number_input(
            "Team ID",
            min_value=1,
            value=DEMO_TEAM_ID,
            key="fit_card_team",
        )
    ranker = _load_ranker(season, synthetic)
    card = build_fit_card(
        int(player_id),
        int(team_id),
        ranker.table,
        ranker.context,
        lineup_synthetic=synthetic,
    )
    overall = card.get("overall_fit_percentile")
    if overall is not None:
        st.metric("Overall fit percentile", f"{overall:.1f}")
    unc = card.get("fit_uncertainty") or {}
    if unc.get("low_percentile") is not None and unc.get("high_percentile") is not None:
        st.caption(
            f"Uncertainty band: {unc['low_percentile']:.1f} – {unc['high_percentile']:.1f}"
        )
    if card.get("projected_net_rating_delta") is not None:
        st.metric(
            "Projected net rating Δ (pts/100 poss)",
            f"{card['projected_net_rating_delta']:+.2f}",
        )

    degraded = card.get("components_degraded") or card.get("fallbacks") or []
    if degraded:
        st.warning(
            "Neutral fallback used for: "
            + ", ".join(str(d) for d in degraded)
        )

    submetrics = card.get("submetrics") or {}
    if submetrics:
        sub_df = pd.DataFrame(
            {
                "submetric": list(submetrics.keys()),
                "score": list(submetrics.values()),
            }
        )
        st.subheader("Submetrics")
        st.bar_chart(sub_df.set_index("submetric")["score"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Archetype", card.get("archetype") or "—")
    if card.get("soft_role_display"):
        st.caption(f"Soft roles: {card['soft_role_display']}")
    team_need = card.get("team_need_fit")
    if team_need is None:
        team_need = card.get("role_fit")
    c2.metric("Team need fit", f"{team_need:.3f}" if team_need is not None else "—")
    c3.metric("Data source", card.get("data_source", "—"))

    comps = card.get("comps") or []
    if comps:
        st.subheader("Player comps")
        st.dataframe(pd.DataFrame(comps), use_container_width=True, hide_index=True)
    st.caption(card.get("comps_note", ""))

    lineup = card.get("lineup_synergy")
    if lineup and lineup.get("top_lineups"):
        st.subheader("Lineup synergy")
        headline = lineup.get("headline") or {}
        delta = headline.get("projected_net_rating_delta") or lineup.get(
            "projected_net_rating_delta"
        )
        if delta is not None:
            units = headline.get("units", "pts/100 poss")
            st.metric("Lineup projected net rating Δ", f"{delta:+.2f}", units)
        rows = []
        for u in lineup["top_lineups"]:
            rows.append(
                {
                    "lineup": u.get("lineup_label"),
                    "baseline_net": u.get("baseline_net_rating"),
                    "delta": u.get("projected_net_rating_delta"),
                    "minutes": u.get("minutes"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Full JSON"):
        st.code(fit_card_to_json(card), language="json")


def _view_lineup_sim(season: str, synthetic: bool, top_n: int) -> None:
    st.header("Lineup Simulator")
    col1, col2 = st.columns(2)
    with col1:
        player_id = st.number_input(
            "Player ID",
            min_value=1,
            value=DEMO_PLAYER_ID,
            key="lineup_player",
        )
    with col2:
        team_id = st.number_input(
            "Team ID",
            min_value=1,
            value=DEMO_TEAM_ID,
            key="lineup_team",
        )
    result = run_lineup_sim(
        int(player_id),
        int(team_id),
        season,
        top_n=top_n,
        prefer_interim=not synthetic,
        prefer_api=not synthetic,
        synthetic=synthetic,
    )
    if result.projected_net_rating_delta is not None:
        st.metric(
            "Projected net rating Δ",
            f"{result.projected_net_rating_delta:+.2f}",
        )
    table = result.to_table()
    if table.empty:
        st.warning("No projected lineup units for this pair.")
        if result.note:
            st.info(result.note)
        return
    st.dataframe(table, use_container_width=True, hide_index=True)
    if result.note:
        st.caption(result.note)


def _view_archetype_map(season: str, synthetic: bool) -> None:
    st.header("Archetype Map")
    role = _load_role_context(season, synthetic)
    coords = _project_archetype_2d(role.embeddings.embeddings)
    labels = pd.Series(
        role.archetypes.archetype_labels,
        index=role.archetypes.player_ids,
        name="archetype",
    )
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1] if coords.shape[1] > 1 else 0.0,
            "archetype": labels.loc[role.embeddings.player_ids].to_numpy(),
            "player_id": role.embeddings.player_ids,
        }
    )
    df["player"] = df["player_id"].map(
        lambda pid: role.player_names.get(int(pid), str(pid))
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    for archetype, group in df.groupby("archetype"):
        ax.scatter(
            group["x"],
            group["y"],
            label=str(archetype),
            alpha=0.75,
            s=36,
        )
    ax.set_title("Player archetypes (2D role embedding projection)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Archetype target board")
    team_id = st.number_input(
        "Team ID (need board)",
        min_value=1,
        value=DEMO_TEAM_ID,
        key="archetype_team",
    )
    board = archetype_board_for_team(
        int(team_id),
        season,
        top_n=15,
        prefer_interim=not synthetic,
        prefer_api=not synthetic,
        synthetic=synthetic,
    )
    if board.empty:
        st.warning("Archetype board unavailable — run train-roles or enable synthetic.")
    else:
        st.dataframe(board, use_container_width=True, hide_index=True)


def _view_backtest(season: str, synthetic: bool) -> None:
    st.header("Backtest Report")
    metrics = load_metrics()
    sota_path = sota_dir()

    if metrics is None:
        st.warning(
            f"No SOTA validation bundle at `{sota_path}`. "
            "Run the SOTA validation script or copy Option D artifacts into "
            "`reports/validation/sota/`."
        )
    else:
        st.caption(f"SOTA validation: `{sota_path}` · season **{metrics.get('season', season)}**")

        warnings = load_degradation_warnings()
        if warnings:
            for msg in warnings:
                st.warning(str(msg))

        cal = calibration_summary(metrics)
        backtest = metrics.get("backtest") or {}
        st.subheader("Movement backtest")
        m1, m2, m3 = st.columns(3)
        m1.metric("Movements", backtest.get("n_movements", "—"))
        m2.metric("Source", movement_source_label(metrics))
        m3.metric(
            "Calibrated",
            "Yes" if cal.get("fitted") else "No",
        )
        if cal.get("method"):
            st.caption(
                f"Calibration: {cal.get('method')} "
                f"(n_anchor={cal.get('n_anchor', '—')})"
            )

        cal_fig = figure_path("12_calibration_curve")
        if cal_fig is not None:
            st.subheader("Calibration curve")
            st.image(str(cal_fig), use_container_width=True)
        else:
            st.info("Calibration curve figure not found in SOTA bundle.")

        weights_df = learned_weights_frame(metrics)
        if weights_df is not None:
            st.subheader("Ensemble weights")
            st.dataframe(weights_df, use_container_width=True, hide_index=True)
            move_fig = figure_path("13_ensemble_weights")
            if move_fig is not None:
                st.image(str(move_fig), use_container_width=True)

    st.subheader("Live fit index (current season)")
    ranker = _load_ranker(season, synthetic)
    pairs = ranker.table.pairs
    if pairs.empty:
        st.warning("No fit index pairs to summarize.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Player×team pairs", f"{len(pairs):,}")
    c2.metric(
        "Calibrated",
        "Yes" if ranker.table.is_calibrated else "No",
    )
    c3.metric(
        "Median percentile",
        f"{pairs['overall_fit_percentile'].median():.1f}",
    )

    hist_df = pairs[["overall_fit_percentile"]].rename(
        columns={"overall_fit_percentile": "percentile"}
    )
    st.bar_chart(hist_df["percentile"].value_counts(bins=20).sort_index())

    st.subheader("Submetric weights (model spec)")
    weights_df = pd.DataFrame(
        {
            "submetric": list(SUBMETRIC_WEIGHTS),
            "weight": list(SUBMETRIC_WEIGHTS.values()),
        }
    )
    st.dataframe(weights_df, use_container_width=True, hide_index=True)
    st.caption(f"Active submetrics in index: {', '.join(SUBMETRIC_NAMES)}")


def _view_data_health() -> None:
    st.header("Data Health")
    try:
        registry = _load_registry()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python probe_all_nba_endpoints.py", language="bash")
        return

    meta = registry.meta
    summary = probe_summary(registry)
    st.caption(
        f"Probe: **{meta.get('timestamp', 'unknown')}** · "
        f"Season: **{meta.get('season', 'unknown')}** · "
        f"Total endpoints: **{len(registry.endpoints)}**"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("OK", summary.get("OK", 0))
    m2.metric("EMPTY", summary.get("EMPTY", 0))
    m3.metric("FAIL", summary.get("FAIL", 0))

    fig = build_data_health_figure(registry)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Essential registry endpoints")
    table = essential_health_table(registry)
    st.dataframe(table, use_container_width=True, hide_index=True)

    unreliable = [
        name
        for name, spec in ENDPOINT_REGISTRY.items()
        if spec.reliability == "unreliable"
    ]
    if unreliable:
        st.subheader("Flagged unreliable")
        st.write(", ".join(unreliable))


def main() -> None:
    st.set_page_config(
        page_title="NBA Fit Analysis",
        page_icon="🏀",
        layout="wide",
    )
    view, season, synthetic, top_n = _sidebar()

    if view == "Player Destination Explorer":
        _view_player_destinations(season, synthetic, top_n)
    elif view == "Team Target Board":
        _view_team_targets(season, synthetic, top_n)
    elif view == "Fit Card":
        _view_fit_card(season, synthetic)
    elif view == "Lineup Simulator":
        _view_lineup_sim(season, synthetic, top_n)
    elif view == "Archetype Map":
        _view_archetype_map(season, synthetic)
    elif view == "Backtest Report":
        _view_backtest(season, synthetic)
    elif view == "Data Health":
        _view_data_health()


main()
