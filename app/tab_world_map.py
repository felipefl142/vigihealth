"""World map tab — choropleth of indicator value by country and year."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.helpers import (
    list_gold_targets,
    list_indicators,
    load_best_model,
    load_bronze_panel,
    load_gold,
)
from constants import REGIONAL_AGGREGATE_CODES

EXPERIMENT_BY_TARGET = {
    "vaccination_milestone": "vaccination_milestone",
    "outbreak_risk": "outbreak_risk",
    "resurgence": "resurgence",
}

ID_COLS = ("country_iso3", "year", "label")


def _country_panel(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[
        (~panel["country_iso3"].isin(REGIONAL_AGGREGATE_CODES))
        & (panel["country_iso3"].str.len() == 3)
    ]


def _render_indicator_map(panel: pd.DataFrame) -> None:
    indicators = list_indicators(panel)
    indicator = st.selectbox("Indicator", indicators, key="map_indicator")
    sub = _country_panel(panel[panel["indicator_code"] == indicator])
    if sub.empty:
        st.info("No data for this indicator.")
        return

    years = sorted(sub["year"].dropna().unique().tolist())
    year = st.select_slider("Year", years, value=years[-1], key="map_indicator_year")

    snapshot = sub[sub["year"] == year]
    if snapshot.empty:
        st.info(f"No data for {indicator} in {year}.")
        return

    fig = px.choropleth(
        snapshot,
        locations="country_iso3",
        color="value",
        hover_name="country_iso3",
        color_continuous_scale="Viridis",
        title=f"{indicator} — {year}",
    )
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top / bottom 10")
    ranked = snapshot[["country_iso3", "value"]].dropna().sort_values("value", ascending=False)
    col1, col2 = st.columns(2)
    col1.write("Top 10")
    col1.dataframe(ranked.head(10), use_container_width=True, hide_index=True)
    col2.write("Bottom 10")
    col2.dataframe(ranked.tail(10).iloc[::-1], use_container_width=True, hide_index=True)


def _render_prediction_map() -> None:
    targets = list_gold_targets()
    if not targets:
        st.info("No gold ABTs.")
        return
    target = st.selectbox("Target", targets, key="map_target")
    experiment = EXPERIMENT_BY_TARGET.get(target, target)
    try:
        model, run_id = load_best_model(experiment)
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return
    if model is None:
        st.info(f"Train `{experiment}` first.")
        return

    abt = load_gold(target)
    abt = abt[
        (~abt["country_iso3"].isin(REGIONAL_AGGREGATE_CODES))
        & (abt["country_iso3"].str.len() == 3)
    ]

    col1, col2, col3 = st.columns(3)
    with col1:
        years = sorted(abt["year"].dropna().unique().tolist())
        year = st.select_slider("Year", years, value=years[-1], key="map_pred_year")
    with col2:
        if "vaccine" in abt.columns:
            vaccine = st.selectbox("Vaccine", sorted(abt["vaccine"].dropna().unique().tolist()), key="map_vaccine")
        else:
            vaccine = None
    with col3:
        if "horizon" in abt.columns:
            horizon = st.selectbox("Horizon", sorted(abt["horizon"].dropna().unique().tolist()), key="map_horizon")
        else:
            horizon = None

    sub = abt[abt["year"] == year]
    if vaccine is not None:
        sub = sub[sub["vaccine"] == vaccine]
    if horizon is not None:
        sub = sub[sub["horizon"] == horizon]
    if sub.empty:
        st.info("No rows for this selection.")
        return

    feature_cols = [c for c in sub.columns if c not in ID_COLS]
    proba = model.predict_proba(sub[feature_cols])[:, 1]
    plot_df = sub[["country_iso3"]].copy()
    plot_df["proba"] = proba

    fig = px.choropleth(
        plot_df,
        locations="country_iso3",
        color="proba",
        color_continuous_scale="RdYlGn",
        range_color=(0, 1),
        hover_name="country_iso3",
        title=f"P({target}=1) — {year}, {vaccine}, h={horizon}",
    )
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Best run: `{run_id}`")


def render_world_map() -> None:
    st.subheader("World Map")
    panel = load_bronze_panel()

    mode = st.radio("Layer", ["Indicator value", "Model prediction"], horizontal=True)
    if mode == "Indicator value":
        _render_indicator_map(panel)
    else:
        _render_prediction_map()
