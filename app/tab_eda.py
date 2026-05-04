"""EDA tab — explore bronze indicators by country and year."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.helpers import list_indicators, list_iso3_countries, load_bronze_panel


def render_eda() -> None:
    st.subheader("Exploratory Data Analysis")

    panel = load_bronze_panel()
    indicators = list_indicators(panel)
    countries = list_iso3_countries(panel)

    col1, col2 = st.columns([1, 2])
    with col1:
        indicator = st.selectbox("Indicator", indicators)
    with col2:
        defaults = [c for c in ("BRA", "USA", "IND", "CHN", "NGA") if c in countries][:5]
        selected = st.multiselect("Countries (ISO3)", countries, default=defaults)

    sub = panel[
        (panel["indicator_code"] == indicator)
        & (panel["country_iso3"].isin(selected if selected else countries))
    ].copy()

    if sub.empty:
        st.info("No rows for this selection.")
        return

    sub = sub.sort_values(["country_iso3", "year"])

    st.markdown(f"### {indicator} — time series")
    fig = px.line(
        sub,
        x="year",
        y="value",
        color="country_iso3",
        markers=True,
        title=f"{indicator} over time",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Summary stats")
    stats = (
        sub.groupby("country_iso3")["value"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(3)
        .sort_values("mean", ascending=False)
    )
    st.dataframe(stats, use_container_width=True)

    st.markdown("### Coverage matrix (rows × indicators)")
    coverage = (
        panel.groupby("indicator_code")
        .agg(
            countries=("country_iso3", "nunique"),
            rows=("value", "size"),
            year_min=("year", "min"),
            year_max=("year", "max"),
        )
        .sort_values("rows", ascending=False)
    )
    st.dataframe(coverage, use_container_width=True)

    st.markdown("### Raw rows (filtered)")
    st.dataframe(sub.head(500), use_container_width=True, hide_index=True)
