"""Predictions tab — vaccination milestone probabilities."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.helpers import (
    list_gold_targets,
    list_iso3_countries,
    load_best_model,
    load_bronze_panel,
    load_gold,
)

ID_COLS = ("country_iso3", "year", "label")
EXPERIMENT_BY_TARGET = {
    "vaccination_milestone": "vaccination_milestone",
    "outbreak_risk": "outbreak_risk",
    "resurgence": "resurgence",
}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS]


def render_predictions() -> None:
    st.subheader("Predictions")

    targets = list_gold_targets()
    if not targets:
        st.warning("No gold ABTs found. Run `python -m etl.gold`.")
        return

    target = st.selectbox("Target", targets, index=0)
    experiment = EXPERIMENT_BY_TARGET.get(target, target)

    try:
        model, run_id = load_best_model(experiment)
    except Exception as exc:
        st.error(f"Could not load best model for `{experiment}`: {exc}")
        return

    if model is None:
        st.info(f"No MLflow runs for experiment `{experiment}`. Train it via `python -m ml.{target.split('_')[0]}_model`.")
        return

    st.caption(f"Loaded best run `{run_id}` from experiment `{experiment}`.")

    abt = load_gold(target)
    panel = load_bronze_panel()
    countries = list_iso3_countries(panel)
    countries = [c for c in countries if c in set(abt["country_iso3"].unique())]

    col1, col2, col3 = st.columns(3)
    with col1:
        country = st.selectbox("Country (ISO3)", countries, index=countries.index("BRA") if "BRA" in countries else 0)
    with col2:
        if "vaccine" in abt.columns:
            vaccines = sorted(abt["vaccine"].dropna().unique().tolist())
            vaccine = st.selectbox("Vaccine", vaccines)
        else:
            vaccine = None
    with col3:
        if "horizon" in abt.columns:
            horizons = sorted(abt["horizon"].dropna().unique().tolist())
            horizon = st.selectbox("Horizon (years)", horizons, index=0)
        else:
            horizon = None

    subset = abt[abt["country_iso3"] == country].copy()
    if vaccine is not None:
        subset = subset[subset["vaccine"] == vaccine]
    if horizon is not None:
        subset = subset[subset["horizon"] == horizon]
    if subset.empty:
        st.warning("No rows match this selection.")
        return

    feature_cols = _feature_cols(subset)
    X = subset[feature_cols]
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception as exc:
        st.error(f"Model inference failed: {exc}")
        return

    out = subset[["country_iso3", "year", "label"]].copy()
    if vaccine is not None:
        out["vaccine"] = vaccine
    if horizon is not None:
        out["horizon"] = horizon
    out["proba"] = proba
    out["pred"] = (proba >= 0.5).astype("int8")
    out = out.sort_values("year").reset_index(drop=True)

    st.markdown(f"### {country} — proba over time")
    fig = px.line(
        out,
        x="year",
        y="proba",
        markers=True,
        title=f"P(milestone hit | {country}, {vaccine}, h={horizon})",
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="threshold 0.5")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Predictions table")
    st.dataframe(out, use_container_width=True)

    st.download_button(
        "Download CSV",
        out.to_csv(index=False).encode(),
        file_name=f"{target}_{country}_{vaccine}_h{horizon}.csv",
        mime="text/csv",
    )
