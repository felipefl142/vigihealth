"""Model comparison tab — MLflow runs side-by-side."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.helpers import list_mlflow_experiments, load_mlflow_runs

METRIC_COLS = ("pr_auc", "roc_auc", "f1", "brier", "inner_pr_auc")


def render_model_comparison() -> None:
    st.subheader("Model Comparison")

    experiments = list_mlflow_experiments()
    if not experiments:
        st.warning("No MLflow experiments found. Train a model first.")
        return

    experiment = st.selectbox("Experiment", experiments)
    runs = load_mlflow_runs(experiment)
    if runs.empty:
        st.info(f"No runs in `{experiment}`.")
        return

    st.markdown(f"### Runs ({len(runs)})")
    st.dataframe(runs, use_container_width=True, hide_index=True)

    metric_options = [c for c in METRIC_COLS if c in runs.columns]
    if not metric_options:
        return

    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Metric", metric_options, index=0)
    with col2:
        ascending = metric == "brier"  # lower is better for Brier

    chart_df = runs[["model", metric]].dropna().sort_values(metric, ascending=ascending)
    fig = px.bar(
        chart_df,
        x="model",
        y=metric,
        title=f"{metric} per model — `{experiment}`",
        text=chart_df[metric].round(4),
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Metric matrix")
    matrix_cols = [c for c in METRIC_COLS if c in runs.columns]
    matrix = runs[["model", *matrix_cols]].set_index("model").round(4)
    st.dataframe(matrix, use_container_width=True)
