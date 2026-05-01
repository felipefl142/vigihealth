"""VigiHealth Streamlit entry point."""

import streamlit as st

from app.tab_duckdb import render_duckdb
from app.tab_eda import render_eda
from app.tab_model_comparison import render_model_comparison
from app.tab_predictions import render_predictions
from app.tab_world_map import render_world_map


st.set_page_config(page_title="VigiHealth", page_icon=":globe_with_meridians:", layout="wide")

st.title("VigiHealth")
st.markdown(
    "Global health analytics on WHO GHO data. DuckDB + Parquet medallion layers, "
    "multiple ML targets, and a Streamlit app with maps."
)

tab_pred, tab_models, tab_eda, tab_map, tab_sql = st.tabs(
    [":crystal_ball: Predictions", ":microscope: Model Comparison", ":bar_chart: EDA", ":world_map: World Map", ":duck: DuckDB Console"]
)

with tab_pred:
    render_predictions()

with tab_models:
    render_model_comparison()

with tab_eda:
    render_eda()

with tab_map:
    render_world_map()

with tab_sql:
    render_duckdb()

def main() -> None:
    return None
