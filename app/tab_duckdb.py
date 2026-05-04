"""DuckDB console tab — run ad-hoc SQL against medallion parquet."""

from __future__ import annotations

import streamlit as st

from app.helpers import get_duckdb_connection, list_duckdb_views

DEFAULT_QUERY = """-- Sample: top 10 country-years by life expectancy
SELECT country_iso3, year, value
FROM bronze_panel
WHERE indicator_code = 'WHOSIS_000001'
ORDER BY value DESC
LIMIT 10;
"""


def render_duckdb() -> None:
    st.subheader("DuckDB Console")
    st.caption("All bronze/silver/gold parquet files exposed as views (e.g. `bronze_panel`, `silver_fs_country_all`, `gold_abt_vaccination_milestone`).")

    con = get_duckdb_connection()
    with st.expander("Available views", expanded=False):
        st.dataframe(list_duckdb_views(con), use_container_width=True, hide_index=True)

    query = st.text_area("SQL", value=DEFAULT_QUERY, height=200)
    run = st.button("Run", type="primary")

    if not run:
        return

    try:
        result = con.execute(query).df()
    except Exception as exc:
        st.error(str(exc))
        return

    st.success(f"{len(result)} rows")
    st.dataframe(result, use_container_width=True)
    st.download_button(
        "Download CSV",
        result.to_csv(index=False).encode(),
        file_name="duckdb_result.csv",
        mime="text/csv",
    )
