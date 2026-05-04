"""Shared UI helpers for VigiHealth."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import streamlit as st

from constants import (
    BRONZE_DIR,
    GOLD_DIR,
    MLFLOW_DB,
    MLRUNS_DIR,
    REGIONAL_AGGREGATE_CODES,
    SILVER_DIR,
)


# ---------------------------------------------------------------------------
# Cached parquet loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_bronze_panel() -> pd.DataFrame:
    return pd.read_parquet(BRONZE_DIR / "panel.parquet")


@st.cache_data(show_spinner=False)
def load_bronze_regions() -> pd.DataFrame:
    return pd.read_parquet(BRONZE_DIR / "regions.parquet")


@st.cache_data(show_spinner=False)
def load_silver(table: str = "fs_country_all") -> pd.DataFrame:
    path = SILVER_DIR / f"{table}.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_gold(target: str) -> pd.DataFrame:
    path = GOLD_DIR / f"abt_{target}.parquet"
    return pd.read_parquet(path)


def list_gold_targets() -> list[str]:
    return sorted(p.stem.replace("abt_", "") for p in GOLD_DIR.glob("abt_*.parquet"))


def list_silver_tables() -> list[str]:
    return sorted(p.stem for p in SILVER_DIR.glob("*.parquet"))


def list_indicators(panel: pd.DataFrame) -> list[str]:
    return sorted(panel["indicator_code"].dropna().unique().tolist())


def list_iso3_countries(panel: pd.DataFrame) -> list[str]:
    codes = panel["country_iso3"].dropna().unique().tolist()
    return sorted(c for c in codes if c not in REGIONAL_AGGREGATE_CODES and len(c) == 3)


# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------
def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with bronze/silver/gold parquet files registered as views."""
    con = duckdb.connect()
    for layer_dir, prefix in [(BRONZE_DIR, "bronze"), (SILVER_DIR, "silver"), (GOLD_DIR, "gold")]:
        for parquet in sorted(layer_dir.glob("*.parquet")):
            view = f"{prefix}_{parquet.stem}"
            con.execute(
                f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{parquet.as_posix()}')"
            )
    return con


def list_duckdb_views(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        "SELECT table_schema, table_name FROM information_schema.tables WHERE table_type='VIEW' ORDER BY table_name"
    ).df()


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _mlflow_module():
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    return mlflow


@st.cache_data(show_spinner=False)
def list_mlflow_experiments() -> list[str]:
    mlflow = _mlflow_module()
    return [e.name for e in mlflow.search_experiments() if e.name != "Default"]


@st.cache_data(show_spinner=False)
def load_mlflow_runs(experiment: str) -> pd.DataFrame:
    mlflow = _mlflow_module()
    runs = mlflow.search_runs(experiment_names=[experiment])
    if runs.empty:
        return runs
    runs = runs.copy()
    runs["model"] = runs.get("tags.mlflow.runName", runs.get("params.model"))
    keep = [
        "run_id",
        "model",
        "metrics.pr_auc",
        "metrics.roc_auc",
        "metrics.f1",
        "metrics.brier",
        "metrics.inner_pr_auc",
        "params.split_year",
        "params.n_trials",
        "start_time",
    ]
    keep = [c for c in keep if c in runs.columns]
    runs = runs[keep].sort_values("metrics.pr_auc", ascending=False).reset_index(drop=True)
    runs.columns = [c.replace("metrics.", "").replace("params.", "") for c in runs.columns]
    return runs


@st.cache_resource(show_spinner=False)
def load_best_model(experiment: str, metric: str = "pr_auc"):
    mlflow = _mlflow_module()
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if runs.empty:
        return None, None
    run_id = runs.iloc[0]["run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    return model, run_id


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def format_iso3(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def format_indicator_label(code: str) -> str:
    code = str(code).strip()
    if not code:
        return code
    return code.replace("_", " ").title()


def normalize_country_name(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()
