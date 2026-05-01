from importlib import import_module
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module_name, required_attrs",
    [
        ("constants", ["DATA_DIR", "RAW_DIR", "BRONZE_DIR", "SILVER_DIR", "GOLD_DIR"]),
        ("app.helpers", ["get_duckdb_connection", "format_iso3", "format_indicator_label"]),
        ("etl.collect", ["collect_indicator", "collect_indicators", "load_indicator_catalog"]),
        ("etl.bronze", ["build_bronze"]),
        ("etl.silver", ["build_silver"]),
        ("etl.gold", ["build_gold"]),
        ("etl.run_pipeline", ["run_pipeline"]),
        ("ml.model_selection", ["get_batch_models"]),
        ("ml.outbreak_model", ["train_outbreak_models", "OUTBREAK_FEATURES"]),
        ("ml.resurgence_model", ["train_resurgence_models", "RESURGENCE_FEATURES"]),
        ("ml.vaccination_model", ["train_vaccination_models", "VACCINATION_FEATURES"]),
        ("app.main", ["main"]),
        ("app.tab_predictions", ["render_predictions"]),
        ("app.tab_model_comparison", ["render_model_comparison"]),
        ("app.tab_eda", ["render_eda"]),
        ("app.tab_world_map", ["render_world_map"]),
        ("app.tab_duckdb", ["render_duckdb"]),
    ],
)
def test_module_contract(module_name, required_attrs):
    module = import_module(module_name)
    for attr in required_attrs:
        assert hasattr(module, attr), f"{module_name} is missing required attribute {attr}"


def test_constants_are_path_objects_when_constants_module_exists():
    constants = import_module("constants")
    for name in ["DATA_DIR", "RAW_DIR", "BRONZE_DIR", "SILVER_DIR", "GOLD_DIR"]:
        assert isinstance(getattr(constants, name), Path)
