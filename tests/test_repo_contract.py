from importlib import import_module
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    [
        "constants.py",
        "app/__init__.py",
        "app/main.py",
        "app/helpers.py",
        "etl/__init__.py",
        "etl/collect.py",
        "etl/bronze.py",
        "etl/silver.py",
        "etl/gold.py",
        "etl/run_pipeline.py",
        "ml/__init__.py",
        "ml/model_selection.py",
        "ml/outbreak_model.py",
        "ml/resurgence_model.py",
        "ml/vaccination_model.py",
        "README.md",
        "CLAUDE.md",
        "Plan.md",
    ],
)
def test_expected_project_files_exist(relative_path):
    assert (ROOT / relative_path).exists(), f"Missing required project file: {relative_path}"


@pytest.mark.parametrize(
    "module_name",
    [
        "constants",
        "app.main",
        "app.helpers",
        "etl.collect",
        "etl.bronze",
        "etl.silver",
        "etl.gold",
        "etl.run_pipeline",
        "ml.model_selection",
        "ml.outbreak_model",
        "ml.resurgence_model",
        "ml.vaccination_model",
    ],
)
def test_expected_modules_are_importable(module_name):
    import_module(module_name)
