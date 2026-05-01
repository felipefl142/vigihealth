"""Shared filesystem paths for VigiHealth."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MLRUNS_DIR = BASE_DIR / "mlruns"
MLFLOW_DB = BASE_DIR / "mlflow.db"

INDICATOR_CATALOG_URL = "https://ghoapi.azureedge.net/api/Indicator"
GHO_API_BASE_URL = "https://ghoapi.azureedge.net/api"
