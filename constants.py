"""Shared filesystem paths for VigiHealth."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
INDICATORS_DIR = DATA_DIR / "indicators"
MLRUNS_DIR = BASE_DIR / "mlruns"
MLFLOW_DB = BASE_DIR / "mlflow.db"

INDICATOR_CATALOG_URL = "https://ghoapi.azureedge.net/api/Indicator"
GHO_API_BASE_URL = "https://ghoapi.azureedge.net/api"

V1_VACCINATION_INDICATORS = [
    "WHS4_100",
    "WHS4_544",
    "WHS4_117",
    "WHS4_543",
    "GHED_CHEGDP_SHA2011",
    "WHOSIS_000001",
    "MDG_0000000007",
    "WSH_WATER_SAFELY_MANAGED",
    "WSH_SANITATION_SAFELY_MANAGED",
]

REGIONAL_AGGREGATE_CODES = {
    "GLOBAL", "AFR", "AMR", "EMR", "EUR", "SEAR", "WPR",
    "WB_LI", "WB_LMI", "WB_UMI", "WB_HI",
}
