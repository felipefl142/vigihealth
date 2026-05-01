"""Run the full VigiHealth pipeline: collect -> bronze -> silver -> gold."""

from etl.bronze import build_bronze
from etl.collect import collect_indicators
from etl.gold import build_gold
from etl.silver import build_silver


def run_pipeline(indicators=None, force=False):
    if indicators is None:
        indicators = []
    collect_indicators(indicators)
    build_bronze()
    build_silver()
    build_gold()
