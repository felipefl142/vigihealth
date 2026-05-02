"""Bronze layer builder for VigiHealth.

Stacks every Parquet under ``RAW_DIR`` into a tidy long panel and writes:

* ``BRONZE_DIR/panel.parquet``     — country × year × indicator (modeling input)
* ``BRONZE_DIR/regions.parquet``   — regional / income-group / global aggregates
* ``INDICATORS_DIR/coverage.parquet`` — per-indicator country×year fill rate
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb

from constants import BRONZE_DIR, INDICATORS_DIR, RAW_DIR

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parent / "sql"


def _load_sql(name: str) -> str:
    return (SQL_DIR / name).read_text()


def build_bronze() -> None:
    raw_glob = str(RAW_DIR / "*.parquet")
    raw_files = sorted(RAW_DIR.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No raw indicator files in {RAW_DIR}. Run etl.collect first.")

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    INDICATORS_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    panel = con.execute(_load_sql("bronze_panel.sql"), [raw_glob]).fetch_df()
    panel_out = BRONZE_DIR / "panel.parquet"
    panel.to_parquet(panel_out, index=False)
    logger.info("wrote %s (%d rows, %d indicators, %d countries)",
                panel_out, len(panel),
                panel["indicator_code"].nunique(),
                panel["country_iso3"].nunique())

    regions = con.execute(_load_sql("bronze_regions.sql"), [raw_glob]).fetch_df()
    regions_out = BRONZE_DIR / "regions.parquet"
    regions.to_parquet(regions_out, index=False)
    logger.info("wrote %s (%d rows)", regions_out, len(regions))

    coverage = (
        panel.groupby("indicator_code")
        .agg(
            country_years=("value", "size"),
            countries=("country_iso3", "nunique"),
            year_min=("year", "min"),
            year_max=("year", "max"),
        )
        .reset_index()
        .sort_values("country_years", ascending=False)
    )
    coverage_out = INDICATORS_DIR / "coverage.parquet"
    coverage.to_parquet(coverage_out, index=False)
    logger.info("wrote %s\n%s", coverage_out, coverage.to_string(index=False))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build VigiHealth bronze layer.")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_bronze()


if __name__ == "__main__":
    main()
