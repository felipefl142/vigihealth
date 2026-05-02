"""Silver layer builder for VigiHealth.

Bronze (long) -> wide pivot per (country, year) -> per-indicator temporal features
across four windows (life, last5, last10, last20). Output:

* ``SILVER_DIR/fs_country_life.parquet``
* ``SILVER_DIR/fs_country_last5.parquet``
* ``SILVER_DIR/fs_country_last10.parquet``
* ``SILVER_DIR/fs_country_last20.parquet``
* ``SILVER_DIR/fs_country_all.parquet``  -- all windows joined for modeling

All features are point-in-time correct: the row at (country, year=Y) only uses
observations with year <= Y. Time-based train/test splits downstream remain
honest because no future leakage enters here.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb

from constants import BRONZE_DIR, SILVER_DIR

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parent / "sql"

WINDOWS = {
    "life": None,    # all prior years
    "last5": 5,
    "last10": 10,
    "last20": 20,
}

LAGS = (1, 3, 5)


def _load_sql(name: str) -> str:
    return (SQL_DIR / name).read_text()


def _window_clause(size: int | None) -> str:
    if size is None:
        return "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    return f"ROWS BETWEEN {size - 1} PRECEDING AND CURRENT ROW"


def _feature_sql(indicator_cols: list[str], window_name: str, size: int | None) -> str:
    """Generate a SELECT that computes per-indicator window features.

    For window W and indicator I, emit:
        I_W_mean, I_W_std, I_W_slope, I_W_min, I_W_max
    Plus point-in-time lags (lag1, lag3, lag5) and YoY change, which are
    window-independent — emitted only for the 'life' window to avoid duplication.
    """
    rows_clause = _window_clause(size)
    parts: list[str] = ["country_iso3", "year"]

    for col in indicator_cols:
        q = f'"{col}"'
        parts.extend([
            f"{q} AS {col}_value",
            f"AVG({q})       OVER w_{window_name} AS {col}_{window_name}_mean",
            f"STDDEV_SAMP({q}) OVER w_{window_name} AS {col}_{window_name}_std",
            f"REGR_SLOPE({q}, year) OVER w_{window_name} AS {col}_{window_name}_slope",
            f"MIN({q})       OVER w_{window_name} AS {col}_{window_name}_min",
            f"MAX({q})       OVER w_{window_name} AS {col}_{window_name}_max",
        ])
        if window_name == "life":
            for lag in LAGS:
                parts.append(
                    f"LAG({q}, {lag}) OVER (PARTITION BY country_iso3 ORDER BY year) "
                    f"AS {col}_lag{lag}"
                )
            parts.append(
                f"({q} - LAG({q}, 1) OVER (PARTITION BY country_iso3 ORDER BY year)) "
                f"AS {col}_yoy"
            )

    select_list = ",\n    ".join(parts)
    return f"""
SELECT
    {select_list}
FROM pivoted
WINDOW w_{window_name} AS (
    PARTITION BY country_iso3
    ORDER BY year
    {rows_clause}
)
ORDER BY country_iso3, year
"""


def build_silver() -> None:
    panel_path = BRONZE_DIR / "panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} missing. Run etl.bronze first.")

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        f"CREATE TEMP VIEW panel AS SELECT * FROM read_parquet('{panel_path}')"
    )

    pivoted = con.execute(_load_sql("silver_pivot.sql")).fetch_df()
    indicator_cols = [c for c in pivoted.columns if c not in ("country_iso3", "year")]
    logger.info("pivoted: %d rows, %d indicator columns", len(pivoted), len(indicator_cols))

    con.register("pivoted", pivoted)

    per_window: dict[str, "pd.DataFrame"] = {}
    for window_name, size in WINDOWS.items():
        sql = _feature_sql(indicator_cols, window_name, size)
        df = con.execute(sql).fetch_df()
        out = SILVER_DIR / f"fs_country_{window_name}.parquet"
        df.to_parquet(out, index=False)
        per_window[window_name] = df
        logger.info("wrote %s (%d rows, %d cols)", out, len(df), len(df.columns))

    # Join all windows on (country_iso3, year). Drop duplicate _value/_lag/_yoy
    # columns that 'life' already carries.
    base = per_window["life"]
    keys = ["country_iso3", "year"]
    merged = base
    for window_name in ("last5", "last10", "last20"):
        df = per_window[window_name]
        keep = keys + [c for c in df.columns if c.endswith(f"_{window_name}_mean")
                       or c.endswith(f"_{window_name}_std")
                       or c.endswith(f"_{window_name}_slope")
                       or c.endswith(f"_{window_name}_min")
                       or c.endswith(f"_{window_name}_max")]
        merged = merged.merge(df[keep], on=keys, how="left")

    out_all = SILVER_DIR / "fs_country_all.parquet"
    merged.to_parquet(out_all, index=False)
    logger.info("wrote %s (%d rows, %d cols)", out_all, len(merged), len(merged.columns))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build VigiHealth silver layer.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_silver()


if __name__ == "__main__":
    main()
