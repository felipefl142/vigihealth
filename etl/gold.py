"""Gold layer builder for VigiHealth.

Builds one ABT per ML target. v1 ships ``abt_vaccination_milestone``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd

from constants import GOLD_DIR, SILVER_DIR

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).resolve().parent / "sql"

VACCINES = ("dtp3", "mcv1", "hepb3", "pol3")
HORIZONS = (1, 3, 5)
MILESTONE_THRESHOLD = 90.0


def _load_sql(name: str) -> str:
    return (SQL_DIR / name).read_text()


def build_vaccination_milestone() -> Path:
    silver_all = SILVER_DIR / "fs_country_all.parquet"
    silver_pivot_view_input = SILVER_DIR / "fs_country_life.parquet"  # carries _value cols
    if not silver_all.exists():
        raise FileNotFoundError(f"{silver_all} missing. Run etl.silver first.")

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # Pivot of vaccine *_value columns to feed the future-window label SQL.
    features = pd.read_parquet(silver_all)
    vacc_value_cols = {
        "WHS4_100_value": "WHS4_100",
        "WHS4_544_value": "WHS4_544",
        "WHS4_117_value": "WHS4_117",
        "WHS4_543_value": "WHS4_543",
    }
    vacc_panel = features[["country_iso3", "year"] + list(vacc_value_cols.keys())].rename(
        columns=vacc_value_cols
    )
    vacc_panel_path = GOLD_DIR / "_vaccine_panel.parquet"
    vacc_panel.to_parquet(vacc_panel_path, index=False)

    label_wide = con.execute(
        _load_sql("abt_vaccination_milestone.sql"), [str(vacc_panel_path)]
    ).fetch_df()

    # Stack labels long: one row per (country, year, vaccine, horizon).
    rows: list[pd.DataFrame] = []
    for vaccine in VACCINES:
        for h in HORIZONS:
            col = f"{vaccine}_h{h}_max"
            sub = label_wide[["country_iso3", "year", col]].rename(columns={col: "future_max"})
            sub = sub[sub["future_max"].notna()].copy()
            sub["vaccine"] = vaccine
            sub["horizon"] = h
            sub["label"] = (sub["future_max"] >= MILESTONE_THRESHOLD).astype("int8")
            rows.append(sub.drop(columns=["future_max"]))
    labels_long = pd.concat(rows, ignore_index=True)

    # Join silver features. One row per (country, year) gets repeated across
    # (vaccine, horizon) combinations.
    abt = labels_long.merge(features, on=["country_iso3", "year"], how="left")

    out = GOLD_DIR / "abt_vaccination_milestone.parquet"
    abt.to_parquet(out, index=False)
    vacc_panel_path.unlink(missing_ok=True)

    logger.info(
        "wrote %s (%d rows, %d cols, label rate=%.3f)",
        out, len(abt), len(abt.columns), abt["label"].mean(),
    )
    by_pair = (
        abt.groupby(["vaccine", "horizon"])["label"]
        .agg(["size", "mean"])
        .rename(columns={"size": "n_rows", "mean": "label_rate"})
    )
    logger.info("per-pair label rates:\n%s", by_pair.to_string())
    return out


def build_gold() -> None:
    build_vaccination_milestone()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build VigiHealth gold ABTs.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_gold()


if __name__ == "__main__":
    main()
