"""Shared UI helpers for VigiHealth."""

from __future__ import annotations

from typing import Optional

import duckdb

from constants import BRONZE_DIR, GOLD_DIR, RAW_DIR, SILVER_DIR


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


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
