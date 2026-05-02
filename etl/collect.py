"""WHO GHO ingestion for VigiHealth.

Fetches the indicator catalog and per-indicator observation Parquets via the
public OData endpoint at https://ghoapi.azureedge.net/api/. One Parquet per
indicator code is written to ``RAW_DIR``. Idempotent: existing files are
skipped unless ``force=True``.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import requests

from constants import (
    GHO_API_BASE_URL,
    INDICATOR_CATALOG_URL,
    RAW_DIR,
    V1_VACCINATION_INDICATORS,
)

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
BACKOFF_BASE = 1.5
USER_AGENT = "VigiHealth/0.1 (+https://github.com/felipefl142/vigihealth)"


@dataclass(frozen=True)
class IndicatorSpec:
    code: str
    title: str | None = None


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


def _get_json(session: requests.Session, url: str) -> dict:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            sleep_for = BACKOFF_BASE ** attempt
            logger.warning(
                "GET %s failed (attempt %d/%d): %s — retrying in %.1fs",
                url, attempt, MAX_RETRIES, exc, sleep_for,
            )
            time.sleep(sleep_for)
    raise RuntimeError(f"GET {url} failed after {MAX_RETRIES} attempts") from last_exc


def load_indicator_catalog() -> list[IndicatorSpec]:
    """Return the full GHO indicator catalog."""
    session = _session()
    payload = _get_json(session, INDICATOR_CATALOG_URL)
    rows = payload.get("value", [])
    return [
        IndicatorSpec(code=row["IndicatorCode"], title=row.get("IndicatorName"))
        for row in rows
    ]


def _fetch_indicator_rows(session: requests.Session, code: str) -> list[dict]:
    url = f"{GHO_API_BASE_URL}/{code}"
    rows: list[dict] = []
    while url:
        payload = _get_json(session, url)
        rows.extend(payload.get("value", []))
        url = payload.get("@odata.nextLink") or ""
    return rows


def collect_indicator(indicator_code: str, *, force: bool = False) -> None:
    """Fetch one indicator and write to ``RAW_DIR/{code}.parquet``."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{indicator_code}.parquet"
    if out.exists() and not force:
        logger.info("skip %s (exists)", indicator_code)
        return

    session = _session()
    rows = _fetch_indicator_rows(session, indicator_code)
    if not rows:
        logger.warning("no rows returned for %s", indicator_code)
        return

    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    logger.info("wrote %s (%d rows -> %s)", indicator_code, len(df), out)


def collect_indicators(indicators: Sequence[str], *, force: bool = False) -> None:
    for code in indicators:
        collect_indicator(code, force=force)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect WHO GHO indicators.")
    p.add_argument(
        "--indicators",
        nargs="*",
        default=None,
        help="Indicator codes to fetch. Defaults to V1_VACCINATION_INDICATORS.",
    )
    p.add_argument("--force", action="store_true", help="Re-download existing files.")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    codes = args.indicators if args.indicators else V1_VACCINATION_INDICATORS
    logger.info("collecting %d indicators -> %s", len(codes), RAW_DIR)
    collect_indicators(codes, force=args.force)


if __name__ == "__main__":
    main()
