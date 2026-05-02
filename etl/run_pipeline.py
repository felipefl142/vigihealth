"""Run the full VigiHealth pipeline: collect -> bronze -> silver -> gold."""

from __future__ import annotations

import argparse
import logging

from constants import V1_VACCINATION_INDICATORS
from etl.bronze import build_bronze
from etl.collect import collect_indicators
from etl.gold import build_gold
from etl.silver import build_silver

logger = logging.getLogger(__name__)


def run_pipeline(
    indicators: list[str] | None = None,
    *,
    force: bool = False,
    skip_silver: bool = False,
    skip_gold: bool = False,
) -> None:
    codes = indicators if indicators else V1_VACCINATION_INDICATORS
    logger.info("pipeline: %d indicators (force=%s)", len(codes), force)
    collect_indicators(codes, force=force)
    build_bronze()
    if skip_silver:
        logger.info("skip silver")
        return
    build_silver()
    if skip_gold:
        logger.info("skip gold")
        return
    build_gold()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the VigiHealth pipeline.")
    p.add_argument("--indicators", nargs="*", default=None,
                   help="Indicator codes. Defaults to V1_VACCINATION_INDICATORS.")
    p.add_argument("--force", action="store_true", help="Re-collect existing indicators.")
    p.add_argument("--skip-silver", action="store_true", help="Stop after bronze.")
    p.add_argument("--skip-gold", action="store_true", help="Stop after silver.")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_pipeline(
        indicators=args.indicators,
        force=args.force,
        skip_silver=args.skip_silver,
        skip_gold=args.skip_gold,
    )


if __name__ == "__main__":
    main()
