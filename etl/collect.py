"""WHO GHO ingestion for VigiHealth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from constants import GHO_API_BASE_URL, INDICATOR_CATALOG_URL, RAW_DIR


@dataclass(frozen=True)
class IndicatorSpec:
    code: str
    title: str | None = None


def load_indicator_catalog() -> list[IndicatorSpec]:
    raise NotImplementedError("WHO GHO ingestion is not implemented yet.")


def collect_indicator(indicator_code: str) -> None:
    raise NotImplementedError("WHO GHO ingestion is not implemented yet.")


def collect_indicators(indicators: Sequence[str]) -> None:
    raise NotImplementedError("WHO GHO ingestion is not implemented yet.")
