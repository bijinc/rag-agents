"""Shared finance-domain normalization utilities.

These helpers centralize ticker alias resolution, fiscal period normalization,
and section label normalization so ingestion, query analysis, and retrieval use consistent contracts.
"""

from __future__ import annotations

import re
from typing import Iterable

VALID_TICKERS = {"AAPL", "JPM", "F"}

# Keep aliases lowercase and punctuation-light for robust lookup.
_TICKER_ALIASES = {
    "apple": "AAPL",
    "apple inc": "AAPL",
    "jpmorgan": "JPM",
    "jpmorgan chase": "JPM",
    "jp morgan": "JPM",
    "ford": "F",
    "ford motor": "F",
}

VALID_SOURCE_TYPES = {"sec_filing", "ect"}
VALID_FILING_TYPES = {"10-K", "10-Q", "8-K"}
VALID_SECTIONS = {
    "risk_factors",
    "md&a",
    "financial_statements",
    "forward_looking_statements",
    "q_and_a",
    "prepared_remarks",
    "general",
}

_SECTION_ALIASES = {
    "risk factors": "risk_factors",
    "risk_factor": "risk_factors",
    "mda": "md&a",
    "md and a": "md&a",
    "management discussion and analysis": "md&a",
    "financial statement": "financial_statements",
    "financial statements": "financial_statements",
    "forward looking statements": "forward_looking_statements",
    "forward-looking-statements": "forward_looking_statements",
    "q&a": "q_and_a",
    "q and a": "q_and_a",
    "prepared remarks": "prepared_remarks",
}


def normalize_ticker(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    upper = raw.upper()
    if upper in VALID_TICKERS:
        return upper

    key = re.sub(r"\s+", " ", raw.lower()).strip()
    key = key.replace(".", "")
    alias = _TICKER_ALIASES.get(key)
    if alias in VALID_TICKERS:
        return alias
    return None


def normalize_tickers(values: Iterable[str] | None) -> list[str] | None:
    if values is None:
        return None
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        ticker = normalize_ticker(value)
        if ticker and ticker not in seen:
            seen.add(ticker)
            deduped.append(ticker)
    return deduped or None


def normalize_source_type(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip().lower()
    if raw in {"", "null", "none", "n/a"}:
        return None
    return raw if raw in VALID_SOURCE_TYPES else None


def normalize_filing_type(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip().upper()
    if raw in {"", "NULL", "NONE", "N/A"}:
        return None
    if raw in {"10K", "10 K"}:
        raw = "10-K"
    elif raw in {"10Q", "10 Q"}:
        raw = "10-Q"
    elif raw in {"8K", "8 K"}:
        raw = "8-K"
    return raw if raw in VALID_FILING_TYPES else None


def normalize_fiscal_quarter(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip().upper()
    if raw in {"", "NULL", "NONE", "N/A"}:
        return None
    if raw in {"1", "2", "3", "4"}:
        return f"Q{raw}"
    if raw in {"Q1", "Q2", "Q3", "Q4"}:
        return raw
    return None


def canonical_period_key(year: int | str | None, quarter: str | None) -> str | None:
    if year is None or quarter is None:
        return None
    try:
        y = int(year)
    except (TypeError, ValueError):
        return None
    q = normalize_fiscal_quarter(quarter)
    if q is None:
        return None
    return f"{y}-{q}"


def parse_year_quarter(value: str | None) -> tuple[int | None, str | None]:
    if value is None:
        return None, None
    raw = value.strip().upper()
    if not raw:
        return None, None

    # Matches: 2024_Q1, 2024-Q1, 2024 Q1
    m = re.search(r"\b(20\d{2})[-_\s]?Q([1-4])\b", raw)
    if m:
        return int(m.group(1)), f"Q{m.group(2)}"

    # Matches: Q1 2024, Q1-2024
    m = re.search(r"\bQ([1-4])[-_\s]?(20\d{2})\b", raw)
    if m:
        return int(m.group(2)), f"Q{m.group(1)}"

    # Matches annual forms like FY2024; no quarter signal available.
    m = re.search(r"\bFY[-_\s]?(20\d{2})\b", raw)
    if m:
        return int(m.group(1)), None

    return None, None


def normalize_section_types(values: Iterable[str] | None) -> list[str] | None:
    if values is None:
        return None

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        raw = str(value).strip()
        if not raw:
            continue

        lower = raw.lower()
        candidate = _SECTION_ALIASES.get(lower, lower)
        if candidate in VALID_SECTIONS and candidate not in seen:
            seen.add(candidate)
            normalized.append(candidate)

    return normalized or None
