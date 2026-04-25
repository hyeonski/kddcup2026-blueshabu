"""Value normalization aligned with the official DABench grader rules.

Rules (https://dataagent.top/rules):
- Numeric: parsed as Decimal, rounded to 2 places ROUND_HALF_UP.
- Dates: ISO 8601 (YYYY-MM-DD).
- DateTime: UTC with Z suffix or preserved ISO format.
- Strings: case-sensitive; leading/trailing whitespace and \\r\\n removed.
- Null variants ("", "null", "none", "nan", "nat", "<na>") all map to "".
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

_NULL_LITERALS = {"", "null", "none", "nan", "nat", "<na>"}
_NUMERIC_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_TWO_PLACES = Decimal("0.01")


def _normalize_string(text: str) -> str:
    cleaned = text.replace("\r\n", "").replace("\n", "").replace("\r", "")
    return cleaned.strip()


def _normalize_numeric(text: str) -> str | None:
    if not _NUMERIC_RE.match(text):
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    quantized = value.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
    return format(quantized, "f")


def _normalize_datetime(text: str) -> str | None:
    candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
        return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
    if "T" in text or " " in text:
        return parsed.isoformat(sep="T", timespec="seconds")
    return parsed.strftime("%Y-%m-%d")


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, Decimal)):
        try:
            quantized = Decimal(str(value)).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
            return format(quantized, "f")
        except InvalidOperation:
            return _normalize_string(str(value))
    text = _normalize_string(str(value))
    if text.lower() in _NULL_LITERALS:
        return ""
    numeric = _normalize_numeric(text)
    if numeric is not None:
        return numeric
    dt_value = _normalize_datetime(text)
    if dt_value is not None:
        return dt_value
    return text


def normalize_column_values(values: list[Any]) -> list[str]:
    return [normalize_value(v) for v in values]
