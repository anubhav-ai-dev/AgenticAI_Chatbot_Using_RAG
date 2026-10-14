"""Time helpers."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

_DURATION = re.compile(
    r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?")


def utcnow() -> datetime:
    """Naive UTC timestamp (keeps compatibility with rows written before)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_duration(value: Optional[str]) -> Optional[float]:
    """Parse '2.28s', '1m26.4s', '120ms' or a bare number of seconds."""
    if value is None:
        return None
    value = str(value).strip().lower()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    m = _DURATION.fullmatch(value)
    if not m or not any(m.groups()):
        return None
    h, mi, s, ms = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000.0
