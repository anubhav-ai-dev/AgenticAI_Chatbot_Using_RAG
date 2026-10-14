"""Server-sent-events wire format."""

from __future__ import annotations

import json
from typing import Any


def format_sse(event: dict[str, Any]) -> str:
    """Encode {"event": name, "data": obj} as one SSE frame."""
    payload = json.dumps(event["data"], ensure_ascii=False, default=str)
    return f"event: {event['event']}\ndata: {payload}\n\n"
