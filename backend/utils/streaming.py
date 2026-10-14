"""Streaming text filters."""

from __future__ import annotations

import re
from typing import Optional


_THINK_TAGS = ("think", "thinking", "thought", "reasoning")
_OPEN_RE = re.compile(r"<(" + "|".join(_THINK_TAGS) + r")>", re.IGNORECASE)


class ThinkFilter:
    """Streaming remover for <think>…</think>-style blocks (tags may span chunks).

    If a block is never closed, the held text is returned by `flush()` so the user
    still receives an answer instead of an empty message.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._held = ""
        self._close: Optional[re.Pattern[str]] = None
        self._tag = ""

    @staticmethod
    def _partial_open_start(buf: str) -> Optional[int]:
        idx = buf.rfind("<")
        if idx == -1:
            return None
        tail = buf[idx:].lower()
        if any(f"<{t}>".startswith(tail) for t in _THINK_TAGS):
            return idx
        return None

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while True:
            if self._close is None:
                m = _OPEN_RE.search(self._buf)
                if m:
                    out.append(self._buf[:m.start()])
                    self._tag = m.group(1).lower()
                    self._close = re.compile(rf"</{self._tag}>", re.IGNORECASE)
                    self._buf = self._buf[m.end():]
                    self._held = ""
                    continue
                cut = self._partial_open_start(self._buf)
                if cut is None:
                    out.append(self._buf)
                    self._buf = ""
                else:
                    out.append(self._buf[:cut])
                    self._buf = self._buf[cut:]
                break
            m = self._close.search(self._buf)
            if m:
                self._held = ""
                self._buf = self._buf[m.end():]
                self._close = None
                continue
            keep = len(self._tag) + 3                      # possible partial "</tag>"
            if len(self._buf) > keep:
                self._held += self._buf[:-keep]
                self._buf = self._buf[-keep:]
            break
        return "".join(out)

    def flush(self) -> str:
        rest = self._buf
        self._buf = ""
        if self._close is not None:                        # unclosed block → surface it
            held, self._held, self._close = self._held + rest, "", None
            return held
        return rest
