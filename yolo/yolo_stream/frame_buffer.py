"""Single-slot latest-frame buffer between the capture and processing threads."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Frame:
    seq: int  # capture sequence number, starts at 1
    timestamp: float  # time.monotonic() when captured
    image: Any


class LatestFrameBuffer:
    """Holds at most one frame. A new frame replaces an unread one, which is
    counted as dropped, so memory and latency stay bounded however slow the
    consumer is."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: Optional[Frame] = None
        self._put = 0
        self._dropped = 0

    def put(self, image: Any) -> None:
        with self._cond:
            if self._frame is not None:
                self._dropped += 1
            self._put += 1
            self._frame = Frame(self._put, time.monotonic(), image)
            self._cond.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[Frame]:
        """Take the newest unread frame, waiting up to `timeout` seconds."""
        with self._cond:
            if self._frame is None:
                self._cond.wait(timeout)
            frame, self._frame = self._frame, None
            return frame

    @property
    def put_count(self) -> int:
        with self._cond:
            return self._put

    @property
    def dropped_count(self) -> int:
        with self._cond:
            return self._dropped
