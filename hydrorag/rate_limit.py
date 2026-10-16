from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Process-local sliding-window limiter for the single-instance prototype."""

    def __init__(self):
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._calls = 0

    def allow(self, key: str, limit: int, window_seconds: float = 60.0) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - max(1.0, float(window_seconds))
        with self._lock:
            self._calls += 1
            if self._calls % 128 == 0:
                expired = [key for key, values in self._events.items() if not values or values[-1] <= cutoff]
                for expired_key in expired:
                    del self._events[expired_key]
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= max(1, int(limit)):
                retry_after = max(1, int(events[0] + window_seconds - now + 0.999))
                return False, retry_after
            events.append(now)
            return True, 0
