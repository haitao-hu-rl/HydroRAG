from __future__ import annotations

import threading
import time
from typing import Any


class Metrics:
    """In-process counters suitable for a single-node health/operations view."""

    def __init__(self):
        self.started_at = time.time()
        self._counters: dict[str, int] = {}
        self._timers: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            timer = self._timers.setdefault(name, {"count": 0.0, "total": 0.0, "max": 0.0})
            timer["count"] += 1
            timer["total"] += float(value)
            timer["max"] = max(timer["max"], float(value))

    def record_http(self, status_code: int, latency_ms: float) -> None:
        self.increment("http_requests_total")
        self.increment(f"http_status_{int(status_code)}_total")
        self.observe("http_latency_ms", latency_ms)

    def snapshot(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            timers = {
                name: {
                    "count": int(values["count"]),
                    "average": round(values["total"] / values["count"], 2)
                    if values["count"]
                    else 0.0,
                    "max": round(values["max"], 2),
                }
                for name, values in self._timers.items()
            }
        result: dict[str, Any] = {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "counters": counters,
            "timers": timers,
        }
        if extra:
            result.update(extra)
        return result
