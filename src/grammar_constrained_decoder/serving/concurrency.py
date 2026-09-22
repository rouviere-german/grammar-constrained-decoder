"""Non-blocking concurrency caps: a global limit and a per-client limit on running generations.

Decoding is CPU-bound, so a flood of slow generations would starve every other caller even under
the request-rate limit. :meth:`ConcurrencyLimiter.acquire` never waits: it either returns a
:class:`Lease` or ``None`` (the API answers 429). Leases are released exactly once, including when
a streaming client disconnects mid-stream.
"""

from __future__ import annotations

import threading
from collections import Counter

__all__ = ["ConcurrencyLimiter", "Lease"]


class Lease:
    def __init__(self, limiter: ConcurrencyLimiter, client: str) -> None:
        self._limiter = limiter
        self.client = client
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._limiter._release(self.client)


class ConcurrencyLimiter:
    def __init__(self, global_limit: int, per_client_limit: int) -> None:
        if global_limit < 1 or per_client_limit < 1:
            raise ValueError("limits must be positive")
        self.global_limit = global_limit
        self.per_client_limit = per_client_limit
        self._active: Counter[str] = Counter()
        self._total = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> int:
        return self._total

    def acquire(self, client: str) -> Lease | None:
        with self._lock:
            if self._total >= self.global_limit or self._active[client] >= self.per_client_limit:
                return None
            self._active[client] += 1
            self._total += 1
        return Lease(self, client)

    def _release(self, client: str) -> None:
        with self._lock:
            self._active[client] -= 1
            if self._active[client] <= 0:
                del self._active[client]
            self._total -= 1
