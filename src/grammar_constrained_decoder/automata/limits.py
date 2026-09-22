"""Resource limits for schema compilation and the errors raised when they are hit.

Compilation takes attacker-controlled input (inline schemas on the API), and a regular language
can have a DFA exponentially larger than its regex. Every stage therefore charges a
:class:`CompileBudget`: NFA states, DFA states, the NFA-state entries held by subset construction
(its memory) and a wall-clock deadline are checked inside the construction loops, so a hostile
schema is rejected with a typed error instead of exhausting CPU or memory. The deadline is
cooperative (checked in loops), so it needs no threads or signals and behaves the same on Windows
and Linux; a rejection can therefore land slightly after the deadline (one check interval).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_LIMITS",
    "CompileBudget",
    "CompileError",
    "CompileLimitError",
    "CompileLimits",
    "SchemaError",
]


class CompileError(ValueError):
    """Base class for every rejection by the compiler. ``reason`` is a stable metric label."""

    reason: str = "invalid"

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        if reason is not None:
            self.reason = reason


class SchemaError(CompileError):
    """The schema is invalid or uses a keyword outside the supported subset.

    ``path`` is a JSON Pointer to the offending location; the message never echoes schema values
    beyond keyword names, so errors are safe to return to API clients.
    """

    def __init__(self, path: str, message: str, *, reason: str = "unsupported") -> None:
        super().__init__(f"{path or '/'}: {message}", reason=reason)
        self.path = path or "/"


class CompileLimitError(CompileError):
    """A resource limit was exceeded (state caps, depth, enum size, timeout ...)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message, reason=reason)


@dataclass(frozen=True, slots=True)
class CompileLimits:
    """Caps applied while compiling one schema. Defaults suit the bundled schemas with headroom."""

    max_depth: int = 6
    max_json_depth: int = 64
    max_schema_nodes: int = 2_000
    max_ref_depth: int = 64
    max_properties: int = 64
    max_enum_values: int = 256
    max_literal_bytes: int = 4_096
    max_pattern_length: int = 256
    max_repeat: int = 256
    max_number_digits: int = 24
    max_nfa_states: int = 200_000
    max_dfa_states: int = 20_000
    max_subset_entries: int = 1_000_000
    max_index_entries: int = 8_000_000
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    def cache_key(self) -> str:
        """Limits change which schemas compile, so they are part of the index cache key."""
        return ",".join(
            f"{name}={getattr(self, name)}" for name in sorted(self.__dataclass_fields__)
        )


DEFAULT_LIMITS = CompileLimits()


@dataclass
class CompileBudget:
    """Mutable per-compilation counters plus the deadline."""

    limits: CompileLimits = DEFAULT_LIMITS
    clock: Callable[[], float] = time.monotonic
    started: float = field(default=0.0)
    deadline: float = field(default=0.0)
    nfa_states: int = 0
    dfa_states: int = 0
    subset_entries: int = 0
    _ticks: int = 0

    def __post_init__(self) -> None:
        self.started = self.clock()
        self.deadline = self.started + self.limits.timeout_seconds

    def elapsed(self) -> float:
        return self.clock() - self.started

    def check_time(self) -> None:
        if self.clock() > self.deadline:
            raise CompileLimitError(
                "timeout",
                f"compilation exceeded the {self.limits.timeout_seconds:g}s time limit",
            )

    def tick(self, every: int = 64) -> None:
        """Cheap periodic deadline check for tight loops."""
        self._ticks += 1
        if self._ticks % every == 0:
            self.check_time()

    def charge_nfa(self, count: int = 1) -> None:
        self.nfa_states += count
        if self.nfa_states > self.limits.max_nfa_states:
            raise CompileLimitError(
                "nfa_states",
                f"NFA exceeds {self.limits.max_nfa_states} states",
            )
        self.tick()

    def charge_dfa(self, count: int = 1) -> None:
        self.dfa_states += count
        if self.dfa_states > self.limits.max_dfa_states:
            raise CompileLimitError(
                "dfa_states",
                f"DFA exceeds {self.limits.max_dfa_states} states",
            )
        self.check_time()

    def charge_subsets(self, entries: int) -> None:
        """Charge NFA-state entries held in subset-construction sets (the memory of determinize).

        The DFA state cap bounds how many subsets exist, not how large each one is; a pattern
        such as ``((a{0,90}){0,90}){0,90}`` creates few DFA states whose subsets hold thousands
        of NFA states each. Charging the entries caps that memory (roughly 40 bytes per entry).
        """
        self.subset_entries += entries
        if self.subset_entries > self.limits.max_subset_entries:
            raise CompileLimitError(
                "subset_size",
                f"subset construction holds more than {self.limits.max_subset_entries} "
                "NFA-state entries",
            )
        self.tick()
