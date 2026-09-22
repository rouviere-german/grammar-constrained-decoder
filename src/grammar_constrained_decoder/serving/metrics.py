"""Decoder metrics on the app's Prometheus registry, plus the prompt-length drift monitor.

Metric names follow the spec: ``tokens_per_second``, ``mask_latency_seconds``,
``index_cache_hit_ratio``, ``compile_seconds``, ``compile_rejections_total``,
``output_validity_ratio`` and ``prompt_length_psi`` (all prefixed with the ``app_`` namespace).
Labels are bounded: rejection reasons come from a fixed set of compiler reason codes, validity is
``true``/``false`` and mode is ``constrained``/``unconstrained``. No label ever carries a prompt,
an output or a schema id supplied by a client.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from grammar_constrained_decoder.ml.drift import psi

__all__ = ["DecoderMetrics", "PromptDriftMonitor", "create_decoder_metrics"]

_REASONS = frozenset(
    {
        "unsupported",
        "invalid",
        "pattern",
        "pattern_length",
        "repeat_bound",
        "depth",
        "schema_depth",
        "ref_depth",
        "schema_size",
        "properties",
        "enum_size",
        "literal_size",
        "number_digits",
        "nfa_states",
        "dfa_states",
        "subset_size",
        "index_entries",
        "timeout",
        "dead_end",
        "empty_language",
        "schema_too_large",
        "internal",
    }
)


@dataclass
class DecoderMetrics:
    tokens_per_second: Histogram
    mask_latency_seconds: Histogram
    index_cache_hits_total: Counter
    index_cache_misses_total: Counter
    index_cache_hit_ratio: Gauge
    compile_seconds: Histogram
    compile_rejections_total: Counter
    outputs_total: Counter
    output_validity_ratio: Gauge
    prompt_length_psi: Gauge
    _tally: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _bump(self, key: str, *pair: str) -> tuple[int, int]:
        with self._lock:
            self._tally[key] = self._tally.get(key, 0) + 1
            return self._tally.get(pair[0], 0), self._tally.get(pair[1], 0)

    def on_compiled(self, seconds: float, *, cache_hit: bool) -> None:
        self.compile_seconds.observe(seconds)
        (self.index_cache_hits_total if cache_hit else self.index_cache_misses_total).inc()
        hits, misses = self._bump("hit" if cache_hit else "miss", "hit", "miss")
        self.index_cache_hit_ratio.set(hits / (hits + misses))

    def on_rejected(self, reason: str) -> None:
        self.compile_rejections_total.labels(reason if reason in _REASONS else "other").inc()

    def observe_output(self, *, constrained: bool, valid: bool) -> None:
        mode = "constrained" if constrained else "unconstrained"
        self.outputs_total.labels(mode, "true" if valid else "false").inc()
        good, bad = self._bump(f"{mode}:{valid}", f"{mode}:True", f"{mode}:False")
        self.output_validity_ratio.labels(mode).set(good / (good + bad))


def create_decoder_metrics(registry: CollectorRegistry, namespace: str = "app") -> DecoderMetrics:
    return DecoderMetrics(
        tokens_per_second=Histogram(
            "tokens_per_second",
            "Generated tokens per second, per request.",
            namespace=namespace,
            registry=registry,
            buckets=(10, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
        ),
        mask_latency_seconds=Histogram(
            "mask_latency_seconds",
            "Time to compute and apply one token mask.",
            namespace=namespace,
            registry=registry,
            buckets=(1e-6, 5e-6, 1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 1e-3, 5e-3),
        ),
        index_cache_hits_total=Counter(
            "index_cache_hits_total",
            "Grammar compilations served from the memory or disk cache.",
            namespace=namespace,
            registry=registry,
        ),
        index_cache_misses_total=Counter(
            "index_cache_misses_total",
            "Grammar compilations built from scratch.",
            namespace=namespace,
            registry=registry,
        ),
        index_cache_hit_ratio=Gauge(
            "index_cache_hit_ratio",
            "Share of grammar compilations served from cache.",
            namespace=namespace,
            registry=registry,
        ),
        compile_seconds=Histogram(
            "compile_seconds",
            "Wall time of schema compilation (parse, DFA, token index).",
            namespace=namespace,
            registry=registry,
            buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
        ),
        compile_rejections_total=Counter(
            "compile_rejections_total",
            "Schemas rejected by the compiler, by reason.",
            ["reason"],
            namespace=namespace,
            registry=registry,
        ),
        outputs_total=Counter(
            "outputs_total",
            "Generated outputs by decoding mode and schema validity.",
            ["mode", "valid"],
            namespace=namespace,
            registry=registry,
        ),
        output_validity_ratio=Gauge(
            "output_validity_ratio",
            "Share of outputs that parse and validate against their schema.",
            ["mode"],
            namespace=namespace,
            registry=registry,
        ),
        prompt_length_psi=Gauge(
            "prompt_length_psi",
            "PSI of recent prompt token lengths against the training prompts.",
            namespace=namespace,
            registry=registry,
        ),
    )


class PromptDriftMonitor:
    """Rolling PSI of prompt token lengths against the training-corpus prompt lengths."""

    def __init__(
        self,
        reference: npt.ArrayLike,
        *,
        window: int = 500,
        min_samples: int = 50,
        every: int = 25,
    ) -> None:
        self.reference = np.asarray(reference, dtype=np.float64)
        if self.reference.size == 0:
            raise ValueError("reference prompt lengths are empty")
        self.recent: deque[int] = deque(maxlen=window)
        self.min_samples = min_samples
        self.every = every
        self._seen = 0
        self._lock = threading.Lock()
        self.last_psi: float | None = None

    def observe(self, length: int) -> float | None:
        """Record one prompt length; returns a fresh PSI every ``every`` observations."""
        with self._lock:
            self.recent.append(length)
            self._seen += 1
            if len(self.recent) < self.min_samples or self._seen % self.every:
                return None
            sample = np.fromiter(self.recent, dtype=np.float64)
        value = psi(self.reference, sample)
        self.last_psi = value
        return value
