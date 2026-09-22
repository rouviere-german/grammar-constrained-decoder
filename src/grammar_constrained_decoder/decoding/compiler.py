"""Schema -> (regex, minimal DFA, token index), with resource limits and a two-level cache.

``GrammarCompiler.compile`` runs the whole pipeline under one :class:`CompileBudget`: parse the
schema subset, lower it to a regex, build the minimal DFA, build the token index and prove there
are no dead ends. Any limit breach raises a :class:`CompileError` with a stable ``reason``.

Compiled artifacts are cached in memory (LRU) and on disk, keyed by
``sha256(compiler version, schema canonical-JSON hash, vocabulary hash, limits)``. The disk entry is
an ``.npz`` (``allow_pickle=False``) holding the DFA and index arrays plus a JSON header; it
carries a SHA-256 over its own arrays, and any mismatch, parse failure or structural problem is
treated as a miss and recompiled. The checksum detects corruption, not a malicious writer: the
cache directory must be writable only by the service (see docs/THREAT_MODEL.md).
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
import zipfile
import zlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from grammar_constrained_decoder.automata.dfa import DFA, compile_regex
from grammar_constrained_decoder.automata.limits import (
    DEFAULT_LIMITS,
    CompileBudget,
    CompileError,
    CompileLimits,
)
from grammar_constrained_decoder.automata.regex import EMPTY, Regex, size
from grammar_constrained_decoder.core.canonical import sha256_hex
from grammar_constrained_decoder.core.fileio import atomic_write_bytes
from grammar_constrained_decoder.decoding.index import TokenIndex, build_token_index
from grammar_constrained_decoder.lm.tokenizer import ByteVocabulary
from grammar_constrained_decoder.schema.ir import SchemaIR, parse_schema
from grammar_constrained_decoder.schema.lower import lower_schema

__all__ = [
    "COMPILER_VERSION",
    "CompileObserver",
    "CompileStats",
    "CompiledGrammar",
    "GrammarCompiler",
]

COMPILER_VERSION = "gcd-compiler/1"


class CompileObserver(Protocol):
    def on_compiled(self, seconds: float, *, cache_hit: bool) -> None: ...

    def on_rejected(self, reason: str) -> None: ...


@dataclass(frozen=True, slots=True)
class CompileStats:
    regex_nodes: int
    nfa_states: int
    dfa_states: int
    dfa_classes: int
    index_entries: int
    reachable_states: int
    seconds: float
    cache: str  # "memory", "disk" or "miss"


@dataclass(frozen=True, eq=False)
class CompiledGrammar:
    schema_sha256: str
    cache_key: str
    ir: SchemaIR
    regex: Regex
    dfa: DFA
    index: TokenIndex
    stats: CompileStats

    @property
    def min_tokens(self) -> int:
        """Fewest tokens (EOS excluded) any valid document needs."""
        return int(self.index.distance[self.dfa.start])


def _npz_bytes(arrays: dict[str, npt.NDArray[Any]]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, allow_pickle=False, **arrays)
    return buffer.getvalue()


def _checksum(arrays: dict[str, npt.NDArray[Any]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


class GrammarCompiler:
    """Thread-safe compiler with memory and disk caches."""

    def __init__(
        self,
        vocab: ByteVocabulary,
        *,
        limits: CompileLimits = DEFAULT_LIMITS,
        cache_dir: Path | None = None,
        memory_entries: int = 64,
        observer: CompileObserver | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.vocab = vocab
        self.limits = limits
        self.cache_dir = cache_dir
        self.memory_entries = memory_entries
        self.observer = observer
        self.clock = clock
        self._memory: OrderedDict[str, CompiledGrammar] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def cache_key(self, schema_sha256: str) -> str:
        return sha256_hex(
            "|".join((COMPILER_VERSION, schema_sha256, self.vocab.sha256, self.limits.cache_key()))
        )

    def compile(self, schema: Any) -> CompiledGrammar:
        started = time.perf_counter()
        try:
            compiled, source = self._compile(schema, started)
        except CompileError as exc:
            if self.observer is not None:
                self.observer.on_rejected(exc.reason)
            raise
        if self.observer is not None:
            self.observer.on_compiled(time.perf_counter() - started, cache_hit=source != "miss")
        return compiled

    def _compile(self, schema: Any, started: float) -> tuple[CompiledGrammar, str]:
        """Return the compiled grammar and where it came from ("memory", "disk" or "miss")."""
        budget = CompileBudget(self.limits, self.clock)
        ir = parse_schema(schema, self.limits)
        key = self.cache_key(ir.sha256)
        with self._lock:
            cached = self._memory.get(key)
            if cached is not None:
                self._memory.move_to_end(key)
                self.hits += 1
                return cached, "memory"
        regex = lower_schema(ir, self.limits, budget)
        if regex is EMPTY:
            raise CompileError(
                "the schema admits no instance within the limits", reason="empty_language"
            )
        loaded = self._load(key)
        if loaded is not None:
            dfa, index, stats = loaded
            stats = CompileStats(
                **{**asdict(stats), "cache": "disk", "seconds": time.perf_counter() - started}
            )
            self.hits += 1
        else:
            self.misses += 1
            dfa = compile_regex(regex, budget)
            if dfa.is_empty:
                raise CompileError(
                    "the schema admits no instance within the limits", reason="empty_language"
                )
            index = build_token_index(dfa, self.vocab, budget)
            budget.check_time()
            stats = CompileStats(
                regex_nodes=size(regex),
                nfa_states=budget.nfa_states,
                dfa_states=dfa.n_states,
                dfa_classes=dfa.n_classes,
                index_entries=index.n_entries,
                reachable_states=int(index.reachable.size),
                seconds=time.perf_counter() - started,
                cache="miss",
            )
            self._store(key, ir.sha256, dfa, index, stats)
        compiled = CompiledGrammar(ir.sha256, key, ir, regex, dfa, index, stats)
        with self._lock:
            self._memory[key] = compiled
            self._memory.move_to_end(key)
            while len(self._memory) > self.memory_entries:
                self._memory.popitem(last=False)
        return compiled, stats.cache

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    # -- disk cache ------------------------------------------------------------------------------

    def _path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{key}.npz"

    def _store(
        self, key: str, schema_sha256: str, dfa: DFA, index: TokenIndex, stats: CompileStats
    ) -> None:
        path = self._path(key)
        if path is None:
            return
        arrays: dict[str, npt.NDArray[Any]] = {**dfa.to_arrays(), **index.to_arrays()}
        header = {
            "key": key,
            "version": COMPILER_VERSION,
            "schema_sha256": schema_sha256,
            "vocab_sha256": self.vocab.sha256,
            "stats": asdict(stats),
        }
        arrays["header"] = np.frombuffer(
            json.dumps(header, sort_keys=True).encode("utf-8"), np.uint8
        )
        arrays["checksum"] = np.frombuffer(_checksum(arrays).encode("ascii"), np.uint8)
        atomic_write_bytes(path, _npz_bytes(arrays))

    def _load(self, key: str) -> tuple[DFA, TokenIndex, CompileStats] | None:
        path = self._path(key)
        if path is None or not path.is_file():
            return None
        try:
            with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            checksum = bytes(arrays.pop("checksum")).decode("ascii")
            if checksum != _checksum(arrays):
                return None
            header = json.loads(bytes(arrays.pop("header")).decode("utf-8"))
            if header.get("key") != key or header.get("vocab_sha256") != self.vocab.sha256:
                return None
            dfa = DFA.from_arrays(arrays)
            index = TokenIndex.from_arrays(arrays)
            stats = CompileStats(**header["stats"])
        except (
            OSError,
            EOFError,
            ValueError,
            KeyError,
            TypeError,
            UnicodeDecodeError,
            zlib.error,
            zipfile.BadZipFile,
        ):
            return None
        if index.n_states != dfa.n_states or index.vocab_size != self.vocab.size:
            return None
        return dfa, index, stats
