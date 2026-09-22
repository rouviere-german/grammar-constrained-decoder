"""Grammar compiler: caches (memory, disk, corrupted entries), limits and observer hooks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import make_vocab
from grammar_constrained_decoder.automata.limits import (
    CompileError,
    CompileLimitError,
    CompileLimits,
    SchemaError,
)
from grammar_constrained_decoder.decoding.compiler import GrammarCompiler

pytestmark = pytest.mark.unit

VOCAB = make_vocab()
SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}


class Recorder:
    def __init__(self) -> None:
        self.compiled: list[bool] = []
        self.rejected: list[str] = []

    def on_compiled(self, seconds: float, *, cache_hit: bool) -> None:
        assert seconds >= 0
        self.compiled.append(cache_hit)

    def on_rejected(self, reason: str) -> None:
        self.rejected.append(reason)


def test_memory_and_disk_caches(tmp_path: Path) -> None:
    recorder = Recorder()
    first = GrammarCompiler(VOCAB, cache_dir=tmp_path, observer=recorder)
    miss = first.compile(SCHEMA)
    hit = first.compile(dict(reversed(list(SCHEMA.items()))))  # same canonical schema
    assert miss.stats.cache == "miss"
    assert hit is miss
    assert recorder.compiled == [False, True]
    assert first.hit_ratio == 0.5
    second = GrammarCompiler(VOCAB, cache_dir=tmp_path)
    from_disk = second.compile(SCHEMA)
    assert from_disk.stats.cache == "disk"
    assert from_disk.dfa.signature() == miss.dfa.signature()
    assert np.array_equal(from_disk.index.tokens, miss.index.tokens)


def test_corrupted_or_foreign_cache_entries_are_recompiled(tmp_path: Path) -> None:
    compiler = GrammarCompiler(VOCAB, cache_dir=tmp_path)
    compiled = compiler.compile(SCHEMA)
    path = tmp_path / f"{compiled.cache_key}.npz"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    rebuilt = GrammarCompiler(VOCAB, cache_dir=tmp_path).compile(SCHEMA)
    assert rebuilt.stats.cache == "miss"
    other_vocab = make_vocab([b"zz"])
    foreign = GrammarCompiler(other_vocab, cache_dir=tmp_path)
    assert foreign.cache_key(compiled.schema_sha256) != compiled.cache_key
    path.write_bytes(b"not an npz")
    assert GrammarCompiler(VOCAB, cache_dir=tmp_path).compile(SCHEMA).stats.cache == "miss"


def test_rejections_are_reported_with_reasons() -> None:
    recorder = Recorder()
    compiler = GrammarCompiler(VOCAB, observer=recorder)
    with pytest.raises(SchemaError):
        compiler.compile({"type": "string", "format": "email"})
    with pytest.raises(CompileError) as info:
        compiler.compile({"type": "integer", "minimum": 5, "maximum": 1})
    assert info.value.reason == "empty_language"
    with pytest.raises(CompileError) as syntactic:
        compiler.compile({"type": "object", "properties": {"x": {"$ref": "#"}}, "required": ["x"]})
    assert syntactic.value.reason == "empty_language"
    assert recorder.rejected == ["unsupported", "empty_language", "empty_language"]


def test_limits_are_part_of_the_cache_key() -> None:
    loose = GrammarCompiler(VOCAB)
    tight = GrammarCompiler(VOCAB, limits=CompileLimits(max_dfa_states=50))
    key = "0" * 64
    assert loose.cache_key(key) != tight.cache_key(key)
    with pytest.raises(CompileLimitError) as info:
        tight.compile({"type": "string", "maxLength": 40})
    assert info.value.reason == "dfa_states"


@pytest.mark.parametrize(
    ("schema", "reasons"),
    [
        ({"type": "string", "pattern": "^(a|b)*a(a|b){14}$"}, {"dfa_states"}),
        (
            {"type": "string", "pattern": "^((a{0,50}){0,50}){0,50}$"},
            {"nfa_states", "subset_size", "timeout"},
        ),
    ],
)
def test_state_explosions_are_rejected_within_the_timeout(
    schema: dict[str, Any], reasons: set[str]
) -> None:
    limits = CompileLimits(max_dfa_states=5_000, max_nfa_states=50_000, timeout_seconds=2.0)
    compiler = GrammarCompiler(VOCAB, limits=limits)
    started = time.perf_counter()
    with pytest.raises(CompileLimitError) as info:
        compiler.compile(schema)
    assert info.value.reason in reasons
    assert time.perf_counter() - started < limits.timeout_seconds + 0.5


@pytest.mark.parametrize(
    "pattern",
    ["^(a|b)*a(a|b){20}$", "^((a{0,90}){0,90}){0,90}$", "^(((a|b){0,60}){0,60}){0,60}$"],
)
def test_timeouts_fire_within_one_check_interval(pattern: str) -> None:
    """With every other cap lifted, the deadline alone stops these; the cooperative checks run
    at every new DFA state and every few dozen cheaper steps, so the overshoot stays small."""
    limits = CompileLimits(
        timeout_seconds=0.5, max_dfa_states=10**6, max_nfa_states=10**7, max_subset_entries=10**9
    )
    started = time.perf_counter()
    with pytest.raises(CompileLimitError) as info:
        GrammarCompiler(VOCAB, limits=limits).compile({"type": "string", "pattern": pattern})
    elapsed = time.perf_counter() - started
    assert info.value.reason == "timeout"
    assert elapsed < limits.timeout_seconds + 0.25, elapsed
