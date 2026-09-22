"""Thompson NFA -> subset construction -> Hopcroft minimization, checked against Python re and an
independent Moore-style minimization."""

from __future__ import annotations

import re
import time

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from grammar_constrained_decoder.automata.charset import byte_range
from grammar_constrained_decoder.automata.dfa import (
    DFA,
    byte_partition,
    compile_regex,
    determinize,
    intersect,
    minimize,
)
from grammar_constrained_decoder.automata.limits import (
    CompileBudget,
    CompileLimitError,
    CompileLimits,
)
from grammar_constrained_decoder.automata.nfa import thompson
from grammar_constrained_decoder.automata.pattern import compile_pattern
from grammar_constrained_decoder.automata.regex import (
    EMPTY,
    EPS,
    Regex,
    alt,
    byteset,
    cat,
    literal,
    plus,
    rep,
    star,
)
from grammar_constrained_decoder.testing.strategies import intersections, regex_pairs, text_samples

pytestmark = pytest.mark.unit


@given(regex_pairs(), st.lists(text_samples(), max_size=12))
def test_dfa_membership_agrees_with_python_re(pair: tuple[Regex, str], words: list[str]) -> None:
    regex, pattern = pair
    dfa = compile_regex(regex)
    compiled = re.compile(pattern)
    for word in words:
        assert dfa.accepts(word.encode()) == (compiled.fullmatch(word) is not None), (pattern, word)


@given(intersections(), st.lists(text_samples(), max_size=12))
def test_intersection_accepts_exactly_the_common_strings(
    pair: tuple[Regex, list[str]], words: list[str]
) -> None:
    regex, patterns = pair
    dfa = compile_regex(regex)
    for word in words:
        expected = all(re.fullmatch(pattern, word) for pattern in patterns)
        assert dfa.accepts(word.encode()) == expected


def _moore_state_count(dfa: DFA) -> int:
    """Independent minimization (Moore's refinement by signatures) for cross-checking Hopcroft."""
    block = dfa.accepting.astype(np.int64)
    while True:
        signature = [
            (int(block[s]), *(int(block[t]) for t in dfa.table[s].tolist()))
            for s in range(dfa.n_states)
        ]
        index: dict[tuple[int, ...], int] = {}
        refined = np.array([index.setdefault(sig, len(index)) for sig in signature])
        if len(index) == len(set(block.tolist())):
            return len(index)
        block = refined


@given(regex_pairs())
def test_hopcroft_result_is_minimal(pair: tuple[Regex, str]) -> None:
    budget = CompileBudget()
    raw = determinize(thompson(pair[0], budget, compile_regex), budget)
    minimal = minimize(raw, budget)
    assert minimal.n_states == _moore_state_count(raw)
    assert _moore_state_count(minimal) == minimal.n_states


@given(regex_pairs(), regex_pairs())
def test_canonical_form_identifies_languages(a: tuple[Regex, str], b: tuple[Regex, str]) -> None:
    left, right = compile_regex(a[0]), compile_regex(b[0])
    same_signature = left.signature() == right.signature()
    union_ab = compile_regex(alt(a[0], b[0])).signature()
    union_ba = compile_regex(alt(b[0], a[0])).signature()
    assert union_ab == union_ba
    if same_signature:
        for word in ["", "a", "b", "c", "ab", "ba", "abc", "cab", "aaa", "bbb"]:
            assert left.accepts(word.encode()) == right.accepts(word.encode())


def test_equivalent_expressions_compile_to_identical_dfas() -> None:
    a, b = literal("a"), literal("b")
    pairs = [
        (cat(a, star(a)), cat(star(a), a)),
        (star(alt(a, b)), star(cat(star(a), star(b)))),
        (rep(a, 2, 4), cat(a, a, rep(a, 0, 2))),
        (plus(alt(a, b)), cat(alt(a, b), star(alt(b, a)))),
    ]
    for left, right in pairs:
        assert compile_regex(left).signature() == compile_regex(right).signature()


def test_exponential_blowup_family_has_exactly_2_to_the_n_live_states() -> None:
    a, b = literal("a"), literal("b")
    ab = alt(a, b)
    for n in (3, 5, 8):
        dfa = compile_regex(cat(star(ab), a, rep(ab, n, n)))
        assert dfa.n_live_states == 2 ** (n + 1)


def test_empty_language_and_dead_state() -> None:
    empty = compile_regex(EMPTY)
    assert empty.is_empty
    assert not empty.accepts(b"")
    anything = compile_regex(star(byteset(byte_range(0, 255))))
    assert anything.dead == -1
    assert anything.accepts(b"\x00\xff")
    eps = compile_regex(EPS)
    assert eps.accepts(b"")
    assert not eps.accepts(b"a")
    assert eps.run(b"ab") == eps.dead


def test_distances_to_accept_are_shortest_paths() -> None:
    dfa = compile_regex(cat(literal("ab"), rep(literal("c"), 2, 5)))
    distance = dfa.distances_to_accept()
    assert int(distance[dfa.start]) == 4
    assert int(distance[dfa.run(b"abcc")]) == 0
    assert int(distance[dfa.dead]) == -1


def test_byte_partition_is_coarsest_refinement() -> None:
    byte_class, blocks = byte_partition([byte_range(0x30, 0x39), byte_range(0x35, 0x40), 0])
    assert len(blocks) == 4
    assert byte_class[0x30] == byte_class[0x34]
    assert byte_class[0x35] != byte_class[0x34]
    assert byte_class[0x3A] == byte_class[0x40]
    assert sum(bin(block).count("1") for block in blocks) == 256


def test_arrays_round_trip_and_malformed_arrays_are_rejected() -> None:
    dfa = compile_regex(cat(literal("x"), star(literal("y"))))
    restored = DFA.from_arrays(dfa.to_arrays())
    assert restored.signature() == dfa.signature()
    arrays = dfa.to_arrays()
    broken_table = dict(arrays)
    broken_table["dfa_table"] = arrays["dfa_table"] + 100
    broken_dead = dict(arrays)
    broken_dead["dfa_meta"] = np.array([0, 0], dtype=np.int32)
    broken_shape = dict(arrays)
    broken_shape["dfa_byte_class"] = arrays["dfa_byte_class"][:10]
    for arrays_bad in (broken_table, broken_dead, broken_shape):
        with pytest.raises(ValueError, match="invalid DFA"):
            DFA.from_arrays(arrays_bad)


def test_intersection_with_an_empty_operand_is_empty() -> None:
    budget = CompileBudget()
    left = compile_regex(literal("a"))
    right = compile_regex(EMPTY)
    assert intersect(left, right, budget).is_empty
    assert intersect(right, left, budget).is_empty
    both = intersect(
        compile_regex(star(literal("a"))), compile_regex(rep(literal("a"), 2, 3)), budget
    )
    assert [both.accepts(b"a" * n) for n in range(5)] == [False, False, True, True, False]


def test_state_caps_stop_the_subset_construction() -> None:
    a, b = literal("a"), literal("b")
    ab = alt(a, b)
    limits = CompileLimits(max_dfa_states=500)
    with pytest.raises(CompileLimitError) as info:
        compile_regex(cat(star(ab), a, rep(ab, 12, 12)), CompileBudget(limits))
    assert info.value.reason == "dfa_states"
    with pytest.raises(CompileLimitError) as nfa_info:
        compile_regex(
            rep(rep(literal("abc"), 0, 100), 0, 100),
            CompileBudget(CompileLimits(max_nfa_states=5_000)),
        )
    assert nfa_info.value.reason == "nfa_states"


def test_cooperative_deadline_aborts_compilation() -> None:
    ticks = iter(range(10**9))
    budget = CompileBudget(CompileLimits(timeout_seconds=50), clock=lambda: float(next(ticks)))
    a, b = literal("a"), literal("b")
    with pytest.raises(CompileLimitError) as info:
        compile_regex(cat(star(alt(a, b)), a, rep(alt(a, b), 10, 10)), budget)
    assert info.value.reason == "timeout"


def test_matching_is_linear_on_classic_redos_patterns() -> None:
    dfa = compile_regex(compile_pattern("^(a+)+$"))
    evil = b"a" * 30_000 + b"!"
    started = time.perf_counter()
    assert not dfa.accepts(evil)
    assert dfa.accepts(b"a" * 30_000)
    assert time.perf_counter() - started < 1.0
    nested = compile_regex(compile_pattern("^(a|aa)*b$"))
    assert not nested.accepts(b"a" * 20_000)
