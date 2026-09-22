"""Token-transition index: exactness against a brute-force walk, multi-byte tokens across grammar
symbols, EOS handling, budget-aware masks and the dead-end check."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import make_vocab
from grammar_constrained_decoder.automata.charset import ALL_BYTES
from grammar_constrained_decoder.automata.dfa import DFA, compile_regex
from grammar_constrained_decoder.automata.limits import (
    CompileBudget,
    CompileLimitError,
    CompileLimits,
)
from grammar_constrained_decoder.automata.regex import Regex, alt, byteset, cat, literal, rep, star
from grammar_constrained_decoder.decoding.index import DeadEndError, TokenIndex, build_token_index
from grammar_constrained_decoder.lm.tokenizer import ByteVocabulary
from grammar_constrained_decoder.schema.ir import parse_schema
from grammar_constrained_decoder.schema.lower import lower_schema
from grammar_constrained_decoder.testing.strategies import json_schemas, regex_pairs

pytestmark = pytest.mark.unit


def build(regex: Regex, vocab: ByteVocabulary) -> tuple[DFA, TokenIndex]:
    dfa = compile_regex(regex)
    return dfa, build_token_index(dfa, vocab, CompileBudget())


def brute_force(dfa: DFA, vocab: ByteVocabulary, state: int) -> dict[int, int]:
    allowed = {}
    for token, data in enumerate(vocab.token_bytes):
        if token in vocab.special_ids:
            continue
        target = dfa.run(data, state)
        if target != dfa.dead:
            allowed[token] = target
    return allowed


@settings(max_examples=40)
@given(json_schemas(max_leaves=3))
def test_index_equals_a_brute_force_walk_of_every_token(schema: dict[str, object]) -> None:
    vocab = make_vocab()
    dfa, index = build(lower_schema(parse_schema(schema)), vocab)
    for state in range(dfa.n_states):
        if state == dfa.dead:
            assert index.allowed_tokens(state).size == 0
            continue
        expected = brute_force(dfa, vocab, state)
        assert index.allowed_tokens(state).tolist() == sorted(expected)
        for token, target in expected.items():
            assert index.next_state(state, token) == target
        assert bool(index.mask(state)[0]) == bool(dfa.accepting[state])


@given(regex_pairs())
def test_distances_are_shortest_token_paths(pair: tuple[Regex, str]) -> None:
    vocab = make_vocab([b"ab", b"abc", b"cc"])
    dfa, index = build(pair[0], vocab)
    if dfa.is_empty:
        return
    for state in index.reachable_states:
        seen = {state: 0}
        queue = deque([state])
        best = -1
        while queue:
            current = queue.popleft()
            if dfa.accepting[current]:
                best = seen[current]
                break
            for token in index.allowed_tokens(current).tolist():
                nxt = index.next_state(current, token)
                if nxt not in seen:
                    seen[nxt] = seen[current] + 1
                    queue.append(nxt)
        assert int(index.distance[state]) == best


def test_multi_byte_tokens_are_allowed_only_when_every_byte_fits() -> None:
    vocab = make_vocab()
    close2 = vocab.token_bytes.index(b"}}")
    close_arr = vocab.token_bytes.index(b"]}")
    nested = parse_schema(
        {
            "type": "object",
            "properties": {"a": {"type": "object", "properties": {"b": {"type": "null"}}}},
        }
    )
    dfa, index = build(lower_schema(nested), vocab)
    after_inner = dfa.run(b'{"a":{"b":null')
    assert index.next_state(after_inner, close2) == dfa.run(b'{"a":{"b":null}}')
    assert index.next_state(after_inner, close_arr) == -1
    after_outer_value = dfa.run(b'{"a":{}')
    assert index.next_state(after_outer_value, close2) == -1  # would close one brace too many
    assert close2 not in index.allowed_tokens(dfa.start).tolist()


def test_eos_only_in_accepting_states_and_budget_masks() -> None:
    vocab = make_vocab([])
    regex = cat(literal("ab"), rep(literal("c"), 0, 3))
    dfa, index = build(regex, vocab)
    start_mask = index.mask(dfa.start)
    assert not start_mask[0]
    done = dfa.run(b"ab")
    assert index.mask(done)[0]
    c = vocab.token_bytes.index(b"c")
    assert index.mask(done)[c]
    assert not index.mask(done, remaining=0)[c]  # no budget left: only EOS
    assert index.mask(done, remaining=0)[0]
    assert index.mask(dfa.start, remaining=2)[vocab.token_bytes.index(b"a")]
    mask = index.mask(dfa.start)
    with pytest.raises(ValueError, match="read-only"):
        mask[5] = True


def test_dead_ends_are_detected_at_compile_time() -> None:
    # Language x(y|z) with tokens "x" and "xy": after the token "x" no token can continue.
    vocab = ByteVocabulary.from_tokens([b"", b"x", b"xy"], eos_id=0)
    dfa = compile_regex(cat(literal("x"), alt(literal("y"), literal("z"))))
    with pytest.raises(DeadEndError) as info:
        build_token_index(dfa, vocab, CompileBudget())
    assert info.value.reason == "dead_end"
    fine = ByteVocabulary.from_tokens([b"", b"x", b"y", b"z"], eos_id=0)
    index = build_token_index(dfa, fine, CompileBudget())
    assert int(index.distance[dfa.start]) == 2


def test_unreachable_states_are_not_required_to_finish() -> None:
    # "xz" is in the language but "x" alone is never a token boundary here: only "xz" exists.
    vocab = ByteVocabulary.from_tokens([b"", b"xz", b"xy"], eos_id=0)
    dfa, index = build(alt(literal("xz"), literal("xy")), vocab)
    assert dfa.run(b"x") not in index.reachable_states
    assert sorted(index.allowed_tokens(dfa.start).tolist()) == [1, 2]


def test_index_entry_cap() -> None:
    budget = CompileBudget(CompileLimits(max_index_entries=10))
    dfa = compile_regex(star(byteset(ALL_BYTES)))  # every token is allowed everywhere
    with pytest.raises(CompileLimitError) as info:
        build_token_index(dfa, make_vocab(), budget)
    assert info.value.reason == "index_entries"


def test_arrays_round_trip_and_are_validated() -> None:
    vocab = make_vocab()
    _, index = build(
        lower_schema(parse_schema({"type": "array", "items": {"type": "boolean"}})), vocab
    )
    restored = TokenIndex.from_arrays(index.to_arrays())
    assert np.array_equal(restored.tokens, index.tokens)
    arrays = index.to_arrays()
    arrays["index_targets"] = arrays["index_targets"] + 10_000
    with pytest.raises(ValueError, match="invalid token index"):
        TokenIndex.from_arrays(arrays)


@given(st.integers(0, 40))
def test_budget_mask_never_offers_a_token_that_cannot_finish(remaining: int) -> None:
    vocab = make_vocab()
    dfa, index = build(
        lower_schema(parse_schema({"type": "array", "items": {"type": "integer"}, "maxItems": 5})),
        vocab,
    )
    for state in index.reachable_states:
        if int(index.distance[state]) > remaining:
            continue
        mask = index.mask(state, remaining)
        for token in np.flatnonzero(mask).tolist():
            if token == 0:
                assert dfa.accepting[state]
            else:
                assert int(index.distance[index.next_state(state, token)]) <= remaining - 1
