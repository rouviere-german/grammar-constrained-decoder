"""The Brzozowski-derivative oracle: membership, emptiness, and detection of broken artifacts."""

from __future__ import annotations

import re

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from grammar_constrained_decoder.automata.dfa import DFA, compile_regex
from grammar_constrained_decoder.automata.limits import CompileBudget
from grammar_constrained_decoder.automata.oracle import DerivativeOracle, OracleLimitError
from grammar_constrained_decoder.automata.regex import (
    EMPTY,
    Regex,
    alt,
    cat,
    inter,
    literal,
    rep,
    star,
)
from grammar_constrained_decoder.decoding.index import TokenIndex, build_token_index
from grammar_constrained_decoder.lm.tokenizer import ByteVocabulary
from grammar_constrained_decoder.testing.strategies import intersections, regex_pairs, text_samples

pytestmark = pytest.mark.unit


@given(regex_pairs(), st.lists(text_samples(), max_size=10))
def test_derivative_membership_agrees_with_python_re(
    pair: tuple[Regex, str], words: list[str]
) -> None:
    regex, pattern = pair
    oracle = DerivativeOracle(regex)
    for word in words:
        assert oracle.accepts(word.encode()) == (re.fullmatch(pattern, word) is not None)


@given(intersections())
def test_oracle_proves_the_compiled_dfa_equivalent(pair: tuple[Regex, list[str]]) -> None:
    regex, _ = pair
    report = DerivativeOracle(regex).check_equivalence(compile_regex(regex))
    assert report.ok, report.mismatches
    assert report.pairs_checked >= 1


def test_emptiness_of_intersections_is_decided_semantically() -> None:
    a, b = literal("a"), literal("b")
    disjoint = inter(star(a), cat(b, star(b)))
    oracle = DerivativeOracle(disjoint)
    assert disjoint is not EMPTY  # not syntactically empty ...
    assert not oracle.nonempty(disjoint)  # ... but the language is empty
    assert compile_regex(disjoint).is_empty
    with pytest.raises(KeyError, match="derivative"):
        oracle.nonempty(literal("zzz"))


def test_a_wrong_dfa_is_caught_with_a_witness() -> None:
    a, b = literal("a"), literal("b")
    good = cat(star(alt(a, b)), a)
    dfa = compile_regex(good)
    table = dfa.table.copy()
    accepting = dfa.accepting.copy()
    accepting[dfa.start] = True  # now accepts the empty string too
    broken = DFA(dfa.byte_class, table, accepting, dfa.start, dfa.dead)
    report = DerivativeOracle(good).check_equivalence(broken)
    assert not report.ok
    assert "acceptance differs after bytes ''" in report.mismatches[0]
    other = DerivativeOracle(cat(star(alt(a, b)), b)).check_equivalence(dfa)
    assert not other.ok


def _index(regex: Regex, vocab: ByteVocabulary) -> tuple[DFA, TokenIndex]:
    dfa = compile_regex(regex)
    return dfa, build_token_index(dfa, vocab, CompileBudget())


def test_token_index_check_passes_and_catches_corruption(small_vocab: ByteVocabulary) -> None:
    regex = cat(literal('{"'), rep(alt(literal("ab"), literal("c")), 0, 3), literal('"}'))
    dfa, index = _index(regex, small_vocab)
    oracle = DerivativeOracle(regex)
    equivalence = oracle.check_equivalence(dfa)
    clean = oracle.check_token_index(index, small_vocab.token_bytes, equivalence, eos_id=0)
    assert clean.ok, clean.mismatches
    assert clean.states_checked == len(index.reachable_states)

    tokens = index.tokens.copy()
    tokens[0] = (tokens[0] + 1) % small_vocab.size  # one allowed token replaced by another
    corrupted = TokenIndex(
        index.offsets, np.sort(tokens), index.targets, index.accepting, index.distance,
        index.reachable, index.start, index.eos_id, index.vocab_size,
    )  # fmt: skip
    report = oracle.check_token_index(corrupted, small_vocab.token_bytes, equivalence, eos_id=0)
    assert not report.ok

    targets = index.targets.copy()
    targets[:] = index.start  # every successor wrong
    wrong_next = TokenIndex(
        index.offsets, index.tokens, targets, index.accepting, index.distance,
        index.reachable, index.start, index.eos_id, index.vocab_size,
    )  # fmt: skip
    assert not oracle.check_token_index(
        wrong_next, small_vocab.token_bytes, equivalence, eos_id=0
    ).ok

    accepting = index.accepting.copy()
    accepting[index.start] = True
    wrong_eos = TokenIndex(
        index.offsets, index.tokens, index.targets, accepting, index.distance,
        index.reachable, index.start, index.eos_id, index.vocab_size,
    )  # fmt: skip
    assert any(
        "EOS" in problem
        for problem in oracle.check_token_index(
            wrong_eos, small_vocab.token_bytes, equivalence, eos_id=0
        ).mismatches
    )


def test_oracle_state_cap() -> None:
    ab = alt(literal("a"), literal("b"))
    with pytest.raises(OracleLimitError):
        DerivativeOracle(cat(star(ab), literal("a"), rep(ab, 10, 10)), max_states=100)
