"""Masked decoding: every constrained output is valid, whatever the model wants, within budget."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from jsonschema import Draft202012Validator

from conftest import AdversarialLM, RandomLM, UniformLM, make_vocab
from grammar_constrained_decoder.automata.limits import CompileLimits
from grammar_constrained_decoder.decoding.compiler import GrammarCompiler
from grammar_constrained_decoder.decoding.decoder import (
    Decoder,
    GrammarCursor,
    GrammarViolationError,
    LanguageModel,
    _nucleus,
)
from grammar_constrained_decoder.testing.strategies import json_schemas

pytestmark = pytest.mark.unit

VOCAB = make_vocab()
COMPILER = GrammarCompiler(VOCAB, limits=CompileLimits(timeout_seconds=30))
OBJECT = {
    "type": "object",
    "properties": {
        "id": {"type": "integer", "minimum": 1, "maximum": 999},
        "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "maxItems": 3},
        "ok": {"type": "boolean"},
    },
    "required": ["id", "ok"],
    "additionalProperties": False,
}


def valid(schema: dict[str, Any], text: str) -> bool:
    try:
        return Draft202012Validator(schema).is_valid(json.loads(text))
    except json.JSONDecodeError:
        return False


@settings(max_examples=30, suppress_health_check=list(HealthCheck))
@given(
    json_schemas(max_leaves=3),
    st.sampled_from(["greedy", "top_p", "beam"]),
    st.integers(0, 10_000),
    st.sampled_from(["random", "adversarial", "uniform"]),
)
def test_constrained_outputs_are_always_valid(
    schema: dict[str, Any], strategy: str, seed: int, model: str
) -> None:
    models: dict[str, LanguageModel] = {
        "random": RandomLM(VOCAB.size, seed),
        "adversarial": AdversarialLM(VOCAB),
        "uniform": UniformLM(VOCAB.size),
    }
    lm = models[model]
    grammar = COMPILER.compile(schema)
    budget = grammar.min_tokens + 12
    out = Decoder(lm, VOCAB).generate(
        [1, 2, 3], grammar=grammar.index, max_tokens=budget, strategy=strategy,  # type: ignore[arg-type]
        seed=seed, beam_width=3,
    )  # fmt: skip
    assert out.stop_reason == "eos"
    assert len(out.token_ids) <= budget
    assert valid(schema, out.text), (schema, out.text)
    assert grammar.dfa.accepts(out.text.encode("utf-8"))


def test_the_tightest_budget_still_finishes_validly() -> None:
    grammar = COMPILER.compile(OBJECT)
    decoder = Decoder(AdversarialLM(VOCAB), VOCAB)
    out = decoder.generate([], grammar=grammar.index, max_tokens=grammar.min_tokens)
    assert valid(OBJECT, out.text)
    assert len(out.token_ids) == grammar.min_tokens
    with pytest.raises(ValueError, match="too small"):
        decoder.generate([], grammar=grammar.index, max_tokens=grammar.min_tokens - 1)


def test_unconstrained_decoding_stops_at_the_budget() -> None:
    decoder = Decoder(AdversarialLM(VOCAB), VOCAB)
    out = decoder.generate([], grammar=None, max_tokens=5)
    assert out.stop_reason == "max_tokens"
    assert len(out.token_ids) == 5
    assert not valid(OBJECT, out.text)
    assert out.mask_seconds == []


def test_beam_search_scores_at_least_as_well_as_greedy() -> None:
    grammar = COMPILER.compile({"enum": ["aa", "ab", "ba", "bb", "abc"]})
    lm = RandomLM(VOCAB.size, 7, sharpness=2.0)
    decoder = Decoder(lm, VOCAB)
    greedy = decoder.generate([4], grammar=grammar.index, max_tokens=20)
    beam = decoder.generate(
        [4], grammar=grammar.index, max_tokens=20, strategy="beam", beam_width=8
    )
    assert beam.logprob >= greedy.logprob - 1e-9
    assert json.loads(beam.text) in ["aa", "ab", "ba", "bb", "abc"]
    free = decoder.beam_search([4], grammar=None, max_tokens=3, beam_width=2)
    assert free.stop_reason in ("eos", "max_tokens")
    with pytest.raises(ValueError, match="beam_width"):
        decoder.beam_search([4], grammar=None, max_tokens=3, beam_width=0)
    with pytest.raises(ValueError, match="max_tokens"):
        decoder.beam_search([4], grammar=None, max_tokens=0)


def test_stream_yields_tokens_then_eos_and_matches_generate() -> None:
    grammar = COMPILER.compile(OBJECT)
    decoder = Decoder(RandomLM(VOCAB.size, 3), VOCAB)
    events = list(
        decoder.stream([9], grammar=grammar.index, max_tokens=60, seed=5, strategy="top_p")
    )
    assert events[-1].eos
    assert all(not event.eos for event in events[:-1])
    text = "".join(event.text for event in events)
    again = decoder.generate([9], grammar=grammar.index, max_tokens=60, seed=5, strategy="top_p")
    assert text == again.text
    assert again.logprob == pytest.approx(sum(event.logprob for event in events))
    assert again.tokens_per_second > 0


def test_invalid_decoder_arguments() -> None:
    decoder = Decoder(UniformLM(VOCAB.size), VOCAB)
    with pytest.raises(ValueError, match="max_tokens"):
        list(decoder.stream([], grammar=None, max_tokens=0))
    with pytest.raises(ValueError, match="top_p"):
        list(decoder.stream([], grammar=None, max_tokens=3, top_p=0.0))
    with pytest.raises(ValueError, match="vocabularies"):
        Decoder(UniformLM(7), VOCAB)


def test_cursor_refuses_tokens_outside_the_grammar() -> None:
    grammar = COMPILER.compile({"const": True})
    cursor = GrammarCursor(grammar.index, 10)
    with pytest.raises(GrammarViolationError, match="non-accepting"):
        cursor.advance(VOCAB.eos_id)
    with pytest.raises(GrammarViolationError, match="not allowed"):
        cursor.advance(VOCAB.token_bytes.index(b"x"))
    cursor.advance(VOCAB.token_bytes.index(b"true"))
    clone = cursor.copy()
    cursor.advance(VOCAB.eos_id)
    with pytest.raises(GrammarViolationError, match="already finished"):
        cursor.advance(VOCAB.eos_id)
    assert not clone.finished


def test_nucleus_sampling_keeps_the_smallest_top_set() -> None:
    logp = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    rng = np.random.default_rng(0)
    draws = {_nucleus(logp, temperature=1.0, top_p=0.7, rng=rng) for _ in range(300)}
    assert draws == {0, 1}
    masked = np.array([-np.inf, np.log(0.2), -np.inf, np.log(0.8)])
    assert {_nucleus(masked, temperature=0.5, top_p=1.0, rng=rng) for _ in range(200)} <= {1, 3}
