"""Exact distortion measurement on enumerable languages."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import RandomLM, Trained, UniformLM, make_vocab
from grammar_constrained_decoder.decoding.compiler import GrammarCompiler
from grammar_constrained_decoder.decoding.distortion import (
    LanguageTooLargeError,
    distortion_report,
    enumerate_language,
)
from grammar_constrained_decoder.training.corpus import load_distortion_schemas, prompt_for

pytestmark = pytest.mark.unit

VOCAB = make_vocab()
COMPILER = GrammarCompiler(VOCAB)


def test_enumeration_finds_every_tokenization_and_q_is_normalized() -> None:
    grammar = COMPILER.compile({"enum": ["ab", "abc"]})
    language = enumerate_language(UniformLM(VOCAB.size), grammar.index, VOCAB, [])
    assert sorted(set(language.strings)) == ['"ab"', '"abc"']
    # '"ab"': quote, then a|b|"ab"... every split of the string into vocabulary tokens.
    assert len(language.sequences) == len(set(language.sequences))
    assert float(np.exp(language.log_q).sum()) == pytest.approx(1.0)
    assert float(np.exp(language.log_p).sum()) < 1.0


def test_uniform_model_on_a_single_path_language_has_no_distortion() -> None:
    vocab = make_vocab([], singles=True)
    compiler = GrammarCompiler(vocab)
    grammar = compiler.compile({"const": None})  # exactly one tokenization with single bytes
    report = distortion_report(UniformLM(vocab.size), grammar.index, vocab, [], samples=20)
    assert report.n_sequences == 1
    assert report.kl_token == pytest.approx(0.0, abs=1e-12)
    assert report.kl_string == pytest.approx(0.0, abs=1e-12)
    assert report.greedy_is_conditional_mode
    assert report.greedy_conditional_probability == pytest.approx(1.0)
    assert report.sampled_tv_to_masked == pytest.approx(0.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_kl_is_non_negative_and_marginalizing_tokenizations_cannot_increase_it(seed: int) -> None:
    grammar = COMPILER.compile({"type": "array", "items": {"type": "boolean"}, "maxItems": 2})
    report = distortion_report(RandomLM(VOCAB.size, seed), grammar.index, VOCAB, [7], samples=0)
    assert report.kl_token >= -1e-12
    assert report.kl_string >= -1e-12
    assert report.kl_string <= report.kl_token + 1e-9  # data-processing inequality
    assert 0 < report.valid_mass <= 1
    assert report.n_strings == 7


def test_real_model_distortion_on_tiny_schemas(trained: Trained) -> None:
    model = trained.model
    compiler = GrammarCompiler(model.tokenizer.vocab)
    tiny = {item.id: item for item in load_distortion_schemas()}
    assert len(tiny) == 20
    for name in ("bool", "digit"):
        grammar = compiler.compile(tiny[name].schema)
        prompt = model.tokenizer.encode(prompt_for(tiny[name]))
        report = distortion_report(
            model.lm, grammar.index, model.tokenizer.vocab, prompt, samples=50
        )
        assert math.isfinite(report.kl_string)
        assert report.n_strings == (2 if name == "bool" else 10)
        assert report.as_dict()["n_strings"] == report.n_strings


def test_enumeration_cap() -> None:
    grammar = COMPILER.compile({"type": "string", "maxLength": 4, "pattern": "^[ab]*$"})
    with pytest.raises(LanguageTooLargeError):
        enumerate_language(UniformLM(VOCAB.size), grammar.index, VOCAB, [], max_sequences=10)
