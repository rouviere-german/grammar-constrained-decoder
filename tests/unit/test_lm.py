"""Byte-level BPE tokenizer and the interpolated Kneser-Ney model (checked against a naive,
independent implementation of the same formulas)."""

from __future__ import annotations

import io
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tokenizers import pre_tokenizers

from conftest import Trained
from grammar_constrained_decoder.lm.kneser_ney import KneserNeyLM, LMFormatError
from grammar_constrained_decoder.lm.tokenizer import (
    EOS_TOKEN,
    BPETokenizer,
    ByteVocabulary,
    TokenizerIntegrityError,
    bytes_to_unicode,
)

pytestmark = pytest.mark.unit


def test_byte_mapping_is_a_bijection_onto_the_library_alphabet() -> None:
    table = bytes_to_unicode()
    assert sorted(table) == list(range(256))
    assert len(set(table.values())) == 256
    assert set(table.values()) == set(pre_tokenizers.ByteLevel.alphabet())


@settings(max_examples=100)
@given(st.text(max_size=60))
def test_token_bytes_reassemble_the_text_exactly(trained: Trained, text: str) -> None:
    tokenizer = trained.model.tokenizer
    ids = tokenizer.encode(text)
    assert tokenizer.decode_bytes(ids) == text.encode("utf-8", errors="surrogatepass")
    assert tokenizer.vocab.eos_id not in ids


def test_literal_eos_markers_in_text_are_not_special(trained: Trained) -> None:
    tokenizer = trained.model.tokenizer
    ids = tokenizer.encode(f"a{EOS_TOKEN}b{EOS_TOKEN}")
    assert tokenizer.vocab.eos_id not in ids
    assert tokenizer.decode(ids) == f"a{EOS_TOKEN}b{EOS_TOKEN}"


def test_vocabulary_has_every_byte_and_multi_byte_json_tokens(trained: Trained) -> None:
    vocab = trained.model.tokenizer.vocab
    assert vocab.size == 1024
    singles = {data for data in vocab.token_bytes if len(data) == 1}
    assert len(singles) == 256
    merged = set(vocab.token_bytes)
    assert {b'":', b'{"', b'","'} <= merged
    assert vocab.max_token_length > 3
    assert vocab.token_bytes[vocab.eos_id] == b""
    assert vocab.decode([vocab.token_bytes.index(b'{"')]) == b'{"'


def test_training_is_deterministic() -> None:
    texts = ['{"a":1,"b":[true,false]}'] * 50 + ["plain words here"] * 20
    first = BPETokenizer.train(texts, vocab_size=300)
    second = BPETokenizer.train(texts, vocab_size=300)
    assert first.sha256 == second.sha256
    assert first.vocab.sha256 == second.vocab.sha256
    with pytest.raises(ValueError, match="vocab_size"):
        BPETokenizer.train(texts, vocab_size=100)


def test_tokenizer_json_is_verified_on_load(trained: Trained) -> None:
    tokenizer = trained.model.tokenizer
    text = tokenizer.to_json()
    restored = BPETokenizer.from_json(text, expected_sha256=tokenizer.sha256)
    assert restored.vocab == tokenizer.vocab
    with pytest.raises(TokenizerIntegrityError, match="SHA-256"):
        BPETokenizer.from_json(text, expected_sha256="0" * 64)
    with pytest.raises(TokenizerIntegrityError, match="not JSON"):
        BPETokenizer.from_json("{nope")
    spec = json.loads(text)
    spec["model"]["type"] = "WordPiece"
    with pytest.raises(TokenizerIntegrityError, match="byte-level BPE"):
        BPETokenizer.from_json(json.dumps(spec))


def test_vocabulary_validation() -> None:
    with pytest.raises(ValueError, match="at least one byte"):
        ByteVocabulary.from_tokens([b"", b""], eos_id=0)
    vocab = ByteVocabulary.from_tokens([b"x", b"", b"yz"], eos_id=1)
    assert vocab.special_ids == frozenset({1})
    assert vocab.max_token_length == 2
    assert vocab.decode([0, 2]) == b"xyz"


# -- Kneser-Ney ----------------------------------------------------------------------------------


def naive_kn_prob(
    documents: Sequence[Sequence[int]],
    order: int,
    *,
    vocab: int,
    bos: int,
    context: Sequence[int],
    token: int,
) -> float:
    """Straight dictionary implementation of interpolated modified Kneser-Ney (Chen and Goodman)."""
    pad = order - 1
    grams: dict[int, Counter[tuple[int, ...]]] = defaultdict(Counter)
    for doc in documents:
        seq = [bos] * pad + list(doc)
        for position in range(pad, len(seq)):
            for n in range(1, order + 1):
                grams[n][tuple(seq[position - n + 1 : position + 1])] += 1
    counts: dict[int, Counter[tuple[int, ...]]] = {order: grams[order]}
    for n in range(order - 1, 0, -1):
        continuation: Counter[tuple[int, ...]] = Counter()
        for gram in grams[n + 1]:
            continuation[gram[1:]] += 1
        counts[n] = continuation

    def discounts(values: list[int]) -> tuple[float, float, float]:
        n = [sum(1 for v in values if v == k) for k in (1, 2, 3, 4)]
        if n[0] == 0 or n[1] == 0:
            return 0.5, 1.0, 1.5
        y = n[0] / (n[0] + 2 * n[1])
        d = (1 - 2 * y * n[1] / n[0], 2 - 3 * y * n[2] / n[1] if n[1] else 1.0,
             3 - 4 * y * n[3] / n[2] if n[2] else 1.5)  # fmt: skip
        floor = 0.1  # the model floors every discount so no context can give a zero probability
        return (min(max(d[0], floor), 1), min(max(d[1], floor), 2), min(max(d[2], floor), 3))

    def discount(value: int, d: tuple[float, float, float]) -> float:
        return d[0] if value == 1 else d[1] if value == 2 else d[2]

    uni = counts[1]
    d1 = discounts(list(uni.values()))
    total = sum(uni.values())
    gamma0 = sum(discount(c, d1) for c in uni.values()) / total
    prob = (
        max(uni.get((token,), 0) - discount(uni.get((token,), 1), d1), 0) / total + gamma0 / vocab
    )
    history = ([bos] * pad + list(context))[-pad:]
    for n in range(2, order + 1):
        h = tuple(history[-(n - 1) :])
        table = {g: c for g, c in counts[n].items() if g[:-1] == h}
        if not table:
            break
        d = discounts(list(counts[n].values()))
        ctx_total = sum(table.values())
        gamma = sum(discount(c, d) for c in table.values()) / ctx_total
        own = table.get((*h, token), 0)
        prob = gamma * prob + (max(own - discount(own, d), 0) / ctx_total if own else 0.0)
    return prob


DOCS = [[1, 2, 3, 0], [1, 2, 4, 0], [2, 3, 3, 1, 0], [4, 4, 4, 0], [1, 3, 2, 0], [3, 1, 2, 4, 0]]


@pytest.mark.parametrize("order", [2, 3, 5])
def test_matches_a_naive_reference_implementation(order: int) -> None:
    lm = KneserNeyLM.fit(DOCS, vocab_size=6, order=order, bos_id=0)
    for context in ([], [1], [1, 2], [2, 3, 3], [5, 5], [4, 4, 4, 4]):
        probs = lm.probs(context)
        for token in range(6):
            expected = naive_kn_prob(DOCS, order, vocab=6, bos=0, context=context, token=token)
            assert probs[token] == pytest.approx(expected, rel=1e-9, abs=1e-12)
            assert lm.token_logprob(context, token) == pytest.approx(math.log(expected), rel=1e-9)


@given(st.lists(st.integers(0, 1023), max_size=8))
def test_distributions_are_normalized_and_positive(trained: Trained, context: list[int]) -> None:
    probs = trained.model.lm.probs(context)
    assert probs.shape == (1024,)
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-9)
    assert float(probs.min()) > 0


def test_sparse_count_statistics_never_zero_a_probability() -> None:
    """Count-of-counts where Chen and Goodman's estimate of D3+ is negative (many n-grams seen
    four times, few seen three times): with the estimate clipped to zero, a context whose
    continuations were all seen 3+ times would give every other token probability 0. Discounts
    are floored instead, so every distribution stays strictly positive."""
    sequences = [list(range(start, start + 6)) for start in range(1, 37, 6)]  # disjoint tokens
    docs = [
        *[sequences[0]] * 4, *[sequences[1]] * 4, *[sequences[2]] * 4,
        *[sequences[3]] * 3, *[sequences[4]] * 2, sequences[5],
    ]  # fmt: skip
    lm = KneserNeyLM.fit(docs, vocab_size=40, order=5, bos_id=0)
    assert float(lm.discounts[1:].min()) > 0
    for doc in docs:
        for cut in range(len(doc) + 1):
            probs = lm.probs(doc[:cut])
            assert float(probs.sum()) == pytest.approx(1.0, abs=1e-9)
            assert float(probs.min()) > 0, doc[:cut]


def test_model_beats_a_unigram_baseline_on_held_out_text(trained: Trained) -> None:
    tokenizer = trained.model.tokenizer
    lm = trained.model.lm
    text = '{"id":7,"name":"Ada Lovelace","email":"ada.lovelace@example.com","active":true}'
    doc = [*tokenizer.encode(text), tokenizer.vocab.eos_id]
    unigram_nll = -sum(math.log(lm.unigram[t]) for t in doc) / len(doc)
    assert math.log(lm.perplexity([doc])) < unigram_nll
    assert lm.document_logprob([], doc) == pytest.approx(-math.log(lm.perplexity([doc])) * len(doc))


def test_arrays_round_trip_and_are_validated() -> None:
    lm = KneserNeyLM.fit(DOCS, vocab_size=6, order=3, bos_id=0)
    arrays = lm.to_arrays()
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **arrays)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as archive:
        restored = KneserNeyLM.from_arrays({name: archive[name] for name in archive.files})
    assert np.allclose(restored.probs([1, 2]), lm.probs([1, 2]))

    missing = dict(arrays)
    del missing["gamma_2"]
    with pytest.raises(LMFormatError, match="incomplete"):
        KneserNeyLM.from_arrays(missing)
    bad_uni = dict(arrays)
    bad_uni["unigram"] = np.full(6, 0.5)
    with pytest.raises(LMFormatError, match="sum to one"):
        KneserNeyLM.from_arrays(bad_uni)
    bad_tok = dict(arrays)
    bad_tok["next_tok_3"] = arrays["next_tok_3"] + 100
    with pytest.raises(LMFormatError, match="order-3"):
        KneserNeyLM.from_arrays(bad_tok)
    bad_meta = dict(arrays)
    bad_meta["meta"] = np.array([99, 6, 0])
    with pytest.raises(LMFormatError):
        KneserNeyLM.from_arrays(bad_meta)


def test_fit_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="order"):
        KneserNeyLM.fit(DOCS, vocab_size=6, order=1, bos_id=0)
    with pytest.raises(ValueError, match="62 bits"):
        KneserNeyLM.fit(DOCS, vocab_size=2**20, order=5, bos_id=0)
    with pytest.raises(ValueError, match="non-empty"):
        KneserNeyLM.fit([[]], vocab_size=6, order=3, bos_id=0)
    with pytest.raises(ValueError, match="out of range"):
        KneserNeyLM.fit([[9]], vocab_size=6, order=3, bos_id=0)
    with pytest.raises(ValueError, match="no tokens"):
        KneserNeyLM.fit(DOCS, vocab_size=6, order=3, bos_id=0).perplexity([[]])
