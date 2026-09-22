"""Interpolated modified Kneser-Ney n-gram language model over token ids, in numpy.

Reference: Chen and Goodman, "An Empirical Study of Smoothing Techniques for Language Modeling"
(1998), building on Kneser and Ney (1995). For an order-n context h and next token w::

    P(w | h) = max(c(h w) - D(c(h w)), 0) / c(h)  +  gamma(h) * P(w | h')
    gamma(h) = (D1 N1(h) + D2 N2(h) + D3+ N3+(h)) / c(h)

where h' drops the oldest token of h. ``c`` is the raw count at the highest order and the
continuation count (number of distinct left extensions) at lower orders. ``D1, D2, D3+`` come from
count-of-counts per order (``Y = n1 / (n1 + 2 n2)``, ``D1 = 1 - 2Y n2/n1``, ``D2 = 2 - 3Y n3/n2``,
``D3+ = 3 - 4Y n4/n3``). The unigram level interpolates continuation probabilities with the
uniform distribution, so every token has non-zero probability. When h was never seen, P(w | h)
is P(w | h'). Probabilities sum to exactly one by construction (tests check this).

Storage: for each order n >= 2 the model keeps sorted context keys (token ids packed base V into
an int64), per-context offsets, backoff weights and, per (context, token) entry, the already
discounted probability mass. :meth:`KneserNeyLM.logprobs` is then n - 1 binary searches plus
vector updates. Everything round-trips through plain numeric arrays (npz, ``allow_pickle=False``).

Documents are padded on the left with ``order - 1`` EOS tokens (used as beginning-of-document) and
end with EOS; only positions inside a document are counted, so padding never teaches the model to
predict EOS after EOS.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["KneserNeyLM", "LMFormatError"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
_MAX_KEY_BITS = 62


class LMFormatError(ValueError):
    """Arrays loaded from disk do not describe a valid model."""


def _discounts(counts: npt.NDArray[np.int64]) -> tuple[float, float, float]:
    n = [int(np.count_nonzero(counts == k)) for k in (1, 2, 3, 4)]
    if n[0] == 0 or n[1] == 0:
        return 0.5, 1.0, 1.5
    y = n[0] / (n[0] + 2 * n[1])
    d1 = 1 - 2 * y * n[1] / n[0]
    d2 = 2 - 3 * y * n[2] / n[1] if n[1] else 1.0
    d3 = 3 - 4 * y * n[3] / n[2] if n[2] else 1.5
    return (
        float(min(max(d1, 0.0), 1.0)),
        float(min(max(d2, 0.0), 2.0)),
        float(min(max(d3, 0.0), 3.0)),
    )


def _apply_discount(counts: npt.NDArray[np.int64], d: tuple[float, float, float]) -> FloatArray:
    discount = np.where(counts == 1, d[0], np.where(counts == 2, d[1], d[2]))
    return np.maximum(counts - discount, 0.0)


@dataclass
class _Level:
    ctx_keys: IntArray
    offsets: IntArray
    gamma: FloatArray
    next_tok: npt.NDArray[np.int32]
    next_prob: FloatArray


class KneserNeyLM:
    """Fitted model. Use :meth:`fit` or :meth:`from_arrays`; instances are read-only."""

    def __init__(
        self,
        *,
        order: int,
        vocab_size: int,
        bos_id: int,
        unigram: FloatArray,
        levels: dict[int, _Level],
        discounts: FloatArray,
    ) -> None:
        self.order = order
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.unigram = unigram
        self.levels = levels
        self.discounts = discounts
        self._log_unigram = np.log(unigram)

    # -- fitting ---------------------------------------------------------------------------------

    @classmethod
    def fit(
        cls, documents: Sequence[Sequence[int]], *, vocab_size: int, order: int = 5, bos_id: int
    ) -> KneserNeyLM:
        if order < 2:
            raise ValueError("order must be at least 2")
        if vocab_size < 2 or math.log2(vocab_size) * order > _MAX_KEY_BITS:
            raise ValueError("vocab_size ** order must fit in 62 bits")
        if not documents or not any(documents):
            raise ValueError("at least one non-empty document is required")
        pad = order - 1
        pieces: list[IntArray] = []
        masks: list[npt.NDArray[np.bool_]] = []
        for doc in documents:
            tokens = np.asarray(doc, dtype=np.int64)
            if tokens.size and (tokens.min() < 0 or tokens.max() >= vocab_size):
                raise ValueError("token id out of range")
            pieces.append(np.concatenate([np.full(pad, bos_id, dtype=np.int64), tokens]))
            masks.append(
                np.concatenate([np.zeros(pad, dtype=np.bool_), np.ones(tokens.size, np.bool_)])
            )
        stream = np.concatenate(pieces)
        positions = np.flatnonzero(np.concatenate(masks))
        v = np.int64(vocab_size)

        def keys(n: int) -> IntArray:
            key = np.zeros(positions.size, dtype=np.int64)
            for offset in range(n - 1, -1, -1):
                key = key * v + stream[positions - offset]
            return key

        # counts[n]: (ngram keys, counts) where counts are raw at the top order and continuation
        # counts below it.
        counts: dict[int, tuple[IntArray, npt.NDArray[np.int64]]] = {}
        top_keys, top_counts = np.unique(keys(order), return_counts=True)
        counts[order] = (top_keys, top_counts.astype(np.int64))
        for n in range(order - 1, 0, -1):
            longer = np.unique(keys(n + 1))
            ngram_keys, continuation = np.unique(longer % v**n, return_counts=True)
            counts[n] = (ngram_keys, continuation.astype(np.int64))

        discounts = np.zeros((order + 1, 3), dtype=np.float64)
        uni_keys, uni_counts = counts[1]
        d = _discounts(uni_counts)
        discounts[1] = d
        total = float(uni_counts.sum())
        gamma0 = (
            d[0] * np.count_nonzero(uni_counts == 1)
            + d[1] * np.count_nonzero(uni_counts == 2)
            + d[2] * np.count_nonzero(uni_counts >= 3)
        ) / total
        unigram = np.full(vocab_size, gamma0 / vocab_size, dtype=np.float64)
        unigram[uni_keys] += _apply_discount(uni_counts, d) / total

        levels: dict[int, _Level] = {}
        for n in range(2, order + 1):
            ngram_keys, ngram_counts = counts[n]
            d = _discounts(ngram_counts)
            discounts[n] = d
            contexts = ngram_keys // v
            ctx_keys, starts = np.unique(contexts, return_index=True)
            offsets = np.append(starts, ngram_keys.size).astype(np.int64)
            ctx_total = np.add.reduceat(ngram_counts, starts).astype(np.float64)
            ones = np.add.reduceat((ngram_counts == 1).astype(np.int64), starts)
            twos = np.add.reduceat((ngram_counts == 2).astype(np.int64), starts)
            more = np.add.reduceat((ngram_counts >= 3).astype(np.int64), starts)
            gamma = (d[0] * ones + d[1] * twos + d[2] * more) / ctx_total
            per_entry_total = np.repeat(ctx_total, np.diff(offsets))
            levels[n] = _Level(
                ctx_keys=ctx_keys.astype(np.int64),
                offsets=offsets,
                gamma=gamma.astype(np.float64),
                next_tok=(ngram_keys % v).astype(np.int32),
                next_prob=_apply_discount(ngram_counts, d) / per_entry_total,
            )
        return cls(
            order=order,
            vocab_size=vocab_size,
            bos_id=bos_id,
            unigram=unigram,
            levels=levels,
            discounts=discounts,
        )

    # -- inference -------------------------------------------------------------------------------

    def _context(self, context: Sequence[int]) -> list[int]:
        tail = list(context[-(self.order - 1) :]) if context else []
        return [self.bos_id] * (self.order - 1 - len(tail)) + tail

    def _find(self, n: int, history: Sequence[int]) -> int:
        level = self.levels[n]
        key = 0
        for token in history:
            key = key * self.vocab_size + int(token)
        index = int(np.searchsorted(level.ctx_keys, key))
        if index < level.ctx_keys.size and int(level.ctx_keys[index]) == key:
            return index
        return -1

    def probs(self, context: Sequence[int]) -> FloatArray:
        """Next-token distribution (sums to one) given the preceding token ids."""
        history = self._context(context)
        probs = self.unigram.copy()
        for n in range(2, self.order + 1):
            index = self._find(n, history[-(n - 1) :])
            if index < 0:
                break
            level = self.levels[n]
            start, end = int(level.offsets[index]), int(level.offsets[index + 1])
            probs *= level.gamma[index]
            probs[level.next_tok[start:end]] += level.next_prob[start:end]
        return probs

    def logprobs(self, context: Sequence[int]) -> FloatArray:
        return np.log(self.probs(context))

    def token_logprob(self, context: Sequence[int], token: int) -> float:
        """log P(token | context) without materializing the whole distribution."""
        history = self._context(context)
        prob = float(self.unigram[token])
        for n in range(2, self.order + 1):
            index = self._find(n, history[-(n - 1) :])
            if index < 0:
                break
            level = self.levels[n]
            start, end = int(level.offsets[index]), int(level.offsets[index + 1])
            prob *= float(level.gamma[index])
            slot = start + int(np.searchsorted(level.next_tok[start:end], token))
            if slot < end and int(level.next_tok[slot]) == token:
                prob += float(level.next_prob[slot])
        return math.log(prob)

    def document_logprob(self, prefix: Sequence[int], continuation: Sequence[int]) -> float:
        context = list(prefix)
        total = 0.0
        for token in continuation:
            total += self.token_logprob(context, token)
            context.append(token)
        return total

    def perplexity(self, documents: Sequence[Sequence[int]]) -> float:
        total, count = 0.0, 0
        for doc in documents:
            total += self.document_logprob([], doc)
            count += len(doc)
        if count == 0:
            raise ValueError("no tokens to score")
        return math.exp(-total / count)

    # -- serialization ---------------------------------------------------------------------------

    def to_arrays(self) -> dict[str, npt.NDArray[np.generic]]:
        arrays: dict[str, npt.NDArray[np.generic]] = {
            "meta": np.array([self.order, self.vocab_size, self.bos_id], dtype=np.int64),
            "unigram": self.unigram,
            "discounts": self.discounts,
        }
        for n, level in self.levels.items():
            arrays[f"ctx_keys_{n}"] = level.ctx_keys
            arrays[f"offsets_{n}"] = level.offsets
            arrays[f"gamma_{n}"] = level.gamma
            arrays[f"next_tok_{n}"] = level.next_tok
            arrays[f"next_prob_{n}"] = level.next_prob
        return arrays

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, npt.NDArray[Any]]) -> KneserNeyLM:
        """Rebuild a model from arrays, checking every structural invariant first."""
        try:
            meta = np.asarray(arrays["meta"], dtype=np.int64)
            order, vocab_size, bos_id = (int(value) for value in meta)
            unigram = np.asarray(arrays["unigram"], dtype=np.float64)
            discounts = np.asarray(arrays["discounts"], dtype=np.float64)
            levels = {
                n: _Level(
                    ctx_keys=np.asarray(arrays[f"ctx_keys_{n}"], dtype=np.int64),
                    offsets=np.asarray(arrays[f"offsets_{n}"], dtype=np.int64),
                    gamma=np.asarray(arrays[f"gamma_{n}"], dtype=np.float64),
                    next_tok=np.asarray(arrays[f"next_tok_{n}"], dtype=np.int32),
                    next_prob=np.asarray(arrays[f"next_prob_{n}"], dtype=np.float64),
                )
                for n in range(2, order + 1)
            }
        except (KeyError, ValueError, TypeError) as exc:
            raise LMFormatError("model arrays are incomplete or malformed") from exc
        if not 2 <= order <= 8 or not 2 <= vocab_size <= 1 << 16 or not 0 <= bos_id < vocab_size:
            raise LMFormatError("model metadata is out of range")
        if unigram.shape != (vocab_size,) or not np.all(unigram > 0):
            raise LMFormatError("unigram distribution is malformed")
        if abs(float(unigram.sum()) - 1.0) > 1e-6:
            raise LMFormatError("unigram distribution does not sum to one")
        for n, level in levels.items():
            k = level.ctx_keys.size
            entries = level.next_tok.size
            if (
                level.offsets.shape != (k + 1,)
                or level.gamma.shape != (k,)
                or level.next_prob.shape != (entries,)
                or (k and (level.offsets[0] != 0 or level.offsets[-1] != entries))
                or np.any(np.diff(level.offsets) <= 0)
                or np.any(np.diff(level.ctx_keys) <= 0)
                or (entries and (level.next_tok.min() < 0 or level.next_tok.max() >= vocab_size))
                or np.any(level.gamma < 0)
                or np.any(level.next_prob < 0)
            ):
                raise LMFormatError(f"order-{n} tables are malformed")
        return cls(
            order=order,
            vocab_size=vocab_size,
            bos_id=bos_id,
            unigram=unigram,
            levels=levels,
            discounts=discounts,
        )
