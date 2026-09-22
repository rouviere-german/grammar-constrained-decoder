"""Masked decoding: greedy, nucleus (top-p) sampling and beam search over a token LM.

At each step the language model gives log-probabilities over the whole vocabulary. With a grammar,
tokens that the :class:`TokenIndex` does not allow in the current DFA state get ``-inf`` before the
choice is made (greedy argmax, renormalized nucleus sampling, or beam expansion). EOS is allowed
only in accepting states, and the budget-aware mask only offers tokens from which acceptance is
still reachable within the remaining ``max_tokens``, so every constrained generation ends with
EOS in an accepting state: the output is in the schema language by construction.

Without a grammar the same code decodes freely, which is the baseline the benchmarks compare with.
"""

from __future__ import annotations

import codecs
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from grammar_constrained_decoder.decoding.index import TokenIndex
from grammar_constrained_decoder.lm.tokenizer import ByteVocabulary

__all__ = [
    "Decoder",
    "Generation",
    "GrammarCursor",
    "GrammarViolationError",
    "LanguageModel",
    "StepEvent",
    "Strategy",
]

FloatArray = npt.NDArray[np.float64]
Strategy = Literal["greedy", "top_p", "beam"]
StopReason = Literal["eos", "max_tokens"]


class LanguageModel(Protocol):
    vocab_size: int

    def logprobs(self, context: Sequence[int]) -> FloatArray: ...


class GrammarViolationError(RuntimeError):
    """A token outside the grammar was fed to a cursor (a decoder bug, never user input)."""


class GrammarCursor:
    """Tracks the DFA state and the remaining token budget of one constrained generation."""

    def __init__(self, index: TokenIndex, max_tokens: int) -> None:
        needed = int(index.distance[index.start])
        if needed < 0 or needed > max_tokens:
            raise ValueError(
                f"max_tokens={max_tokens} is too small: this schema needs at least {needed} tokens"
            )
        self.index = index
        self.state = index.start
        self.remaining = max_tokens
        self.finished = False

    def mask(self) -> npt.NDArray[np.bool_]:
        return self.index.mask(self.state, self.remaining)

    def advance(self, token: int) -> None:
        if self.finished:
            raise GrammarViolationError("generation already finished")
        if token == self.index.eos_id:
            if not self.index.accepting[self.state]:
                raise GrammarViolationError("EOS in a non-accepting state")
            self.finished = True
            return
        successor = self.index.next_state(self.state, token)
        if successor < 0 or self.remaining < 1:
            raise GrammarViolationError(f"token {token} is not allowed here")
        self.state = successor
        self.remaining -= 1

    def copy(self) -> GrammarCursor:
        clone = object.__new__(GrammarCursor)
        clone.index, clone.state = self.index, self.state
        clone.remaining, clone.finished = self.remaining, self.finished
        return clone


@dataclass
class StepEvent:
    """One decoding step. The final EOS step has ``eos=True`` and empty text."""

    index: int
    token_id: int
    text: str
    logprob: float
    mask_seconds: float
    eos: bool = False


@dataclass
class Generation:
    token_ids: list[int]
    text: str
    stop_reason: StopReason
    logprob: float
    constrained: bool
    mask_seconds: list[float] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return len(self.token_ids) / self.seconds if self.seconds > 0 else 0.0


@dataclass
class _Beam:
    tokens: list[int]
    cursor: GrammarCursor | None
    score: float


def _nucleus(
    logp: FloatArray, *, temperature: float, top_p: float, rng: np.random.Generator
) -> int:
    finite = np.isfinite(logp)
    scaled = np.where(finite, logp / temperature, -np.inf)
    scaled -= scaled[finite].max()
    probs = np.where(finite, np.exp(scaled), 0.0)
    probs /= probs.sum()
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    cutoff = int(np.searchsorted(cumulative, top_p, side="left")) + 1
    keep = order[: max(1, min(cutoff, int(finite.sum())))]
    kept = probs[keep] / probs[keep].sum()
    return int(rng.choice(keep, p=kept))


class Decoder:
    """Decoding strategies over a :class:`LanguageModel`, with or without a grammar."""

    def __init__(self, lm: LanguageModel, vocab: ByteVocabulary) -> None:
        if lm.vocab_size != vocab.size:
            raise ValueError("model and tokenizer vocabularies differ in size")
        self.lm = lm
        self.vocab = vocab

    def _masked(
        self, context: Sequence[int], cursor: GrammarCursor | None
    ) -> tuple[FloatArray, FloatArray, float]:
        logp = self.lm.logprobs(context)
        if cursor is None:
            return logp, logp, 0.0
        started = time.perf_counter()
        masked = np.where(cursor.mask(), logp, -np.inf)
        return logp, masked, time.perf_counter() - started

    def stream(
        self,
        prompt_ids: Sequence[int],
        *,
        grammar: TokenIndex | None,
        max_tokens: int,
        strategy: Literal["greedy", "top_p"] = "greedy",
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> Iterator[StepEvent]:
        """Yield one event per generated token, then an EOS event if the model stopped."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not 0 < top_p <= 1 or temperature <= 0:
            raise ValueError("top_p must be in (0, 1] and temperature positive")
        rng = np.random.default_rng(seed)
        cursor = GrammarCursor(grammar, max_tokens) if grammar is not None else None
        context = list(prompt_ids)
        text_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        for position in range(max_tokens + 1):
            if cursor is None and position == max_tokens:
                return
            logp, masked, mask_seconds = self._masked(context, cursor)
            if strategy == "greedy":
                token = int(np.argmax(masked))
            else:
                token = _nucleus(masked, temperature=temperature, top_p=top_p, rng=rng)
            if not np.isfinite(masked[token]):  # pragma: no cover - the budget mask prevents it
                raise GrammarViolationError("no token is allowed")
            if cursor is not None:
                cursor.advance(token)
            if token == self.vocab.eos_id:
                yield StepEvent(
                    position,
                    token,
                    text_decoder.decode(b"", final=True),
                    float(logp[token]),
                    mask_seconds,
                    eos=True,
                )
                return
            context.append(token)
            text = text_decoder.decode(self.vocab.token_bytes[token])
            yield StepEvent(position, token, text, float(logp[token]), mask_seconds)

    def generate(
        self,
        prompt_ids: Sequence[int],
        *,
        grammar: TokenIndex | None,
        max_tokens: int,
        strategy: Strategy = "greedy",
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
        beam_width: int = 4,
    ) -> Generation:
        started = time.perf_counter()
        if strategy == "beam":
            result = self.beam_search(
                prompt_ids, grammar=grammar, max_tokens=max_tokens, beam_width=beam_width
            )
            result.seconds = time.perf_counter() - started
            return result
        tokens: list[int] = []
        masks: list[float] = []
        logprob = 0.0
        stop: StopReason = "max_tokens"
        for event in self.stream(
            prompt_ids,
            grammar=grammar,
            max_tokens=max_tokens,
            strategy=strategy,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        ):
            logprob += event.logprob
            if grammar is not None:
                masks.append(event.mask_seconds)
            if event.eos:
                stop = "eos"
            else:
                tokens.append(event.token_id)
        text = self.vocab.decode(tokens).decode("utf-8", errors="replace")
        return Generation(
            tokens, text, stop, logprob, grammar is not None, masks, time.perf_counter() - started
        )

    def beam_search(
        self,
        prompt_ids: Sequence[int],
        *,
        grammar: TokenIndex | None,
        max_tokens: int,
        beam_width: int = 4,
    ) -> Generation:
        """Standard beam search on summed log-probabilities (no length normalization).

        Scores only decrease as tokens are added, so the search stops as soon as the best
        finished hypothesis scores at least as well as every live beam.
        """
        if not 1 <= beam_width <= 32:
            raise ValueError("beam_width must be in 1..32")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        root = GrammarCursor(grammar, max_tokens) if grammar is not None else None
        beams = [_Beam([], root, 0.0)]
        finished: list[tuple[float, list[int], StopReason]] = []
        eos = self.vocab.eos_id
        masks: list[float] = []
        for step in range(max_tokens + 1):
            candidates: list[tuple[float, int, int]] = []
            for beam_id, beam in enumerate(beams):
                _, masked, mask_seconds = self._masked([*prompt_ids, *beam.tokens], beam.cursor)
                if beam.cursor is not None:
                    masks.append(mask_seconds)
                if beam.cursor is None and step == max_tokens:
                    finished.append((beam.score, beam.tokens, "max_tokens"))
                    continue
                finite = np.flatnonzero(np.isfinite(masked))
                top = finite[np.argsort(-masked[finite], kind="stable")[:beam_width]]
                candidates.extend((beam.score + float(masked[t]), beam_id, int(t)) for t in top)
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
            survivors: list[_Beam] = []
            for score, beam_id, token in candidates[:beam_width]:
                parent = beams[beam_id]
                cursor = parent.cursor.copy() if parent.cursor is not None else None
                if cursor is not None:
                    cursor.advance(token)
                if token == eos:
                    finished.append((score, parent.tokens, "eos"))
                else:
                    survivors.append(_Beam([*parent.tokens, token], cursor, score))
            beams = survivors
            best_finished = max((item[0] for item in finished), default=-math.inf)
            if not beams or best_finished >= max(beam.score for beam in beams):
                break
        score, tokens, stop = max(finished, key=lambda item: item[0])
        text = self.vocab.decode(tokens).decode("utf-8", errors="replace")
        return Generation(tokens, text, stop, score, grammar is not None, masks)
