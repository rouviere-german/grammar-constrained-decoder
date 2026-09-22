"""How much token masking distorts the model's distribution, measured exactly on tiny languages.

Masked (locally renormalized) sampling draws token t at each step with probability
``P(t | prefix) / M(prefix)``, where ``M`` is the model mass on allowed tokens. The ideal
constrained sampler would instead draw a complete sequence from ``P(sequence | valid)``: the
model's own distribution restricted to valid sequences and renormalized once, globally. The two
differ whenever the model puts mass on prefixes that later run into few valid continuations; the
local renormalization then over-weights them (the "label bias" discussed for constrained decoding
in e.g. Park et al., "Grammar-Aligned Decoding", NeurIPS 2024).

For a schema whose language is finite and small, this module enumerates every valid token
sequence through the token index (depth-first, EOS only in accepting states) and computes both
distributions exactly, then reports:

- ``valid_mass``: probability the *unconstrained* model assigns to valid sequences;
- ``kl_token`` / ``kl_string``: KL(P(. | valid) || Q_masked) over token sequences and over output
  strings (all tokenizations of a string summed);
- the greedy output and its conditional probability (is greedy the conditional mode?);
- an empirical check: total variation between strings drawn by the real decoder with masked
  sampling and the exact masked distribution Q, which validates the decoder, not the math.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from grammar_constrained_decoder.decoding.decoder import Decoder, LanguageModel
from grammar_constrained_decoder.decoding.index import TokenIndex
from grammar_constrained_decoder.lm.tokenizer import ByteVocabulary

__all__ = ["DistortionReport", "EnumeratedLanguage", "distortion_report", "enumerate_language"]


class LanguageTooLargeError(ValueError):
    """The constrained token language has more sequences than the enumeration cap."""


@dataclass(frozen=True)
class EnumeratedLanguage:
    sequences: list[tuple[int, ...]]
    strings: list[str]
    log_p: np.ndarray  # log P_model(sequence + EOS)
    log_q: np.ndarray  # log Q_masked(sequence + EOS)


def _logsumexp(values: np.ndarray) -> float:
    peak = float(values.max())
    return peak + math.log(float(np.exp(values - peak).sum()))


def enumerate_language(
    lm: LanguageModel,
    index: TokenIndex,
    vocab: ByteVocabulary,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int = 32,
    max_sequences: int = 50_000,
) -> EnumeratedLanguage:
    """All token sequences the masked decoder can emit, with exact P and Q log-probabilities."""
    sequences: list[tuple[int, ...]] = []
    log_p: list[float] = []
    log_q: list[float] = []
    stack: list[tuple[int, tuple[int, ...], float, float]] = [(index.start, (), 0.0, 0.0)]
    eos = vocab.eos_id
    while stack:
        state, tokens, lp, lq = stack.pop()
        logits = lm.logprobs([*prompt_ids, *tokens])
        allowed = np.flatnonzero(index.mask(state, max_tokens - len(tokens)))
        mass = _logsumexp(logits[allowed])
        for token in allowed.tolist():
            step_p = float(logits[token])
            if token == eos:
                sequences.append(tokens)
                log_p.append(lp + step_p)
                log_q.append(lq + step_p - mass)
                if len(sequences) > max_sequences:
                    raise LanguageTooLargeError(f"more than {max_sequences} token sequences")
            else:
                successor = index.next_state(state, token)
                stack.append((successor, (*tokens, token), lp + step_p, lq + step_p - mass))
    strings = [vocab.decode(seq).decode("utf-8") for seq in sequences]
    return EnumeratedLanguage(sequences, strings, np.array(log_p), np.array(log_q))


@dataclass(frozen=True)
class DistortionReport:
    n_sequences: int
    n_strings: int
    valid_mass: float
    kl_token: float
    kl_string: float
    greedy_text: str
    greedy_conditional_probability: float
    greedy_is_conditional_mode: bool
    samples: int
    sampled_tv_to_masked: float

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "n_sequences": self.n_sequences,
            "n_strings": self.n_strings,
            "valid_mass": self.valid_mass,
            "kl_token": self.kl_token,
            "kl_string": self.kl_string,
            "greedy_text": self.greedy_text,
            "greedy_conditional_probability": self.greedy_conditional_probability,
            "greedy_is_conditional_mode": self.greedy_is_conditional_mode,
            "samples": self.samples,
            "sampled_tv_to_masked": self.sampled_tv_to_masked,
        }


def _kl(p: np.ndarray, log_p: np.ndarray, log_q: np.ndarray) -> float:
    """KL(p || q) in nats. KL is non-negative; tiny negative sums are float rounding."""
    return max(0.0, float(np.sum(p * (log_p - log_q))))


def distortion_report(
    lm: LanguageModel,
    index: TokenIndex,
    vocab: ByteVocabulary,
    prompt_ids: Sequence[int],
    *,
    samples: int = 500,
    seed: int = 0,
    max_tokens: int = 32,
) -> DistortionReport:
    language = enumerate_language(lm, index, vocab, prompt_ids, max_tokens=max_tokens)
    log_z = _logsumexp(language.log_p)
    log_cond = language.log_p - log_z
    cond = np.exp(log_cond)
    kl_token = _kl(cond, log_cond, language.log_q)

    by_string_p: dict[str, list[float]] = defaultdict(list)
    by_string_q: dict[str, list[float]] = defaultdict(list)
    for text, lp, lq in zip(language.strings, log_cond, language.log_q, strict=True):
        by_string_p[text].append(float(lp))
        by_string_q[text].append(float(lq))
    names = sorted(by_string_p)
    string_log_p = np.array([_logsumexp(np.array(by_string_p[name])) for name in names])
    string_log_q = np.array([_logsumexp(np.array(by_string_q[name])) for name in names])
    kl_string = _kl(np.exp(string_log_p), string_log_p, string_log_q)

    decoder = Decoder(lm, vocab)
    greedy = decoder.generate(prompt_ids, grammar=index, max_tokens=max_tokens).text
    mode = names[int(np.argmax(string_log_p))]
    greedy_probability = (
        float(np.exp(string_log_p[names.index(greedy)])) if greedy in names else 0.0
    )

    counts: Counter[str] = Counter()
    for draw in range(samples):
        counts[
            decoder.generate(
                prompt_ids, grammar=index, max_tokens=max_tokens, strategy="top_p", seed=seed + draw
            ).text
        ] += 1
    tv = 0.0
    if samples:
        empirical = np.array([counts.get(name, 0) / samples for name in names])
        tv = 0.5 * float(np.abs(empirical - np.exp(string_log_q)).sum())
    return DistortionReport(
        n_sequences=len(language.sequences),
        n_strings=len(names),
        valid_mass=float(math.exp(log_z)),
        kl_token=kl_token,
        kl_string=kl_string,
        greedy_text=greedy,
        greedy_conditional_probability=greedy_probability,
        greedy_is_conditional_mode=greedy == mode,
        samples=samples,
        sampled_tv_to_masked=tv,
    )
