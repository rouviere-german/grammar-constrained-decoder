"""Shared fixtures. Every test runs offline: non-loopback sockets are blocked."""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import structlog
from hypothesis import HealthCheck, settings

from grammar_constrained_decoder.decoding.compiler import GrammarCompiler
from grammar_constrained_decoder.lm.tokenizer import ByteVocabulary
from grammar_constrained_decoder.training.pipeline import (
    LoadedModel,
    TrainConfig,
    TrainResult,
    load_model,
    train_pipeline,
)

settings.register_profile(
    "dev", max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

REPO_ROOT = Path(__file__).resolve().parents[1]
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
# A small but realistic training run for tests (the demo and benches use the full default).
TEST_TRAIN_CONFIG = TrainConfig(docs_per_schema=60, eval_docs_per_schema=10, free_text_docs=300)
MULTI_BYTE_TOKENS = (
    b"}}", b"]}", b'"}', b'",', b'","', b'":', b'":"', b'{"', b"true", b"false", b"null",
    b"[]", b"{}", b"12", b"ab", b"abc", b",{", b"},{", b"\\n", b'\\"', b"\xc3\xa9", b"\xe4\xb8",
)  # fmt: skip


@pytest.fixture(autouse=True)
def _offline_and_clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("APP_"):
            monkeypatch.delenv(name, raising=False)

    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _LOOPBACK:
            raise RuntimeError(f"tests must not use the network (attempted {host})")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    structlog.reset_defaults()
    root.handlers = handlers
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = False


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


def make_vocab(
    extra: Sequence[bytes] = MULTI_BYTE_TOKENS, *, singles: bool = True
) -> ByteVocabulary:
    """EOS (id 0), optionally all 256 single bytes, then ``extra`` multi-byte tokens."""
    tokens: list[bytes] = [b""]
    if singles:
        tokens += [bytes([value]) for value in range(256)]
    tokens += list(extra)
    return ByteVocabulary.from_tokens(tokens, eos_id=0)


@pytest.fixture(scope="session")
def small_vocab() -> ByteVocabulary:
    return make_vocab()


class UniformLM:
    """Every token equally likely."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def logprobs(self, context: Sequence[int]) -> npt.NDArray[np.float64]:
        del context
        return np.full(self.vocab_size, -np.log(self.vocab_size))


class RandomLM:
    """Deterministic pseudo-random next-token distributions, keyed by the recent context."""

    def __init__(self, vocab_size: int, seed: int = 0, *, sharpness: float = 3.0) -> None:
        self.vocab_size = vocab_size
        self.seed = seed
        self.sharpness = sharpness

    def logprobs(self, context: Sequence[int]) -> npt.NDArray[np.float64]:
        key = hash((self.seed, tuple(context[-3:]))) % (2**32)
        logits = np.random.default_rng(key).normal(size=self.vocab_size) * self.sharpness
        logits -= logits.max()
        result: npt.NDArray[np.float64] = logits - np.log(np.exp(logits).sum())
        return result


class AdversarialLM:
    """Puts almost all mass on tokens that break JSON (a model that 'wants' to be invalid)."""

    def __init__(self, vocab: ByteVocabulary) -> None:
        self.vocab_size = vocab.size
        bad = [
            i for i, data in enumerate(vocab.token_bytes) if data in (b"}}", b"x", b"<", b"\x00")
        ]
        logits = np.full(vocab.size, -20.0)
        logits[bad] = 0.0
        self._logp: npt.NDArray[np.float64] = logits - np.log(np.exp(logits).sum())

    def logprobs(self, context: Sequence[int]) -> npt.NDArray[np.float64]:
        del context
        return self._logp


@dataclass(frozen=True)
class Trained:
    root: Path
    result: TrainResult
    model: LoadedModel


@pytest.fixture(scope="session")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    root = tmp_path_factory.mktemp("model")
    result = train_pipeline(root, TEST_TRAIN_CONFIG)
    return Trained(root, result, load_model(root / "registry"))


@pytest.fixture(scope="session")
def compiler(trained: Trained) -> GrammarCompiler:
    return GrammarCompiler(trained.model.tokenizer.vocab, memory_entries=256)
