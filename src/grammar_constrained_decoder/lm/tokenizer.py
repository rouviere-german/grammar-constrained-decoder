"""Byte-level BPE tokenizer trained in-repo with Hugging Face ``tokenizers`` (no downloads).

Constrained decoding needs the exact bytes of every token. A byte-level BPE model stores tokens as
strings over a 256-character alphabet that stands for the 256 byte values (the GPT-2 mapping: the
printable Latin-1 characters stand for themselves, every other byte is shifted to U+0100 and
up). :func:`bytes_to_unicode` rebuilds that mapping, :class:`ByteVocabulary` inverts it to get
``token id -> bytes``, and loading refuses a tokenizer whose alphabet differs from the mapping.
Because the initial alphabet contains all 256 bytes, every byte string can be tokenized, which is
what makes the dead-end check in ``decoding/index.py`` pass.

The tokenizer is stored as the library's own JSON (never pickle) and identified by the SHA-256 of
that JSON.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from grammar_constrained_decoder.core.canonical import sha256_hex

__all__ = [
    "EOS_TOKEN",
    "BPETokenizer",
    "ByteVocabulary",
    "TokenizerIntegrityError",
    "bytes_to_unicode",
]

# The end-of-sequence marker is a vocabulary entry, not a credential (hence the S105/B105 waivers).
EOS_TOKEN = "<|eos|>"  # noqa: S105  # nosec B105
_MAX_VOCAB = 16_384


class TokenizerIntegrityError(ValueError):
    """The tokenizer JSON is not the expected byte-level BPE model or its hash differs."""


def bytes_to_unicode() -> dict[int, str]:
    """The byte -> character table used by byte-level BPE (GPT-2 convention)."""
    printable = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("\xa1"), ord("\xac") + 1),
        *range(ord("\xae"), ord("\xff") + 1),
    ]
    table = {byte: chr(byte) for byte in printable}
    shift = 0
    for byte in range(256):
        if byte not in table:
            table[byte] = chr(256 + shift)
            shift += 1
    return table


_BYTE_OF_CHAR = {char: byte for byte, char in bytes_to_unicode().items()}


@dataclass(frozen=True)
class ByteVocabulary:
    """Token id -> raw bytes. Special tokens map to ``b""`` and are listed in ``special_ids``."""

    token_bytes: tuple[bytes, ...]
    eos_id: int
    special_ids: frozenset[int]

    @property
    def size(self) -> int:
        return len(self.token_bytes)

    @cached_property
    def max_token_length(self) -> int:
        return max(len(data) for data in self.token_bytes)

    @cached_property
    def sha256(self) -> str:
        payload = json.dumps(
            {
                "tokens": [data.hex() for data in self.token_bytes],
                "eos": self.eos_id,
                "special": sorted(self.special_ids),
            },
            separators=(",", ":"),
        )
        return sha256_hex(payload)

    def decode(self, ids: Sequence[int]) -> bytes:
        return b"".join(self.token_bytes[token] for token in ids)

    @classmethod
    def from_tokens(
        cls, token_bytes: Sequence[bytes], *, eos_id: int, special_ids: Iterable[int] = ()
    ) -> ByteVocabulary:
        specials = frozenset({eos_id, *special_ids})
        tokens = tuple(
            b"" if index in specials else bytes(data) for index, data in enumerate(token_bytes)
        )
        if any(not data for index, data in enumerate(tokens) if index not in specials):
            raise ValueError("non-special tokens must have at least one byte")
        return cls(tokens, eos_id, specials)


class BPETokenizer:
    """Wrapper around a trained ``tokenizers.Tokenizer`` with a verified byte vocabulary."""

    def __init__(self, tokenizer: Tokenizer, *, json_text: str | None = None) -> None:
        self._tokenizer = tokenizer
        self._json = json_text if json_text is not None else tokenizer.to_str()
        self.vocab = self._build_vocabulary()

    @classmethod
    def train(
        cls, texts: Iterable[str], *, vocab_size: int = 1024, min_frequency: int = 2
    ) -> BPETokenizer:
        if not 257 <= vocab_size <= _MAX_VOCAB:
            raise ValueError(f"vocab_size must be in 257..{_MAX_VOCAB}")
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[EOS_TOKEN],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer)

    @classmethod
    def from_json(cls, text: str, *, expected_sha256: str | None = None) -> BPETokenizer:
        """Load tokenizer JSON; verify the digest first when one is given."""
        digest = sha256_hex(text)
        if expected_sha256 is not None and digest != expected_sha256:
            raise TokenizerIntegrityError("tokenizer JSON does not match the expected SHA-256")
        try:
            spec = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TokenizerIntegrityError("tokenizer file is not JSON") from exc
        model = spec.get("model", {}) if isinstance(spec, dict) else {}
        pre = spec.get("pre_tokenizer") if isinstance(spec, dict) else None
        if (
            model.get("type") != "BPE"
            or not isinstance(pre, dict)
            or pre.get("type") != "ByteLevel"
        ):
            raise TokenizerIntegrityError("expected a byte-level BPE tokenizer")
        return cls(Tokenizer.from_str(text), json_text=text)

    def to_json(self) -> str:
        return self._json

    @property
    def sha256(self) -> str:
        return sha256_hex(self._json)

    def _build_vocabulary(self) -> ByteVocabulary:
        vocab = self._tokenizer.get_vocab(with_added_tokens=True)
        if not 0 < len(vocab) <= _MAX_VOCAB:
            raise TokenizerIntegrityError("vocabulary size is out of range")
        if EOS_TOKEN not in vocab:
            raise TokenizerIntegrityError("tokenizer has no EOS token")
        size = max(vocab.values()) + 1
        if size != len(vocab):
            raise TokenizerIntegrityError("token ids are not contiguous")
        alphabet = set(pre_tokenizers.ByteLevel.alphabet())
        if alphabet != set(_BYTE_OF_CHAR):
            raise TokenizerIntegrityError("byte-level alphabet differs from the byte mapping")
        tokens: list[bytes] = [b""] * size
        eos_id = vocab[EOS_TOKEN]
        for text, index in vocab.items():
            if index == eos_id:
                continue
            try:
                tokens[index] = bytes(_BYTE_OF_CHAR[char] for char in text)
            except KeyError as exc:
                raise TokenizerIntegrityError(
                    "token uses a character outside the alphabet"
                ) from exc
        singles = {data for data in tokens if len(data) == 1}
        if len(singles) != 256:
            raise TokenizerIntegrityError("vocabulary does not contain all 256 byte tokens")
        return ByteVocabulary.from_tokens(tokens, eos_id=eos_id)

    def encode(self, text: str) -> list[int]:
        """Token ids for ``text``. A literal ``<|eos|>`` in the text never becomes the EOS token.

        ``tokenizers`` always splits registered special tokens out of raw text, so user text could
        otherwise end a document early. Each occurrence is encoded as two ordinary pieces instead.
        """
        pieces = text.split(EOS_TOKEN)
        ids: list[int] = []
        for position, piece in enumerate(pieces):
            if position:
                ids += self._encode_plain(EOS_TOKEN[:2]) + self._encode_plain(EOS_TOKEN[2:])
            ids += self._encode_plain(piece)
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        if not text:
            return []
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode_bytes(self, ids: Sequence[int]) -> bytes:
        return self.vocab.decode(ids)

    def decode(self, ids: Sequence[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")
