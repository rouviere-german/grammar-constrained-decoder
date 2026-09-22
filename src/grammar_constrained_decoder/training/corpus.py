"""Synthetic training corpus: prompted JSON documents from bundled schemas, mixed with free text.

Each JSON record is ``prompt + canonical JSON`` where the prompt is
``"Q: <description>\\n<schema id>:\\n"``. The schema id sits right before the document, inside
the n-gram window, so the model's first tokens depend on it (a 5-gram cannot see a description a
dozen tokens back); for held-out schemas the id is new and the model has to back off. The newline
before the document keeps the first JSON token on a pre-tokenizer boundary, so the prompt
tokenizes the same way at training and at generation time. Free-text records are short
template sentences; they make the unconstrained model sometimes answer with prose, which is part
of why unconstrained output is often invalid.

Generation is deterministic for a seed. The corpus is serialized as canonical JSON Lines and
identified by its SHA-256.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from grammar_constrained_decoder.core.canonical import canonical_json, sha256_hex
from grammar_constrained_decoder.schema.canonical import canonical_dumps
from grammar_constrained_decoder.schema.ir import parse_schema
from grammar_constrained_decoder.schema.sampler import CITIES, WORDS, InstanceSampler

__all__ = [
    "BundledSchema",
    "CorpusRecord",
    "corpus_jsonl",
    "corpus_sha256",
    "free_text",
    "generate_corpus",
    "load_bundled_schemas",
    "load_distortion_schemas",
    "prompt_for",
]

PROMPT_TEMPLATE = "Q: {description}\n{name}:\n"
_SUBJECTS = ("the report", "our team", "the server", "a customer", "the city", "this week")
_VERBS = ("mentions", "needs", "describes", "updates", "shows", "tracks")
_ENDINGS = ("today", "again", "for the review", "before noon", "in the logs", "on the map")


@dataclass(frozen=True)
class BundledSchema:
    id: str
    split: Literal["train", "eval", "tiny"]
    description: str
    schema: dict[str, Any]


class CorpusRecord(BaseModel):
    """One training document. ``completion`` is canonical JSON for ``kind == "json"``."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    kind: Literal["json", "text"]
    schema_id: str | None = Field(default=None, pattern=r"^[a-z0-9_]{1,64}$")
    prompt: str = Field(max_length=512)
    completion: str = Field(min_length=1, max_length=4096)

    @property
    def text(self) -> str:
        return self.prompt + self.completion


def _load(name: str) -> list[BundledSchema]:
    raw = resources.files("grammar_constrained_decoder.data").joinpath(name).read_text("utf-8")
    payload = json.loads(raw)
    return [
        BundledSchema(item["id"], item["split"], item["description"], item["schema"])
        for item in payload["schemas"]
    ]


def load_bundled_schemas() -> list[BundledSchema]:
    """The 30 bundled schemas (18 train, 12 held-out eval)."""
    return _load("bundled_schemas.json")


def load_distortion_schemas() -> list[BundledSchema]:
    """20 tiny schemas with small, enumerable languages for the distortion study."""
    return _load("distortion_schemas.json")


def prompt_for(schema: BundledSchema) -> str:
    """The prompt used for a bundled schema in training, the demo and the benchmarks."""
    return PROMPT_TEMPLATE.format(description=schema.description, name=schema.id)


def free_text(rng: np.random.Generator) -> str:
    """One short template sentence (sometimes two)."""

    def pick(options: Sequence[str]) -> str:
        return options[int(rng.integers(len(options)))]

    sentences = []
    for _ in range(int(rng.integers(1, 3))):
        sentences.append(
            f"{pick(_SUBJECTS).capitalize()} {pick(_VERBS)} {pick(WORDS)} {pick(WORDS)} "
            f"in {pick(CITIES)} {pick(_ENDINGS)}."
        )
    return " ".join(sentences)


def generate_corpus(
    schemas: Iterable[BundledSchema],
    *,
    docs_per_schema: int,
    free_text_docs: int,
    seed: int,
) -> list[CorpusRecord]:
    """Deterministic corpus: ``docs_per_schema`` JSON records per schema plus free text."""
    rng = np.random.default_rng(seed)
    records: list[CorpusRecord] = []
    for bundled in schemas:
        ir = parse_schema(bundled.schema)
        sampler = InstanceSampler(ir, rng)
        prompt = prompt_for(bundled)
        for _ in range(docs_per_schema):
            instance = sampler.sample()
            records.append(
                CorpusRecord(
                    kind="json",
                    schema_id=bundled.id,
                    prompt=prompt,
                    completion=canonical_dumps(ir, instance),
                )
            )
    for _ in range(free_text_docs):
        records.append(CorpusRecord(kind="text", prompt="", completion=free_text(rng)))
    order = rng.permutation(len(records)).tolist()
    return [records[index] for index in order]


def corpus_jsonl(records: Iterable[CorpusRecord]) -> bytes:
    lines = [canonical_json(record.model_dump(mode="json")) for record in records]
    return ("\n".join(lines) + "\n").encode("utf-8")


def corpus_sha256(records: Iterable[CorpusRecord]) -> str:
    return sha256_hex(corpus_jsonl(records))
