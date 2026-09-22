"""End-to-end training pipeline and verified model loading.

``train_pipeline`` (well under a minute on a laptop CPU):

1. generate the synthetic corpus from the 18 train schemas plus free text, and a held-out corpus
   from the 12 eval schemas; both pass the data contracts in ``ml/contracts.py``;
2. train the byte-level BPE tokenizer on the training corpus;
3. fit the interpolated Kneser-Ney 5-gram model on the tokenized corpus;
4. write a run manifest (seed, corpus fingerprints, package versions, tokenizer and counts
   SHA-256, perplexities) and register ``counts.npz`` + ``tokenizer.json`` + ``model.json`` in
   the content-addressed registry under the alias ``production``.

``load_model`` reads a version back only after verification: the registry checks the version
record and every blob against their content hashes, the manifest's ``tokenizer_sha256`` and
``counts_sha256`` must match the loaded files, npz is read with ``allow_pickle=False``, and the
model arrays are structurally validated before use.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

from grammar_constrained_decoder.core.canonical import sha256_hex
from grammar_constrained_decoder.core.fileio import atomic_write_bytes
from grammar_constrained_decoder.lm.kneser_ney import KneserNeyLM, LMFormatError
from grammar_constrained_decoder.lm.tokenizer import BPETokenizer, TokenizerIntegrityError
from grammar_constrained_decoder.ml.contracts import validate_corpus
from grammar_constrained_decoder.ml.manifest import RunManifest, create_manifest, seed_everything
from grammar_constrained_decoder.ml.registry import ModelRegistry, ModelVersion, RegistryError
from grammar_constrained_decoder.ml.serialization import dumps_npz, loads_npz
from grammar_constrained_decoder.training.corpus import (
    CorpusRecord,
    corpus_jsonl,
    generate_corpus,
    load_bundled_schemas,
)

__all__ = [
    "MODEL_NAME",
    "LoadedModel",
    "ModelIntegrityError",
    "TrainConfig",
    "TrainResult",
    "load_model",
    "train_pipeline",
]

MODEL_NAME = "kn5-json-lm"
PACKAGES = ("numpy", "tokenizers", "jsonschema", "pydantic")


class ModelIntegrityError(RuntimeError):
    """A registered model failed verification and was not loaded."""


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    docs_per_schema: int = 120
    eval_docs_per_schema: int = 20
    free_text_docs: int = 600
    vocab_size: int = 1024
    order: int = 5


@dataclass(frozen=True)
class TrainResult:
    version: ModelVersion
    corpus_path: Path
    eval_corpus_path: Path
    metrics: dict[str, float]
    seconds: dict[str, float]


@dataclass(frozen=True)
class LoadedModel:
    lm: KneserNeyLM
    tokenizer: BPETokenizer
    version: ModelVersion
    prompt_lengths: npt.NDArray[np.int64]

    @property
    def manifest(self) -> RunManifest:
        return self.version.manifest


def _encode_records(tokenizer: BPETokenizer, records: list[CorpusRecord]) -> list[list[int]]:
    eos = tokenizer.vocab.eos_id
    return [[*tokenizer.encode(record.text), eos] for record in records]


def train_pipeline(workdir: Path, config: TrainConfig | None = None) -> TrainResult:
    cfg = config or TrainConfig()
    seed_everything(cfg.seed)
    workdir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    started = time.perf_counter()

    schemas = load_bundled_schemas()
    train_schemas = [item for item in schemas if item.split == "train"]
    eval_schemas = [item for item in schemas if item.split == "eval"]
    corpus = generate_corpus(
        train_schemas,
        docs_per_schema=cfg.docs_per_schema,
        free_text_docs=cfg.free_text_docs,
        seed=cfg.seed,
    )
    eval_corpus = generate_corpus(
        eval_schemas, docs_per_schema=cfg.eval_docs_per_schema, free_text_docs=0, seed=cfg.seed + 1
    )
    validate_corpus(corpus, schemas, allowed_split="train", min_docs_per_schema=cfg.docs_per_schema)
    validate_corpus(
        eval_corpus, schemas, allowed_split="eval", min_docs_per_schema=cfg.eval_docs_per_schema
    )
    corpus_path = workdir / "corpus.jsonl"
    eval_path = workdir / "eval_corpus.jsonl"
    atomic_write_bytes(corpus_path, corpus_jsonl(corpus))
    atomic_write_bytes(eval_path, corpus_jsonl(eval_corpus))
    timings["corpus"] = time.perf_counter() - started

    mark = time.perf_counter()
    tokenizer = BPETokenizer.train((record.text for record in corpus), vocab_size=cfg.vocab_size)
    timings["tokenizer"] = time.perf_counter() - mark

    mark = time.perf_counter()
    documents = _encode_records(tokenizer, corpus)
    lm = KneserNeyLM.fit(
        documents, vocab_size=tokenizer.vocab.size, order=cfg.order, bos_id=tokenizer.vocab.eos_id
    )
    timings["lm_fit"] = time.perf_counter() - mark

    mark = time.perf_counter()
    heldout = _encode_records(tokenizer, eval_corpus)
    rng = np.random.default_rng(cfg.seed)
    sample = [
        documents[i]
        for i in rng.choice(len(documents), size=min(200, len(documents)), replace=False)
    ]
    metrics = {
        "train_perplexity": round(lm.perplexity(sample), 4),
        "heldout_perplexity": round(lm.perplexity(heldout), 4),
        "train_tokens": float(sum(len(doc) for doc in documents)),
        "vocab_size": float(tokenizer.vocab.size),
    }
    timings["evaluation"] = time.perf_counter() - mark
    prompt_lengths = np.array(
        [len(tokenizer.encode(record.prompt)) for record in corpus if record.kind == "json"],
        dtype=np.int64,
    )

    counts = dumps_npz({**lm.to_arrays(), "prompt_lengths": prompt_lengths})
    tokenizer_bytes = tokenizer.to_json().encode("utf-8")
    params: dict[str, JsonValue] = {
        "model": "interpolated-modified-kneser-ney",
        "order": cfg.order,
        "vocab_size": cfg.vocab_size,
        "docs_per_schema": cfg.docs_per_schema,
        "free_text_docs": cfg.free_text_docs,
        "train_schema_ids": [item.id for item in train_schemas],
        "eval_schema_ids": [item.id for item in eval_schemas],
        "tokenizer_sha256": sha256_hex(tokenizer_bytes),
        "vocab_sha256": tokenizer.vocab.sha256,
        "counts_sha256": sha256_hex(counts),
        "corpus_sha256": sha256_hex(corpus_path.read_bytes()),
        "eval_corpus_sha256": sha256_hex(eval_path.read_bytes()),
    }
    manifest = create_manifest(
        seed=cfg.seed,
        datasets=[corpus_path, eval_path],
        params=params,
        metrics=metrics,
        packages=PACKAGES,
    )
    model_meta = json.dumps(
        {"name": MODEL_NAME, "order": cfg.order, "vocab_size": tokenizer.vocab.size},
        sort_keys=True,
    ).encode("utf-8")
    registry = ModelRegistry(workdir / "registry")
    version = registry.register(
        MODEL_NAME,
        {"counts.npz": counts, "tokenizer.json": tokenizer_bytes, "model.json": model_meta},
        manifest,
        metrics=metrics,
    )
    registry.set_alias(MODEL_NAME, "production", version.digest)
    manifest.write(workdir / "manifest.json")
    timings["total"] = time.perf_counter() - started
    return TrainResult(version, corpus_path, eval_path, metrics, timings)


def _param(version: ModelVersion, key: str) -> str:
    value: Any = version.manifest.params.get(key)
    if not isinstance(value, str):
        raise ModelIntegrityError(f"manifest is missing {key}")
    return value


def load_model(registry_root: Path, ref: str = "production") -> LoadedModel:
    """Load a registered model after verifying hashes; raises :class:`ModelIntegrityError`."""
    registry = ModelRegistry(registry_root)
    try:
        version = registry.get(MODEL_NAME, ref)
        files = registry.load_files(version)
    except RegistryError as exc:
        raise ModelIntegrityError(f"model registry verification failed: {exc}") from exc
    required = {"counts.npz", "tokenizer.json", "model.json"}
    if set(files) != required:
        raise ModelIntegrityError("model version has unexpected files")
    if sha256_hex(files["counts.npz"]) != _param(version, "counts_sha256"):
        raise ModelIntegrityError("counts.npz does not match the manifest")
    try:
        tokenizer = BPETokenizer.from_json(
            files["tokenizer.json"].decode("utf-8"),
            expected_sha256=_param(version, "tokenizer_sha256"),
        )
        arrays = loads_npz(files["counts.npz"])
        prompt_lengths = np.asarray(arrays.pop("prompt_lengths"), dtype=np.int64)
        lm = KneserNeyLM.from_arrays(arrays)
    except (
        TokenizerIntegrityError,
        LMFormatError,
        ValueError,
        KeyError,
        UnicodeDecodeError,
    ) as exc:
        raise ModelIntegrityError(f"model files failed verification: {type(exc).__name__}") from exc
    if tokenizer.vocab.sha256 != _param(version, "vocab_sha256"):
        raise ModelIntegrityError("tokenizer vocabulary does not match the manifest")
    if lm.vocab_size != tokenizer.vocab.size or lm.bos_id != tokenizer.vocab.eos_id:
        raise ModelIntegrityError("model and tokenizer disagree")
    return LoadedModel(lm, tokenizer, version, prompt_lengths)
