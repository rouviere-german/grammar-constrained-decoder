"""Training pipeline: reproducibility, manifest contents and verified loading of artifacts."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import TEST_TRAIN_CONFIG, Trained
from grammar_constrained_decoder.core.canonical import sha256_hex
from grammar_constrained_decoder.ml.manifest import RunManifest
from grammar_constrained_decoder.ml.registry import ModelRegistry
from grammar_constrained_decoder.training.pipeline import (
    MODEL_NAME,
    ModelIntegrityError,
    load_model,
    train_pipeline,
)

pytestmark = pytest.mark.integration


def test_training_is_fast_and_records_a_complete_manifest(trained: Trained) -> None:
    result = trained.result
    assert result.seconds["total"] < 60
    manifest = RunManifest.load(trained.root / "manifest.json")
    params = manifest.params
    assert manifest.seed == 0
    assert {dataset.name for dataset in manifest.datasets} == {"corpus.jsonl", "eval_corpus.jsonl"}
    assert params["corpus_sha256"] == sha256_hex(result.corpus_path.read_bytes())
    assert params["tokenizer_sha256"] == trained.model.tokenizer.sha256
    assert params["vocab_sha256"] == trained.model.tokenizer.vocab.sha256
    assert len(params["train_schema_ids"]) == 18  # type: ignore[arg-type]
    assert set(params["train_schema_ids"]).isdisjoint(params["eval_schema_ids"])  # type: ignore[arg-type]
    assert set(manifest.packages) == {"numpy", "tokenizers", "jsonschema", "pydantic"}
    assert result.metrics["heldout_perplexity"] > result.metrics["train_perplexity"]
    assert trained.model.prompt_lengths.size > 0


def test_training_is_reproducible(trained: Trained, tmp_path: Path) -> None:
    again = train_pipeline(tmp_path, TEST_TRAIN_CONFIG)
    first, second = trained.result.version.manifest.params, again.version.manifest.params
    for key in ("corpus_sha256", "eval_corpus_sha256", "tokenizer_sha256", "counts_sha256"):
        assert first[key] == second[key], key


def _copy(trained: Trained, tmp_path: Path) -> Path:
    target = tmp_path / "registry"
    shutil.copytree(trained.root / "registry", target)
    return target


def _blob(root: Path, trained: Trained, name: str) -> Path:
    digest = trained.result.version.files[name]
    return root / "blobs" / "sha256" / digest[:2] / digest


def test_tampered_tokenizer_blob_is_refused(trained: Trained, tmp_path: Path) -> None:
    root = _copy(trained, tmp_path)
    path = _blob(root, trained, "tokenizer.json")
    path.write_bytes(path.read_bytes().replace(b'"<|eos|>"', b'"<|EOS|>"', 1))
    with pytest.raises(ModelIntegrityError, match="registry verification"):
        load_model(root)


def test_a_consistently_rehashed_but_foreign_tokenizer_is_refused(
    trained: Trained, tmp_path: Path
) -> None:
    """Someone who can rewrite the registry re-registers a different tokenizer under the same
    manifest: the manifest's tokenizer hash no longer matches, so loading fails."""
    root = _copy(trained, tmp_path)
    registry = ModelRegistry(root)
    version = registry.get(MODEL_NAME, "production")
    files = registry.load_files(version)
    spec = json.loads(files["tokenizer.json"])
    spec["model"]["dropout"] = 0.1
    files["tokenizer.json"] = json.dumps(spec).encode()
    forged = registry.register(MODEL_NAME, files, version.manifest, metrics=version.metrics)
    registry.set_alias(MODEL_NAME, "production", forged.digest)
    with pytest.raises(ModelIntegrityError):
        load_model(root)


def test_pickled_arrays_are_refused_even_with_matching_hashes(
    trained: Trained, tmp_path: Path
) -> None:
    root = _copy(trained, tmp_path)
    registry = ModelRegistry(root)
    version = registry.get(MODEL_NAME, "production")
    files = registry.load_files(version)
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=True, meta=np.array([{"evil": 1}], dtype=object))
    files["counts.npz"] = buffer.getvalue()
    params = dict(version.manifest.params)
    params["counts_sha256"] = sha256_hex(files["counts.npz"])
    manifest = version.manifest.model_copy(update={"params": params})
    forged = registry.register(MODEL_NAME, files, manifest)
    registry.set_alias(MODEL_NAME, "production", forged.digest)
    with pytest.raises(ModelIntegrityError, match=r"UnsafeArtifactError|failed verification"):
        load_model(root)


def test_counts_that_do_not_match_the_manifest_are_refused(
    trained: Trained, tmp_path: Path
) -> None:
    root = _copy(trained, tmp_path)
    registry = ModelRegistry(root)
    version = registry.get(MODEL_NAME, "production")
    files = registry.load_files(version)
    files["counts.npz"] = files["counts.npz"] + b"\x00"
    forged = registry.register(MODEL_NAME, files, version.manifest)
    registry.set_alias(MODEL_NAME, "production", forged.digest)
    with pytest.raises(ModelIntegrityError, match="does not match the manifest"):
        load_model(root)


def test_missing_registry_is_an_integrity_error(tmp_path: Path) -> None:
    with pytest.raises(ModelIntegrityError):
        load_model(tmp_path / "nothing-here")
