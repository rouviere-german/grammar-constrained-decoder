"""Synthetic corpus generation and its data contracts."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from grammar_constrained_decoder.ml.contracts import DataContractError, check_split, validate_corpus
from grammar_constrained_decoder.schema.canonical import matches
from grammar_constrained_decoder.schema.ir import parse_schema
from grammar_constrained_decoder.schema.sampler import InstanceSampler, sample_dfa, string_dfa
from grammar_constrained_decoder.training.corpus import (
    CorpusRecord,
    corpus_jsonl,
    corpus_sha256,
    free_text,
    generate_corpus,
    load_bundled_schemas,
    prompt_for,
)

pytestmark = pytest.mark.unit

SCHEMAS = load_bundled_schemas()
TRAIN = [item for item in SCHEMAS if item.split == "train"]
EVAL = [item for item in SCHEMAS if item.split == "eval"]


def test_bundled_schema_split_is_18_train_and_12_held_out() -> None:
    assert len(TRAIN) == 18
    assert len(EVAL) == 12
    assert not check_split(SCHEMAS)
    for item in SCHEMAS:
        parse_schema(item.schema)  # every bundled schema is in the supported subset


def test_generation_is_deterministic_and_passes_the_contract() -> None:
    first = generate_corpus(TRAIN[:4], docs_per_schema=15, free_text_docs=10, seed=3)
    second = generate_corpus(TRAIN[:4], docs_per_schema=15, free_text_docs=10, seed=3)
    assert corpus_sha256(first) == corpus_sha256(second)
    other = generate_corpus(TRAIN[:4], docs_per_schema=15, free_text_docs=10, seed=4)
    assert corpus_sha256(first) != corpus_sha256(other)
    report = validate_corpus(first, [*TRAIN[:4], *EVAL], min_docs_per_schema=15)
    assert report.json_records == 60
    assert report.text_records == 10
    assert set(report.per_schema) == {item.id for item in TRAIN[:4]}
    lines = corpus_jsonl(first).decode().splitlines()
    assert len(lines) == 70
    assert all(json.loads(line)["kind"] in {"json", "text"} for line in lines)


def test_violations_are_summarized_without_the_offending_values() -> None:
    secret = "hunter2-SECRET-VALUE"
    bundled = TRAIN[3]  # log_event
    records = generate_corpus([bundled], docs_per_schema=5, free_text_docs=2, seed=0)
    bad = [
        CorpusRecord(kind="json", schema_id=bundled.id, prompt=prompt_for(bundled),
                     completion=json.dumps({"level": secret, "message": "x"})),
        CorpusRecord(kind="json", schema_id=bundled.id, prompt=prompt_for(bundled),
                     completion="{oops"),
        CorpusRecord(kind="json", schema_id=bundled.id, prompt=prompt_for(bundled),
                     completion='{"message": "x", "level": "info"}'),
        CorpusRecord(kind="json", schema_id="unknown_schema", prompt="", completion="{}"),
        CorpusRecord(kind="text", schema_id=bundled.id, prompt="", completion="text"),
    ]  # fmt: skip
    with pytest.raises(DataContractError) as info:
        validate_corpus([*records, *bad], SCHEMAS, min_docs_per_schema=5)
    checks = {item["check"]: item["count"] for item in info.value.failures}
    assert checks["validates"] == 1
    assert checks["json_parses"] == 1
    assert checks["canonical"] == 1
    assert checks["schema_known"] == 1
    assert checks["text_records"] == 1
    assert checks["min_docs"] == len(TRAIN) - 1
    assert secret not in str(info.value)
    assert secret not in json.dumps(info.value.failures)


def test_held_out_schemas_cannot_leak_into_training() -> None:
    leaked = generate_corpus([EVAL[0]], docs_per_schema=3, free_text_docs=0, seed=1)
    with pytest.raises(DataContractError) as info:
        validate_corpus(leaked, SCHEMAS)
    assert {item["check"] for item in info.value.failures} >= {"split"}
    copied = dataclasses.replace(EVAL[0], id="copy_of_eval", split="train")
    failures = check_split([*SCHEMAS, copied])
    assert failures["split_disjoint"] == 1
    duplicate = dataclasses.replace(TRAIN[0], split="eval")
    assert check_split([*SCHEMAS, duplicate])["duplicate_ids"] == 1


def test_sampler_produces_valid_instances_for_every_bundled_schema() -> None:
    rng = np.random.default_rng(11)
    for item in SCHEMAS:
        ir = parse_schema(item.schema)
        sampler = InstanceSampler(ir, rng)
        for _ in range(5):
            assert matches(ir, sampler.sample()), item.id


def test_string_dfa_sampling_respects_patterns_and_lengths() -> None:
    rng = np.random.default_rng(0)
    dfa = string_dfa(2, 6, "^[A-Z]{2}[0-9]+$")
    for _ in range(30):
        data = sample_dfa(dfa, rng)
        assert data is not None
        text = data.decode()
        assert 3 <= len(text) <= 6
        assert text[:2].isupper()
        assert text[2:].isdigit()
    assert sample_dfa(string_dfa(3, 2, None), rng) is None
    assert free_text(rng).endswith(".")
