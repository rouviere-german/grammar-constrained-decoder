"""Data contracts for the synthetic corpus, with PII-safe failure reporting.

Checks (all run, then summarized as ``(check, count)`` pairs; offending values are never copied
into the error, the same rule as the template's tabular contracts):

- ``json_parses`` / ``validates``: every JSON record parses and validates against its bundled
  schema with ``jsonschema`` (Draft 2020-12). The bundled schemas are trusted repository files, so
  ``jsonschema``'s own ``pattern`` implementation is acceptable here; untrusted schemas go through
  ``schema/validate.py`` instead.
- ``canonical``: the completion is exactly the canonical serialization the grammar emits.
- ``schema_known`` / ``split``: training records only use train-split schemas; no held-out schema
  id or description appears in the training corpus.
- ``split_disjoint``: the train and eval schema sets share no id and no schema content hash.
- ``text_records``: free-text records carry no schema id and no prompt.
- ``min_docs``: every train schema has at least the requested number of documents.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from jsonschema import Draft202012Validator

from grammar_constrained_decoder.core.canonical import canonical_json, sha256_hex
from grammar_constrained_decoder.schema.canonical import CanonicalError, canonical_dumps
from grammar_constrained_decoder.schema.ir import parse_schema
from grammar_constrained_decoder.training.corpus import BundledSchema, CorpusRecord

__all__ = [
    "CONTRACT_CHECKS",
    "ContractReport",
    "DataContractError",
    "check_split",
    "validate_corpus",
]

CONTRACT_CHECKS = (
    "json_parses",
    "validates",
    "canonical",
    "schema_known",
    "split",
    "split_disjoint",
    "duplicate_ids",
    "text_records",
    "min_docs",
)


class DataContractError(ValueError):
    """The corpus violated its contract. ``failures`` holds (check, count) summaries."""

    def __init__(self, failures: list[dict[str, str | int]]) -> None:
        total = sum(int(item["count"]) for item in failures)
        super().__init__(
            f"data contract violated: {total} failure(s) across {len(failures)} check(s)"
        )
        self.failures = failures


@dataclass
class ContractReport:
    records: int = 0
    json_records: int = 0
    text_records: int = 0
    per_schema: dict[str, int] = field(default_factory=dict)
    checks: tuple[str, ...] = CONTRACT_CHECKS


def check_split(schemas: Sequence[BundledSchema]) -> Counter[str]:
    """Failures of the train/eval split itself (ids and schema contents must be disjoint)."""
    failures: Counter[str] = Counter()
    train = [item for item in schemas if item.split == "train"]
    held_out = [item for item in schemas if item.split == "eval"]
    if {item.id for item in train} & {item.id for item in held_out}:
        failures["split_disjoint"] += 1
    train_hashes = {sha256_hex(canonical_json(item.schema)) for item in train}
    if any(sha256_hex(canonical_json(item.schema)) in train_hashes for item in held_out):
        failures["split_disjoint"] += 1
    if len({item.id for item in schemas}) != len(schemas):
        failures["duplicate_ids"] += 1
    return failures


def validate_corpus(
    records: Iterable[CorpusRecord],
    schemas: Sequence[BundledSchema],
    *,
    allowed_split: str = "train",
    min_docs_per_schema: int = 1,
) -> ContractReport:
    """Validate every record; raise :class:`DataContractError` with summarized failures."""
    by_id: Mapping[str, BundledSchema] = {item.id: item for item in schemas}
    validators = {item.id: Draft202012Validator(item.schema) for item in schemas}
    irs = {item.id: parse_schema(item.schema) for item in schemas}
    forbidden = {item.description for item in schemas if item.split != allowed_split}
    failures: Counter[str] = check_split(schemas)
    report = ContractReport()
    counts: Counter[str] = Counter()
    for record in records:
        report.records += 1
        if record.kind == "text":
            report.text_records += 1
            if record.schema_id is not None or record.prompt:
                failures["text_records"] += 1
            continue
        report.json_records += 1
        bundled = by_id.get(record.schema_id or "")
        if bundled is None:
            failures["schema_known"] += 1
            continue
        if bundled.split != allowed_split or any(text in record.prompt for text in forbidden):
            failures["split"] += 1
        counts[bundled.id] += 1
        try:
            value = json.loads(record.completion)
        except json.JSONDecodeError:
            failures["json_parses"] += 1
            continue
        if not validators[bundled.id].is_valid(value):
            failures["validates"] += 1
            continue
        try:
            if canonical_dumps(irs[bundled.id], value) != record.completion:
                failures["canonical"] += 1
        except CanonicalError:
            failures["canonical"] += 1
    for item in schemas:
        if item.split == allowed_split and counts[item.id] < min_docs_per_schema:
            failures["min_docs"] += 1
    report.per_schema = dict(sorted(counts.items()))
    if failures:
        raise DataContractError(
            [{"check": check, "count": count} for check, count in sorted(failures.items())]
        )
    return report
