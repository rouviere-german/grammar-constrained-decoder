"""Adversarial schemas: every one is rejected with a typed reason, within the compile timeout."""

from __future__ import annotations

import time
from typing import Any

import pytest

from conftest import Trained
from grammar_constrained_decoder.automata.limits import CompileError, CompileLimits
from grammar_constrained_decoder.decoding.compiler import GrammarCompiler

pytestmark = pytest.mark.integration

TIMEOUT = 2.0


def _deep(levels: int) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "null"}
    for _ in range(levels):
        schema = {"type": "array", "items": schema}
    return schema


def _ref_chain(links: int) -> dict[str, Any]:
    definitions: dict[str, Any] = {f"d{i}": {"$ref": f"#/$defs/d{i + 1}"} for i in range(links)}
    definitions[f"d{links}"] = {"type": "null"}
    return {"$defs": definitions, "$ref": "#/$defs/d0"}


def _nested_any_of(levels: int) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "null"}
    for _ in range(levels):
        schema = {"anyOf": [schema, {"type": "boolean"}]}
    return schema


ADVERSARIAL: list[tuple[str, dict[str, Any], set[str]]] = [
    ("deep nesting", _deep(200), {"schema_depth"}),
    ("deep container nesting", _deep(20), {"depth"}),
    ("huge enum", {"enum": [f"value-{i}" for i in range(10_000)]}, {"enum_size"}),
    ("long pattern", {"type": "string", "pattern": "a" * 5_000}, {"pattern_length"}),
    ("exponential DFA", {"type": "string", "pattern": "^(a|b)*a(a|b){20}$"}, {"dfa_states"}),
    ("nested counters", {"type": "string", "pattern": "^((a{0,90}){0,90}){0,90}$"},
     {"nfa_states", "dfa_states", "subset_size", "timeout"}),
    ("huge maxLength", {"type": "string", "maxLength": 10**9}, {"repeat_bound"}),
    ("huge bound", {"type": "integer", "maximum": 10**40}, {"number_digits"}),
    ("wide object",
     {"type": "object", "properties": {f"p{i}": {"type": "null"} for i in range(500)}},
     {"properties"}),
    ("many subschemas", {"anyOf": [{"type": "null"}] * 5_000}, {"schema_size"}),
    ("lookahead", {"type": "string", "pattern": "(?=a)a"}, {"pattern"}),
    ("backreference", {"type": "string", "pattern": "(a+)\\1"}, {"pattern"}),
    ("empty language", {"type": "integer", "minimum": 3, "maximum": 2}, {"empty_language"}),
    ("recursive requirement",
     {"type": "object", "properties": {"x": {"$ref": "#"}}, "required": ["x"]}, {"empty_language"}),
    ("wide string product",
     {"type": "object", "properties": {
         f"s{i}": {"type": "string", "maxLength": 200, "pattern": "^(x|y)*x(x|y){6}$"}
         for i in range(40)}},
     {"dfa_states", "nfa_states", "timeout"}),
    ("lone surrogate in enum", {"enum": ["ok", "\ud800"]}, {"invalid"}),
    ("lone surrogate in a property name",
     {"type": "object", "properties": {"\udc00": {"type": "null"}}}, {"invalid"}),
    ("lone surrogate in a pattern", {"type": "string", "pattern": "a\ud800"}, {"invalid"}),
    ("lone surrogate in an annotation", {"type": "null", "description": "\ud800"}, {"invalid"}),
    ("non-finite annotation", {"type": "null", "default": float("nan")}, {"invalid"}),
    ("reference chain", _ref_chain(600), {"ref_depth"}),
    ("nested anyOf", _nested_any_of(400), {"schema_depth"}),
]  # fmt: skip


@pytest.mark.parametrize(
    ("name", "schema", "reasons"), ADVERSARIAL, ids=[case[0] for case in ADVERSARIAL]
)
def test_adversarial_schemas_are_rejected_cleanly_within_the_timeout(
    trained: Trained, name: str, schema: dict[str, Any], reasons: set[str]
) -> None:
    compiler = GrammarCompiler(
        trained.model.tokenizer.vocab,
        limits=CompileLimits(timeout_seconds=TIMEOUT, max_dfa_states=20_000),
    )
    started = time.perf_counter()
    with pytest.raises(CompileError) as info:
        compiler.compile(schema)
    elapsed = time.perf_counter() - started
    assert info.value.reason in reasons, (name, info.value.reason, str(info.value))
    assert elapsed < TIMEOUT + 0.5, (name, elapsed)
    assert len(str(info.value)) < 400  # errors stay short and never echo the schema
