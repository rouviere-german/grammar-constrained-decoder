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


ADVERSARIAL: list[tuple[str, dict[str, Any], set[str]]] = [
    ("deep nesting", _deep(200), {"depth"}),
    ("huge enum", {"enum": [f"value-{i}" for i in range(10_000)]}, {"enum_size"}),
    ("long pattern", {"type": "string", "pattern": "a" * 5_000}, {"pattern_length"}),
    ("exponential DFA", {"type": "string", "pattern": "^(a|b)*a(a|b){20}$"}, {"dfa_states"}),
    ("nested counters", {"type": "string", "pattern": "^((a{0,90}){0,90}){0,90}$"},
     {"nfa_states", "dfa_states", "timeout"}),
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
    assert elapsed < TIMEOUT + 1.5, (name, elapsed)
    assert len(str(info.value)) < 400  # errors stay short and never echo the schema
