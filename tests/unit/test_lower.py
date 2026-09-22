"""Schema -> regex lowering: soundness (every string of the language validates), completeness
(canonical serializations of arbitrary instances are accepted) and targeted cases."""

from __future__ import annotations

import itertools
import json
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis_jsonschema import from_schema
from jsonschema import Draft202012Validator

from grammar_constrained_decoder.automata.dfa import DFA, compile_regex
from grammar_constrained_decoder.automata.limits import CompileLimitError, CompileLimits
from grammar_constrained_decoder.schema.canonical import (
    CanonicalError,
    canonical_dumps,
    matches,
    nesting_depth,
)
from grammar_constrained_decoder.schema.ir import parse_schema
from grammar_constrained_decoder.schema.lower import escape_code_point, lower_schema
from grammar_constrained_decoder.schema.sampler import sample_dfa
from grammar_constrained_decoder.testing.strategies import json_schemas

pytestmark = pytest.mark.unit


def schema_dfa(schema: dict[str, Any], limits: CompileLimits | None = None) -> DFA:
    effective = limits or CompileLimits()
    return compile_regex(lower_schema(parse_schema(schema, effective), effective))


@settings(max_examples=40, suppress_health_check=list(HealthCheck))
@given(json_schemas(), st.integers(0, 2**32 - 1))
def test_every_string_in_the_language_is_a_valid_instance(
    schema: dict[str, Any], seed: int
) -> None:
    dfa = schema_dfa(schema)
    validator = Draft202012Validator(schema)
    rng = np.random.default_rng(seed)
    for _ in range(8):
        sample = sample_dfa(dfa, rng, soft_limit=40)
        assert sample is not None
        value = json.loads(sample.decode("utf-8"))
        assert validator.is_valid(value), (schema, sample)
        # Numbers may carry trailing fractional zeros; the canonical form is in the language too.
        assert dfa.accepts(canonical_dumps(parse_schema(schema), value).encode("utf-8"))


@settings(max_examples=40, suppress_health_check=list(HealthCheck))
@given(json_schemas(), st.data())
def test_canonical_serializations_of_any_instance_are_accepted(
    schema: dict[str, Any], data: st.DataObject
) -> None:
    ir = parse_schema(schema)
    dfa = compile_regex(lower_schema(ir))
    for _ in range(4):
        instance = data.draw(from_schema(schema))
        text = canonical_dumps(ir, instance)
        assert dfa.accepts(text.encode("utf-8")), (schema, text)
        assert matches(ir, instance)


def test_optional_members_in_declaration_order() -> None:
    names = ["a", "b", "c", "d"]
    schema = {
        "type": "object",
        "properties": {name: {"type": "null"} for name in names},
        "required": ["c"],
        "additionalProperties": False,
    }
    dfa = schema_dfa(schema)
    for mask in itertools.product([False, True], repeat=4):
        present = [name for name, keep in zip(names, mask, strict=True) if keep]
        text = "{" + ",".join(f'"{name}":null' for name in present) + "}"
        assert dfa.accepts(text.encode()) == ("c" in present), text
    assert not dfa.accepts(b'{"c":null,"a":null}')  # out of declaration order
    assert not dfa.accepts(b'{"c":null,}')
    assert not dfa.accepts(b'{ "c":null}')  # no insignificant whitespace


def test_string_lengths_count_code_points_not_bytes() -> None:
    dfa = schema_dfa({"type": "string", "minLength": 2, "maxLength": 3})
    for text, ok in [("ab", True), ("中文字", True), ("😀😀", True), ("a", False), ("abcd", False),
                     ('a"b', True), ("\n\t", True), ("\x01\x02", True)]:  # fmt: skip
        assert dfa.accepts(json.dumps(text, ensure_ascii=False).encode()) == ok, text
    assert not dfa.accepts(b'"\\u0061b"')  # non-canonical escape
    assert not dfa.accepts(b'"\xed\xa0\x80a"')  # encoded surrogate


def test_escapes_are_the_canonical_json_escapes() -> None:
    for code in range(0x20):
        assert json.dumps(chr(code))[1:-1] == escape_code_point(code)
    assert escape_code_point(0x22) == '\\"'
    assert escape_code_point(0x5C) == "\\\\"
    assert escape_code_point(0xE9) == "é"


def test_array_bounds_are_exact() -> None:
    dfa = schema_dfa({"type": "array", "items": {"type": "boolean"}, "minItems": 1, "maxItems": 3})
    for count in range(5):
        text = "[" + ",".join(["true"] * count) + "]"
        assert dfa.accepts(text.encode()) == (1 <= count <= 3)
    empty_items = schema_dfa({"type": "array", "items": {"type": "null"}, "maxItems": 0})
    assert empty_items.accepts(b"[]")
    assert not empty_items.accepts(b"[null]")


def test_enum_and_const_emit_canonical_literals() -> None:
    dfa = schema_dfa({"enum": [{"b": 1, "a": [True, None]}, "é", 2.5]})
    assert dfa.accepts(b'{"a":[true,null],"b":1}')
    assert dfa.accepts('"é"'.encode())
    assert dfa.accepts(b"2.5")
    assert not dfa.accepts(b'{"b":1,"a":[true,null]}')
    assert schema_dfa({"const": None}).accepts(b"null")


def test_any_of_is_a_union() -> None:
    dfa = schema_dfa(
        {"anyOf": [{"type": "integer", "maximum": 3}, {"type": "string", "maxLength": 1}]}
    )
    assert dfa.accepts(b"-7")
    assert dfa.accepts(b'"x"')
    assert not dfa.accepts(b"4")
    assert not dfa.accepts(b'"xy"')


def test_recursive_refs_are_unrolled_to_the_depth_limit() -> None:
    tree = {
        "type": "object",
        "properties": {
            "v": {"type": "integer", "minimum": 0, "maximum": 9},
            "kids": {"type": "array", "items": {"$ref": "#"}, "maxItems": 2},
        },
        "required": ["v"],
        "additionalProperties": False,
    }
    limits = CompileLimits(max_depth=4)
    dfa = schema_dfa(tree, limits)
    ir = parse_schema(tree, limits)
    validator = Draft202012Validator(tree)
    two_levels = {"v": 1, "kids": [{"v": 2}, {"v": 3, "kids": []}]}
    assert nesting_depth(two_levels) == 4
    assert dfa.accepts(canonical_dumps(ir, two_levels).encode())
    three_levels = {"v": 1, "kids": [{"v": 2, "kids": [{"v": 3}]}]}
    assert nesting_depth(three_levels) == 5
    assert not dfa.accepts(canonical_dumps(ir, three_levels).encode())
    rng = np.random.default_rng(0)
    for _ in range(50):
        sample = sample_dfa(dfa, rng, soft_limit=60)
        assert sample is not None
        value = json.loads(sample)
        assert validator.is_valid(value)
        assert nesting_depth(value) <= 4


def test_definition_cycles_without_containers_take_the_least_fixed_point() -> None:
    schema = {
        "$defs": {
            "a": {"anyOf": [{"$ref": "#/$defs/b"}, {"type": "null"}]},
            "b": {"anyOf": [{"$ref": "#/$defs/a"}, {"type": "boolean"}]},
        },
        "type": "array",
        "items": {"anyOf": [{"$ref": "#/$defs/a"}, {"$ref": "#/$defs/b"}]},
    }
    dfa = schema_dfa(schema)
    assert dfa.accepts(b"[null,true,false]")
    assert not dfa.accepts(b"[1]")


def test_non_recursive_depth_violation_is_an_error_but_recursion_is_cut() -> None:
    nested = {
        "type": "array",
        "items": {"type": "array", "items": {"type": "array", "items": {"type": "null"}}},
    }
    with pytest.raises(CompileLimitError):
        schema_dfa(nested, CompileLimits(max_depth=2))
    must_recurse = {
        "type": "object",
        "properties": {"child": {"$ref": "#"}},
        "required": ["child"],
        "additionalProperties": False,
    }
    assert schema_dfa(must_recurse).is_empty  # every instance would be infinitely deep


def test_canonical_dumps_rejects_non_instances() -> None:
    ir = parse_schema(
        {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    )
    for value in [{"a": "x"}, {}, {"a": 1, "b": 2}, [1], {"a": True}]:
        with pytest.raises(CanonicalError):
            canonical_dumps(ir, value)
    assert canonical_dumps(ir, {"a": 3.0}) == '{"a":3}'
    numbers = parse_schema({"type": "number"})
    assert canonical_dumps(numbers, 1e-05) == "0.00001"
    assert canonical_dumps(numbers, 1.5e16) == "15000000000000000"
    assert canonical_dumps(numbers, -0.0) == "0.0"
    assert canonical_dumps(numbers, 7) == "7"
    with pytest.raises(CanonicalError):
        canonical_dumps(numbers, float("nan"))
    enum = parse_schema({"enum": [1.0, "x"]})
    assert canonical_dumps(enum, 1) == "1.0"
