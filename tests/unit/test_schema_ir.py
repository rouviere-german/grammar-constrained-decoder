"""Schema subset parser: supported keywords, explicit rejections, limits and the support matrix."""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from grammar_constrained_decoder.automata.limits import (
    CompileLimitError,
    CompileLimits,
    SchemaError,
)
from grammar_constrained_decoder.schema.ir import (
    ANNOTATION_KEYWORDS,
    REJECTED_KEYWORDS,
    ROOT_KEYWORDS,
    SUPPORTED_KEYWORDS,
    AnyOf,
    Enum,
    Integer,
    Number,
    Object,
    Ref,
    String,
    json_type_of,
    parse_schema,
)

pytestmark = pytest.mark.unit

_ROW = re.compile(r"^\| `([^`]+)` \| (supported|annotation|root-only|rejected) \|", re.M)


def test_support_matrix_matches_the_parser(repo_root: Path) -> None:
    text = (repo_root / "SCHEMA_SUPPORT.md").read_text(encoding="utf-8")
    rows = dict(_ROW.findall(text))
    by_status: dict[str, set[str]] = {}
    for keyword, status in rows.items():
        by_status.setdefault(status, set()).add(keyword)
    assert by_status["supported"] == set(SUPPORTED_KEYWORDS)
    assert by_status["annotation"] == set(ANNOTATION_KEYWORDS)
    assert by_status["root-only"] == set(ROOT_KEYWORDS)
    assert by_status["rejected"] == set(REJECTED_KEYWORDS)


@pytest.mark.parametrize("keyword", sorted(REJECTED_KEYWORDS))
def test_rejected_keywords_raise_explicit_errors(keyword: str) -> None:
    with pytest.raises(SchemaError) as info:
        parse_schema({"type": "string", keyword: {}})
    assert f"'{keyword}'" in str(info.value)
    assert info.value.reason == "unsupported"
    assert info.value.path == f"/{keyword}"


def test_unknown_keywords_are_rejected_with_a_sanitized_name() -> None:
    with pytest.raises(SchemaError, match="unknown keyword 'x-extension'"):
        parse_schema({"type": "string", "x-extension": 1})
    with pytest.raises(SchemaError) as info:
        parse_schema({"type": "string", "a\nb" + "z" * 100: 1})
    assert "\n" not in str(info.value)
    assert len(str(info.value)) < 200


def test_annotations_are_ignored() -> None:
    schema = dict.fromkeys(ANNOTATION_KEYWORDS, "x") | {"type": "integer", "minimum": 3}
    ir = parse_schema(schema)
    assert ir.root == Integer(3, None)


def test_root_only_keywords() -> None:
    ir = parse_schema(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:example",
            "$defs": {"leaf": {"type": "null"}},
            "definitions": {"other": {"type": "boolean"}},
            "anyOf": [{"$ref": "#/$defs/leaf"}, {"$ref": "#/definitions/other"}],
        }
    )
    assert set(ir.definitions) == {"leaf", "other"}
    with pytest.raises(SchemaError, match="only supported at the root"):
        parse_schema({"type": "array", "items": {"$defs": {}, "type": "null"}})


def test_root_identifiers_must_be_strings() -> None:
    """``jsonschema`` crashes on a non-string ``$id`` (it calls ``.rstrip`` on it), so the parser
    rejects both identifiers unless they are short strings."""
    for keyword, value in (("$id", 5), ("$schema", ["x"]), ("$id", "u" * 600)):
        with pytest.raises(SchemaError) as info:
            parse_schema({keyword: value, "type": "null"})
        assert info.value.reason == "invalid"
        assert info.value.path == f"/{keyword}"


def test_type_lists_become_unions() -> None:
    ir = parse_schema({"type": ["string", "integer"], "maxLength": 2, "minimum": 1})
    assert ir.root == AnyOf((String(0, 2, None), Integer(1, None)))
    with pytest.raises(SchemaError, match="does not apply to type integer"):
        parse_schema({"type": "integer", "maxLength": 2})


def test_open_objects_are_rejected() -> None:
    for value in (True, {"type": "string"}):
        with pytest.raises(SchemaError, match="closed objects"):
            parse_schema({"type": "object", "additionalProperties": value})
    ir = parse_schema({"type": "object", "properties": {"a": {"type": "null"}}})
    assert isinstance(ir.root, Object)


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (True, "root schema must be a JSON object"),
        ({"type": "nope"}, "JSON type name"),
        ({"type": []}, "JSON type name"),
        ({"type": ["null", "null"]}, "JSON type name"),
        ({}, "must declare"),
        ({"type": "array"}, "must declare 'items'"),
        ({"type": "array", "items": {"type": "null"}, "uniqueItems": True}, "uniqueItems"),
        ({"type": "array", "items": True}, "boolean schemas"),
        ({"type": "string", "maxLength": -1}, "non-negative integer"),
        ({"type": "string", "maxLength": True}, "non-negative integer"),
        ({"type": "string", "pattern": 5}, "must be a string"),
        ({"type": "integer", "minimum": "3"}, "must be a number"),
        ({"type": "number", "maximum": float("inf")}, "must be finite"),
        ({"type": "object", "properties": []}, "must be an object"),
        ({"type": "object", "required": "a"}, "distinct strings"),
        ({"type": "object", "properties": {}, "required": ["a"]}, "declared in 'properties'"),
        ({"enum": []}, "non-empty array"),
        ({"enum": [1], "const": 1}, "combined with 'type'"),
        ({"enum": [1], "minimum": 1}, "combined with 'type'"),
        ({"enum": ["a"], "type": "integer"}, "no enum/const value"),
        ({"anyOf": []}, "non-empty array"),
        ({"anyOf": [{"type": "null"}], "type": "null"}, "cannot be combined"),
        ({"$ref": "#/$defs/missing"}, "undefined definition"),
        ({"$ref": "https://example.com/schema"}, "only local references"),
        ({"$ref": 3}, "must be a string"),
        ({"$ref": "#", "type": "null"}, "cannot be combined"),
        ({"$defs": [], "type": "null"}, "must be an object"),
        ({"$defs": {"bad name": {"type": "null"}}, "type": "null"}, "invalid definition name"),
        ({"type": "integer", 3: 4}, "keywords must be strings"),
    ],
)
def test_invalid_keyword_values_are_rejected(schema: Any, message: str) -> None:
    with pytest.raises(SchemaError, match=re.escape(message)):
        parse_schema(schema)


def test_duplicate_definition_names_across_spellings() -> None:
    with pytest.raises(SchemaError, match="duplicate definition"):
        parse_schema(
            {
                "$defs": {"a": {"type": "null"}},
                "definitions": {"a": {"type": "null"}},
                "type": "null",
            }
        )


def test_integer_bounds_are_normalized_to_inclusive_integers() -> None:
    ir = parse_schema({"type": "integer", "minimum": 1.5, "exclusiveMaximum": 10, "maximum": 20})
    assert ir.root == Integer(2, 9)
    ir = parse_schema({"type": "integer", "exclusiveMinimum": -3.5, "maximum": 4.9})
    assert ir.root == Integer(-3, 4)


def test_number_bounds_keep_the_tighter_constraint() -> None:
    ir = parse_schema({"type": "number", "minimum": 1, "exclusiveMinimum": 1, "maximum": 0.25})
    assert ir.root == Number(Decimal(1), True, Decimal("0.25"), False)
    ir = parse_schema({"type": "number", "minimum": 5, "exclusiveMinimum": 1})
    assert ir.root == Number(Decimal(5), False, None, False)


def test_enum_values_are_filtered_by_type_and_deduplicated() -> None:
    ir = parse_schema({"enum": [1, 1.0, "1", True, None, 2.5], "type": ["integer", "null"]})
    assert isinstance(ir.root, Enum)
    assert ir.root.literals == ("1", "1.0", "null")
    numbers = parse_schema({"enum": [3, 0.5], "type": "number"})
    assert isinstance(numbers.root, Enum)
    assert numbers.root.literals == ("3", "0.5")
    assert json_type_of(2.0) == "integer"
    assert json_type_of(2.5) == "number"
    with pytest.raises(SchemaError):
        json_type_of(object())


def test_refs_parse_to_named_references() -> None:
    ir = parse_schema({"type": "array", "items": {"$ref": "#"}})
    assert ir.root.items == Ref("#")  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("schema", "limits", "reason"),
    [
        ({"enum": list(range(10))}, CompileLimits(max_enum_values=5), "enum_size"),
        ({"const": "x" * 100}, CompileLimits(max_literal_bytes=50), "literal_size"),
        ({"type": "string", "maxLength": 1000}, CompileLimits(), "repeat_bound"),
        ({"type": "string", "pattern": "a" * 300}, CompileLimits(), "pattern_length"),
        ({"type": "number", "maximum": 1e300}, CompileLimits(), "number_digits"),
        (
            {"type": "object", "properties": {f"p{i}": {"type": "null"} for i in range(70)}},
            CompileLimits(),
            "properties",
        ),
        ({"anyOf": [{"type": "null"}] * 50}, CompileLimits(max_schema_nodes=20), "schema_size"),
    ],
)
def test_parse_time_limits(schema: dict[str, Any], limits: CompileLimits, reason: str) -> None:
    with pytest.raises(CompileLimitError) as info:
        parse_schema(schema, limits)
    assert info.value.reason == reason


def test_static_nesting_depth_is_capped() -> None:
    schema: dict[str, Any] = {"type": "null"}
    for _ in range(8):
        schema = {"type": "array", "items": schema}
    with pytest.raises(CompileLimitError) as info:
        parse_schema(schema)
    assert info.value.reason == "depth"
    parse_schema(schema, CompileLimits(max_depth=8))
