"""JSON Schema subset -> typed intermediate representation (IR), with explicit rejections.

Schemas are parsed only here, by walking the decoded JSON value; nothing is evaluated, no remote
reference is fetched and Python's ``re`` never sees a ``pattern``. Every keyword is either
supported, an annotation that is ignored, or rejected with a :class:`SchemaError` that names the
keyword and a JSON Pointer to it. ``SCHEMA_SUPPORT.md`` lists the matrix, and a test keeps it in
sync with the tables below.

Subset (Draft 2020-12 vocabulary):

- ``type`` (one type or a list), ``enum``, ``const``, ``anyOf``;
- integer/number: ``minimum``, ``maximum``, ``exclusiveMinimum``, ``exclusiveMaximum``;
- string: ``minLength``, ``maxLength``, ``pattern`` (safe ECMA subset, see ``automata/pattern.py``);
- array: ``items`` (required), ``minItems``, ``maxItems``, ``uniqueItems: false``;
- object: ``properties``, ``required``, ``additionalProperties: false`` (objects are closed;
  omitting the keyword means closed too);
- ``$defs``/``definitions`` at the root and local ``$ref`` (``#`` or ``#/$defs/name``). Recursive
  references are unrolled up to the nesting depth limit when the language is lowered.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from grammar_constrained_decoder.automata.limits import (
    DEFAULT_LIMITS,
    CompileLimitError,
    CompileLimits,
    SchemaError,
)
from grammar_constrained_decoder.core.canonical import canonical_json, sha256_hex

__all__ = [
    "ANNOTATION_KEYWORDS",
    "REJECTED_KEYWORDS",
    "ROOT_KEYWORDS",
    "SUPPORTED_KEYWORDS",
    "AnyOf",
    "Array",
    "Boolean",
    "Enum",
    "Integer",
    "Node",
    "Null",
    "Number",
    "Object",
    "Property",
    "Ref",
    "SchemaIR",
    "String",
    "json_type_of",
    "parse_schema",
]

JSON_TYPES = ("null", "boolean", "integer", "number", "string", "array", "object")
ANNOTATION_KEYWORDS = frozenset(
    {
        "title",
        "description",
        "$comment",
        "examples",
        "default",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)
ROOT_KEYWORDS = frozenset({"$schema", "$id", "$defs", "definitions"})
_TYPE_KEYWORDS: Mapping[str, frozenset[str]] = {
    "null": frozenset(),
    "boolean": frozenset(),
    "integer": frozenset({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}),
    "number": frozenset({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}),
    "string": frozenset({"minLength", "maxLength", "pattern"}),
    "array": frozenset({"items", "minItems", "maxItems", "uniqueItems"}),
    "object": frozenset({"properties", "required", "additionalProperties"}),
}
SUPPORTED_KEYWORDS = frozenset(
    {"type", "enum", "const", "anyOf", "$ref"}.union(*_TYPE_KEYWORDS.values())
)
REJECTED_KEYWORDS: Mapping[str, str] = {
    "oneOf": "exclusive disjunction needs language difference; use anyOf",
    "allOf": "conjunction of subschemas is not supported; merge the constraints",
    "not": "negation is not supported",
    "if": "conditional subschemas are not supported",
    "then": "conditional subschemas are not supported",
    "else": "conditional subschemas are not supported",
    "format": "format assertions are not supported; use pattern",
    "multipleOf": "multipleOf is not supported",
    "patternProperties": "only declared properties are supported",
    "propertyNames": "only declared properties are supported",
    "minProperties": "property counts are not supported",
    "maxProperties": "property counts are not supported",
    "dependentRequired": "dependent properties are not supported",
    "dependentSchemas": "dependent schemas are not supported",
    "dependencies": "dependent properties are not supported",
    "unevaluatedProperties": "unevaluated keywords are not supported",
    "unevaluatedItems": "unevaluated keywords are not supported",
    "prefixItems": "tuple validation is not supported",
    "additionalItems": "tuple validation is not supported",
    "contains": "contains is not supported",
    "minContains": "contains is not supported",
    "maxContains": "contains is not supported",
    "contentEncoding": "content keywords are not supported",
    "contentMediaType": "content keywords are not supported",
    "contentSchema": "content keywords are not supported",
    "$anchor": "anchors are not supported; use $defs",
    "$dynamicRef": "dynamic references are not supported",
    "$dynamicAnchor": "dynamic references are not supported",
    "$recursiveRef": "dynamic references are not supported",
    "$recursiveAnchor": "dynamic references are not supported",
    "$vocabulary": "vocabularies are not supported",
    "nullable": "nullable is an OpenAPI keyword; use a type list with 'null'",
}
_DEF_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_MAX_PROPERTY_NAME = 256
ROOT_REF = "#"


@dataclass(frozen=True, slots=True)
class Null:
    pass


@dataclass(frozen=True, slots=True)
class Boolean:
    pass


@dataclass(frozen=True, slots=True)
class Integer:
    minimum: int | None = None
    maximum: int | None = None


@dataclass(frozen=True, slots=True)
class Number:
    minimum: Decimal | None = None
    exclusive_minimum: bool = False
    maximum: Decimal | None = None
    exclusive_maximum: bool = False


@dataclass(frozen=True, slots=True)
class String:
    min_length: int = 0
    max_length: int | None = None
    pattern: str | None = None


@dataclass(frozen=True, slots=True)
class Enum:
    values: tuple[Any, ...]
    literals: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Array:
    items: Node
    min_items: int = 0
    max_items: int | None = None


@dataclass(frozen=True, slots=True)
class Property:
    name: str
    schema: Node
    required: bool


@dataclass(frozen=True, slots=True)
class Object:
    properties: tuple[Property, ...]


@dataclass(frozen=True, slots=True)
class AnyOf:
    branches: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class Ref:
    name: str


Node = Null | Boolean | Integer | Number | String | Enum | Array | Object | AnyOf | Ref


@dataclass(frozen=True, slots=True)
class SchemaIR:
    root: Node
    definitions: Mapping[str, Node]
    sha256: str


def json_type_of(value: Any) -> str:  # noqa: PLR0911 - one return per JSON type
    """JSON Schema type name of a decoded JSON value ("integer" for integral numbers)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "integer" if math.isfinite(value) and value.is_integer() else "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise SchemaError("", f"value of type {type(value).__name__} is not JSON", reason="invalid")


def _safe_name(name: object) -> str:
    text = str(name)
    cleaned = "".join(char if char.isprintable() and char.isascii() else "?" for char in text)
    return cleaned[:40] + ("..." if len(cleaned) > 40 else "")


def _pointer(path: str, key: str | int) -> str:
    token = str(key).replace("~", "~0").replace("/", "~1")
    return f"{path}/{_safe_name(token)}"


def _literal(value: Any) -> str:
    """Canonical text of an enum/const value (compact, sorted keys, UTF-8, no NaN)."""
    try:
        return canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError("", "enum/const values must be JSON values", reason="invalid") from exc


class _Parser:
    def __init__(self, limits: CompileLimits) -> None:
        self.limits = limits
        self.nodes = 0
        self.definition_names: set[str] = set()

    def count(self, path: str) -> None:
        self.nodes += 1
        if self.nodes > self.limits.max_schema_nodes:
            raise CompileLimitError(
                "schema_size",
                f"{path or '/'}: schema has more than {self.limits.max_schema_nodes} subschemas",
            )

    def parse(self, schema: Any) -> SchemaIR:
        if not isinstance(schema, dict):
            raise SchemaError("", "the root schema must be a JSON object", reason="invalid")
        definitions_raw: dict[str, Any] = {}
        for keyword in ("$defs", "definitions"):
            if keyword in schema:
                block = schema[keyword]
                if not isinstance(block, dict):
                    raise SchemaError(f"/{keyword}", "must be an object", reason="invalid")
                for name in block:
                    if not isinstance(name, str) or not _DEF_NAME.fullmatch(name):
                        raise SchemaError(
                            _pointer(f"/{keyword}", name),
                            "invalid definition name",
                            reason="invalid",
                        )
                    if name in definitions_raw:
                        raise SchemaError(
                            _pointer(f"/{keyword}", name), "duplicate definition", reason="invalid"
                        )
                    definitions_raw[name] = block[name]
        self.definition_names = set(definitions_raw)
        definitions = {
            name: self.node(value, f"/$defs/{name}", depth=0, root=False)
            for name, value in sorted(definitions_raw.items())
        }
        root = self.node(schema, "", depth=0, root=True)
        return SchemaIR(
            root=root, definitions=definitions, sha256=sha256_hex(canonical_json(schema))
        )

    def node(  # noqa: PLR0912 - keyword dispatch reads best as one function
        self, schema: Any, path: str, *, depth: int, root: bool
    ) -> Node:
        self.count(path)
        if isinstance(schema, bool):
            raise SchemaError(path, "boolean schemas are not supported", reason="unsupported")
        if not isinstance(schema, dict):
            raise SchemaError(path, "a subschema must be a JSON object", reason="invalid")
        keywords = set(schema)
        for keyword in sorted(keywords, key=str):
            if not isinstance(keyword, str):
                raise SchemaError(path, "keywords must be strings", reason="invalid")
            if keyword in REJECTED_KEYWORDS:
                raise SchemaError(
                    _pointer(path, keyword),
                    f"keyword '{keyword}' is not supported: {REJECTED_KEYWORDS[keyword]}",
                )
            if keyword in ROOT_KEYWORDS and not root:
                raise SchemaError(
                    _pointer(path, keyword), f"keyword '{keyword}' is only supported at the root"
                )
            if keyword not in SUPPORTED_KEYWORDS | ANNOTATION_KEYWORDS | ROOT_KEYWORDS:
                raise SchemaError(
                    _pointer(path, keyword), f"unknown keyword '{_safe_name(keyword)}'"
                )
        constraints = keywords - ANNOTATION_KEYWORDS - ROOT_KEYWORDS
        if "$ref" in constraints:
            return self.ref(schema["$ref"], constraints, path)
        if "enum" in constraints or "const" in constraints:
            return self.enum(schema, constraints, path)
        if "anyOf" in constraints:
            if constraints != {"anyOf"}:
                raise SchemaError(
                    path, "anyOf cannot be combined with other constraints; move them into branches"
                )
            branches = schema["anyOf"]
            if not isinstance(branches, list) or not branches:
                raise SchemaError(
                    _pointer(path, "anyOf"), "must be a non-empty array", reason="invalid"
                )
            return AnyOf(
                tuple(
                    self.node(
                        branch, _pointer(_pointer(path, "anyOf"), index), depth=depth, root=False
                    )
                    for index, branch in enumerate(branches)
                )
            )
        if "type" not in constraints:
            raise SchemaError(
                path,
                "a subschema must declare 'type', 'enum', 'const', 'anyOf' or '$ref' "
                "(unconstrained values are not supported)",
            )
        types = self.types(schema["type"], path)
        allowed = frozenset({"type"}).union(*(_TYPE_KEYWORDS[name] for name in types))
        for keyword in sorted(constraints - allowed):
            raise SchemaError(
                _pointer(path, keyword),
                f"keyword '{keyword}' does not apply to type {'/'.join(types)}",
            )
        nodes = [self.typed(name, schema, path, depth) for name in types]
        return nodes[0] if len(nodes) == 1 else AnyOf(tuple(nodes))

    def types(self, value: Any, path: str) -> list[str]:
        names = [value] if isinstance(value, str) else value
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) and name in JSON_TYPES for name in names)
            or len(set(names)) != len(names)
        ):
            raise SchemaError(
                _pointer(path, "type"),
                "must be a JSON type name or a non-empty list of distinct type names",
                reason="invalid",
            )
        return list(names)

    def ref(self, target: Any, constraints: set[str], path: str) -> Ref:
        if constraints != {"$ref"}:
            raise SchemaError(path, "$ref cannot be combined with other constraints")
        if not isinstance(target, str):
            raise SchemaError(_pointer(path, "$ref"), "must be a string", reason="invalid")
        if target == ROOT_REF:
            return Ref(ROOT_REF)
        for prefix in ("#/$defs/", "#/definitions/"):
            if target.startswith(prefix):
                name = target[len(prefix) :]
                if name in self.definition_names:
                    return Ref(name)
                raise SchemaError(_pointer(path, "$ref"), "refers to an undefined definition")
        raise SchemaError(
            _pointer(path, "$ref"),
            "only local references ('#' or '#/$defs/<name>') are supported",
        )

    def enum(self, schema: Mapping[str, Any], constraints: set[str], path: str) -> Enum:
        extra = constraints - {"enum", "const", "type"}
        if extra or {"enum", "const"} <= constraints:
            raise SchemaError(path, "enum/const can only be combined with 'type'")
        if "const" in constraints:
            values: list[Any] = [schema["const"]]
        else:
            raw = schema["enum"]
            if not isinstance(raw, list) or not raw:
                raise SchemaError(
                    _pointer(path, "enum"), "must be a non-empty array", reason="invalid"
                )
            values = list(raw)
        if len(values) > self.limits.max_enum_values:
            raise CompileLimitError(
                "enum_size",
                f"{_pointer(path, 'enum')}: enum has more than "
                f"{self.limits.max_enum_values} values",
            )
        types = set(self.types(schema["type"], path)) if "type" in constraints else set(JSON_TYPES)
        kept: list[Any] = []
        literals: list[str] = []
        for value in values:
            kind = json_type_of(value)
            if kind not in types and not (kind == "integer" and "number" in types):
                continue
            text = _literal(value)
            if len(text.encode("utf-8")) > self.limits.max_literal_bytes:
                raise CompileLimitError(
                    "literal_size",
                    f"{path or '/'}: an enum/const value is longer than "
                    f"{self.limits.max_literal_bytes} bytes",
                )
            if text not in literals:
                literals.append(text)
                kept.append(value)
        if not kept:
            raise SchemaError(
                path, "no enum/const value matches the declared type", reason="empty_language"
            )
        return Enum(tuple(kept), tuple(literals))

    def typed(  # noqa: PLR0911 - one return per JSON type
        self, name: str, schema: Mapping[str, Any], path: str, depth: int
    ) -> Node:
        if name == "null":
            return Null()
        if name == "boolean":
            return Boolean()
        if name == "integer":
            return self.integer(schema, path)
        if name == "number":
            return self.number(schema, path)
        if name == "string":
            return self.string(schema, path)
        if depth + 1 > self.limits.max_depth:
            raise CompileLimitError(
                "depth",
                f"{path or '/'}: nesting depth exceeds the limit of {self.limits.max_depth}",
            )
        if name == "array":
            return self.array(schema, path, depth)
        return self.obj(schema, path, depth)

    def bound(self, schema: Mapping[str, Any], keyword: str, path: str) -> Decimal | None:
        if keyword not in schema:
            return None
        value = schema[keyword]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SchemaError(_pointer(path, keyword), "must be a number", reason="invalid")
        if isinstance(value, float) and not math.isfinite(value):
            raise SchemaError(_pointer(path, keyword), "must be finite", reason="invalid")
        try:
            decimal = Decimal(value) if isinstance(value, int) else Decimal(repr(value))
        except InvalidOperation as exc:  # pragma: no cover - finite floats always convert
            raise SchemaError(
                _pointer(path, keyword), "must be a number", reason="invalid"
            ) from exc
        digits = decimal.normalize().as_tuple()
        exponent = int(digits.exponent) if isinstance(digits.exponent, int) else 0
        width = max(len(digits.digits), len(digits.digits) + exponent, -exponent)
        if width > self.limits.max_number_digits:
            raise CompileLimitError(
                "number_digits",
                f"{_pointer(path, keyword)}: bound needs more than "
                f"{self.limits.max_number_digits} digits",
            )
        return decimal

    def integer(self, schema: Mapping[str, Any], path: str) -> Integer:
        lows: list[int] = []
        highs: list[int] = []
        if (value := self.bound(schema, "minimum", path)) is not None:
            lows.append(math.ceil(value))
        if (value := self.bound(schema, "exclusiveMinimum", path)) is not None:
            lows.append(math.floor(value) + 1)
        if (value := self.bound(schema, "maximum", path)) is not None:
            highs.append(math.floor(value))
        if (value := self.bound(schema, "exclusiveMaximum", path)) is not None:
            highs.append(math.ceil(value) - 1)
        return Integer(max(lows) if lows else None, min(highs) if highs else None)

    def number(self, schema: Mapping[str, Any], path: str) -> Number:
        low, low_excl = self.bound(schema, "minimum", path), False
        exclusive = self.bound(schema, "exclusiveMinimum", path)
        if exclusive is not None and (low is None or exclusive >= low):
            low, low_excl = exclusive, True
        high, high_excl = self.bound(schema, "maximum", path), False
        exclusive = self.bound(schema, "exclusiveMaximum", path)
        if exclusive is not None and (high is None or exclusive <= high):
            high, high_excl = exclusive, True
        return Number(low, low_excl, high, high_excl)

    def count_keyword(self, schema: Mapping[str, Any], keyword: str, path: str) -> int | None:
        if keyword not in schema:
            return None
        value = schema[keyword]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SchemaError(
                _pointer(path, keyword), "must be a non-negative integer", reason="invalid"
            )
        if value > self.limits.max_repeat:
            raise CompileLimitError(
                "repeat_bound",
                f"{_pointer(path, keyword)}: exceeds the repetition limit of "
                f"{self.limits.max_repeat}",
            )
        return value

    def string(self, schema: Mapping[str, Any], path: str) -> String:
        min_length = self.count_keyword(schema, "minLength", path) or 0
        max_length = self.count_keyword(schema, "maxLength", path)
        pattern = schema.get("pattern")
        if pattern is not None and not isinstance(pattern, str):
            raise SchemaError(_pointer(path, "pattern"), "must be a string", reason="invalid")
        if pattern is not None and len(pattern) > self.limits.max_pattern_length:
            raise CompileLimitError(
                "pattern_length",
                f"{_pointer(path, 'pattern')}: longer than {self.limits.max_pattern_length} "
                "characters",
            )
        return String(min_length, max_length, pattern)

    def array(self, schema: Mapping[str, Any], path: str, depth: int) -> Array:
        if "items" not in schema:
            raise SchemaError(
                path, "arrays must declare 'items' (unconstrained items are not supported)"
            )
        if schema.get("uniqueItems", False) is not False:
            raise SchemaError(_pointer(path, "uniqueItems"), "only uniqueItems: false is supported")
        items = self.node(schema["items"], _pointer(path, "items"), depth=depth + 1, root=False)
        min_items = self.count_keyword(schema, "minItems", path) or 0
        max_items = self.count_keyword(schema, "maxItems", path)
        return Array(items, min_items, max_items)

    def obj(self, schema: Mapping[str, Any], path: str, depth: int) -> Object:
        additional = schema.get("additionalProperties", False)
        if additional is not False:
            raise SchemaError(
                _pointer(path, "additionalProperties"),
                "only closed objects are supported: set additionalProperties to false",
            )
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise SchemaError(_pointer(path, "properties"), "must be an object", reason="invalid")
        if len(properties) > self.limits.max_properties:
            raise CompileLimitError(
                "properties",
                f"{_pointer(path, 'properties')}: more than "
                f"{self.limits.max_properties} properties",
            )
        required = schema.get("required", [])
        if (
            not isinstance(required, list)
            or not all(isinstance(name, str) for name in required)
            or len(set(required)) != len(required)
        ):
            raise SchemaError(
                _pointer(path, "required"), "must be an array of distinct strings", reason="invalid"
            )
        missing = [name for name in required if name not in properties]
        if missing:
            raise SchemaError(
                _pointer(path, "required"),
                "every required property must be declared in 'properties'",
            )
        result: list[Property] = []
        for name, subschema in properties.items():
            if not isinstance(name, str) or len(name) > _MAX_PROPERTY_NAME:
                raise SchemaError(
                    _pointer(path, "properties"),
                    f"property names must be strings of at most {_MAX_PROPERTY_NAME} characters",
                    reason="invalid",
                )
            node = self.node(
                subschema, _pointer(_pointer(path, "properties"), name), depth=depth + 1, root=False
            )
            result.append(Property(name, node, name in required))
        return Object(tuple(result))


def parse_schema(schema: Any, limits: CompileLimits = DEFAULT_LIMITS) -> SchemaIR:
    """Parse a decoded JSON Schema into the IR, or raise :class:`SchemaError`."""
    return _Parser(limits).parse(schema)


def iter_children(node: Node) -> Sequence[Node]:
    if isinstance(node, Array):
        return (node.items,)
    if isinstance(node, Object):
        return tuple(prop.schema for prop in node.properties)
    if isinstance(node, AnyOf):
        return node.branches
    return ()
