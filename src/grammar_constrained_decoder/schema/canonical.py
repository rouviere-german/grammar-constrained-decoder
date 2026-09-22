"""Canonical serialization of a JSON instance for a schema, and an IR-level validity check.

``canonical_dumps(ir, value)`` produces the one serialization the compiled language uses for
``value``: compact separators, object members in the schema's declaration order, canonical string
escapes, integers as plain digits and numbers in fixed-point form (never exponent notation, never
``-0``). Enum values are written as the schema's own literal for the equal enum member, so ``1``
and ``1.0`` in an enum of ``[1.0]`` both serialize as ``1.0``; enum literals follow the same number
rules (``ir.literal_text``).

``matches(ir, value)`` decides validity against the IR without ``jsonschema`` and without Python
``re``: patterns are compiled by the in-repo parser into DFAs. It is used to pick the ``anyOf``
branch whose canonical form applies, and by tests as a second opinion next to ``jsonschema``.
"""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from decimal import Decimal
from typing import Any

from grammar_constrained_decoder.automata.dfa import DFA, compile_regex
from grammar_constrained_decoder.automata.limits import DEFAULT_LIMITS, CompileBudget, CompileLimits
from grammar_constrained_decoder.automata.pattern import compile_pattern
from grammar_constrained_decoder.core.canonical import canonical_json
from grammar_constrained_decoder.schema.ir import (
    ROOT_REF,
    AnyOf,
    Array,
    Boolean,
    Enum,
    Integer,
    Node,
    Null,
    Number,
    Object,
    Ref,
    SchemaIR,
    String,
    fixed_point,
    json_type_of,
)

__all__ = [
    "CanonicalError",
    "canonical_dumps",
    "decimal_of",
    "matches",
    "number_text",
    "pattern_dfa",
    "pattern_search",
]


class CanonicalError(ValueError):
    """The value is not an instance of the schema, so it has no canonical form."""


_PATTERN_CACHE_SIZE = 512
_pattern_cache: OrderedDict[tuple[str, CompileLimits], DFA] = OrderedDict()
_pattern_lock = threading.Lock()


def pattern_dfa(
    pattern: str, limits: CompileLimits = DEFAULT_LIMITS, budget: CompileBudget | None = None
) -> DFA:
    """DFA for an ECMA ``pattern`` (search semantics over raw UTF-8), compiled under ``limits``.

    Compilation charges ``budget`` (a fresh one for ``limits`` when omitted), so a hostile
    pattern costs at most the caller's timeout and state caps and raises :class:`CompileError`.
    Successes are cached per (pattern, limits); failures are not (callers cache rejections).
    """
    key = (pattern, limits)
    with _pattern_lock:
        cached = _pattern_cache.get(key)
        if cached is not None:
            _pattern_cache.move_to_end(key)
            return cached
    dfa = compile_regex(
        compile_pattern(pattern, limits=limits),
        budget if budget is not None else CompileBudget(limits),
    )
    with _pattern_lock:
        _pattern_cache[key] = dfa
        _pattern_cache.move_to_end(key)
        while len(_pattern_cache) > _PATTERN_CACHE_SIZE:
            _pattern_cache.popitem(last=False)
    return dfa


def pattern_search(pattern: str, text: str, limits: CompileLimits = DEFAULT_LIMITS) -> bool:
    """ECMA-style ``pattern`` search over ``text`` using the in-repo parser and a DFA.

    Text with a lone surrogate has no UTF-8 encoding and never matches (the compiled classes
    exclude surrogate code points).
    """
    if any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        return False
    return pattern_dfa(pattern, limits).accepts(text.encode("utf-8"))


def decimal_of(value: int | float) -> Decimal:
    if isinstance(value, float):
        return Decimal(repr(value))
    return Decimal(value)


def number_text(value: Any, *, integer: bool) -> str:
    """Canonical text of a JSON number (fixed point; integral schema values without fraction)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CanonicalError("not a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise CanonicalError("non-finite number")
    if integer:
        if isinstance(value, float):
            if not value.is_integer():
                raise CanonicalError("not an integer")
            return str(int(value))
        return str(value)
    if isinstance(value, int):
        return str(value)
    return fixed_point(value)


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)


def _in_range(node: Integer | Number, value: int | float) -> bool:
    number = decimal_of(value)
    if isinstance(node, Integer):
        return (node.minimum is None or number >= node.minimum) and (
            node.maximum is None or number <= node.maximum
        )
    low_ok = node.minimum is None or (
        number > node.minimum if node.exclusive_minimum else number >= node.minimum
    )
    high_ok = node.maximum is None or (
        number < node.maximum if node.exclusive_maximum else number <= node.maximum
    )
    return low_ok and high_ok


def _depth_of(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + max((_depth_of(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_depth_of(item) for item in value), default=0)
    return 0


class _Checker:
    def __init__(self, ir: SchemaIR) -> None:
        self.ir = ir

    def resolve(self, node: Node) -> Node:
        seen = 0
        while isinstance(node, Ref):
            node = self.ir.root if node.name == ROOT_REF else self.ir.definitions[node.name]
            seen += 1
            if seen > 64:
                raise CanonicalError("reference cycle without a container")
        return node

    def matches(self, node: Node, value: Any) -> bool:  # noqa: PLR0911 - one branch per type
        node = self.resolve(node)
        if isinstance(node, Null):
            return value is None
        if isinstance(node, Boolean):
            return isinstance(value, bool)
        if isinstance(node, Integer):
            return json_type_of(value) == "integer" and _in_range(node, value)
        if isinstance(node, Number):
            return _is_number(value) and math.isfinite(value) and _in_range(node, value)
        if isinstance(node, String):
            return isinstance(value, str) and self.string_ok(node, value)
        if isinstance(node, Enum):
            return self.enum_literal(node, value) is not None
        if isinstance(node, AnyOf):
            return any(self.matches(branch, value) for branch in node.branches)
        if isinstance(node, Array):
            return (
                isinstance(value, list)
                and len(value) >= node.min_items
                and (node.max_items is None or len(value) <= node.max_items)
                and all(self.matches(node.items, item) for item in value)
            )
        if not isinstance(node, Object):  # pragma: no cover - resolve() removed every Ref
            raise CanonicalError("unresolved reference")
        return isinstance(value, dict) and self.object_ok(node, value)

    def string_ok(self, node: String, value: str) -> bool:
        if len(value) < node.min_length or (
            node.max_length is not None and len(value) > node.max_length
        ):
            return False
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            return False
        return node.pattern is None or pattern_search(node.pattern, value)

    def object_ok(self, node: Object, value: dict[str, Any]) -> bool:
        declared = {prop.name: prop for prop in node.properties}
        if any(key not in declared for key in value):
            return False
        return all(
            (prop.name in value and self.matches(prop.schema, value[prop.name]))
            or (prop.name not in value and not prop.required)
            for prop in node.properties
        )

    def enum_literal(self, node: Enum, value: Any) -> str | None:
        kind = json_type_of(value)
        for member, text in zip(node.values, node.literals, strict=True):
            if json_type_of(member) != kind and not (_is_number(member) and _is_number(value)):
                continue
            if _is_number(member) and _is_number(value):
                if decimal_of(member) == decimal_of(value):
                    return text
            elif canonical_json(member) == canonical_json(value):
                return text
        return None

    def dumps(self, node: Node, value: Any) -> str:  # noqa: PLR0911 - one branch per type
        node = self.resolve(node)
        if not self.matches(node, value):
            raise CanonicalError("value is not an instance of the schema")
        if isinstance(node, Null | Boolean):
            return json.dumps(value)
        if isinstance(node, Integer | Number):
            return number_text(value, integer=isinstance(node, Integer))
        if isinstance(node, String):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(node, Enum):
            literal = self.enum_literal(node, value)
            if literal is None:  # pragma: no cover - matches() checked it
                raise CanonicalError("value is not an enum member")
            return literal
        if isinstance(node, AnyOf):
            branch = next(branch for branch in node.branches if self.matches(branch, value))
            return self.dumps(branch, value)
        if isinstance(node, Array):
            return "[" + ",".join(self.dumps(node.items, item) for item in value) + "]"
        if not isinstance(node, Object):  # pragma: no cover - resolve() removed every Ref
            raise CanonicalError("unresolved reference")
        parts = [
            json.dumps(prop.name, ensure_ascii=False)
            + ":"
            + self.dumps(prop.schema, value[prop.name])
            for prop in node.properties
            if prop.name in value
        ]
        return "{" + ",".join(parts) + "}"


def matches(ir: SchemaIR, value: Any) -> bool:
    """True when ``value`` is an instance of the schema (patterns checked by DFA)."""
    return _Checker(ir).matches(ir.root, value)


def canonical_dumps(ir: SchemaIR, value: Any) -> str:
    """Canonical serialization, or :class:`CanonicalError` when ``value`` is not an instance."""
    return _Checker(ir).dumps(ir.root, value)


def nesting_depth(value: Any) -> int:
    """Container nesting depth of a JSON value (scalars have depth 0)."""
    return _depth_of(value)
