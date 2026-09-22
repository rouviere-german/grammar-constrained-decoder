"""Schema IR -> one byte-level regular expression for compact canonical JSON.

The language contains exactly the canonical serializations the decoder may emit:

- no insignificant whitespace; object members in the schema's declaration order;
- strings use the canonical escapes of ``json.dumps(ensure_ascii=False)``: ``\\" \\\\ \\b \\f \\n
  \\r \\t`` and ``\\u00xx`` for other control characters, every other scalar value as raw UTF-8.
  ``minLength``/``maxLength`` count code points, and ``pattern`` is matched against the decoded
  value by translating each code point class into its encoded form, then intersecting;
- numbers as described in ``numeric.py``; enum/const values as their canonical literal.

Bounded nesting is inlined: every object and array adds one level, and the language never needs a
stack. A container deeper than ``max_depth`` is an error, except inside a recursive ``$ref``,
where the recursion is cut (the deeper alternative becomes the empty language). A reference cycle
that passes through no container is cut immediately, which yields the least fixed point. Chains
of references are bounded by ``max_ref_depth`` open expansions, so a long ``$ref`` chain is a typed
``ref_depth`` rejection rather than a Python recursion error.
Expansions are memoized per (definition, depth, recursion flag, open references), so identical
subtrees (for example the two children of a binary tree) are shared in the regex DAG.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

from grammar_constrained_decoder.automata.charset import normalize_ranges
from grammar_constrained_decoder.automata.limits import (
    DEFAULT_LIMITS,
    CompileBudget,
    CompileLimitError,
    CompileLimits,
)
from grammar_constrained_decoder.automata.pattern import (
    ANY_CHAR,
    Ranges,
    compile_pattern,
    utf8_class,
)
from grammar_constrained_decoder.automata.regex import (
    EMPTY,
    EPS,
    Regex,
    alt,
    cat,
    inter,
    literal,
    opt,
    rep,
)
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
)
from grammar_constrained_decoder.schema.numeric import integer_range, number_range

__all__ = ["JSON_CHAR", "escape_code_point", "json_string_class", "lower_schema"]

_SHORT_ESCAPES = {
    0x22: '\\"',
    0x5C: "\\\\",
    0x08: "\\b",
    0x0C: "\\f",
    0x0A: "\\n",
    0x0D: "\\r",
    0x09: "\\t",
}
_QUOTE, _COMMA, _COLON = literal('"'), literal(","), literal(":")


def escape_code_point(code: int) -> str:
    """Canonical JSON text of one code point inside a string literal."""
    if code in _SHORT_ESCAPES:
        return _SHORT_ESCAPES[code]
    if code < 0x20:
        return f"\\u{code:04x}"
    return chr(code)


def json_string_class(ranges: Ranges) -> Regex:
    """Encoded form of a code point class inside a JSON string literal."""
    raw: Ranges = []
    escaped: list[int] = []
    for lo, hi in normalize_ranges(ranges):
        cursor = lo
        while cursor <= hi:
            if cursor < 0x20 or cursor in (0x22, 0x5C):
                escaped.append(cursor)
                cursor += 1
                continue
            end = hi
            for special in (0x22, 0x5C):
                if cursor < special <= end:
                    end = special - 1
            raw.append((cursor, end))
            cursor = end + 1
    return alt(utf8_class(raw), *(literal(escape_code_point(code)) for code in escaped))


JSON_CHAR = json_string_class(ANY_CHAR)


@lru_cache(maxsize=256)
def _pattern_language(pattern: str, limits: CompileLimits) -> Regex:
    return compile_pattern(pattern, encoder=json_string_class, limits=limits)


@dataclass
class _Lowerer:
    ir: SchemaIR
    limits: CompileLimits
    budget: CompileBudget
    memo: dict[tuple[object, ...], Regex] = field(default_factory=dict)
    open_refs: int = 0  # $ref expansions on the current path (bounds the recursion depth)

    def node(self, node: Node, depth: int, stack: dict[str, int], recursive: bool) -> Regex:  # noqa: PLR0911
        self.budget.tick()
        if isinstance(node, Null):
            return literal("null")
        if isinstance(node, Boolean):
            return alt(literal("true"), literal("false"))
        if isinstance(node, Integer):
            return integer_range(node.minimum, node.maximum)
        if isinstance(node, Number):
            return number_range(
                node.minimum, node.exclusive_minimum, node.maximum, node.exclusive_maximum
            )
        if isinstance(node, String):
            return self.string(node)
        if isinstance(node, Enum):
            return alt(*(literal(text) for text in node.literals))
        if isinstance(node, AnyOf):
            return alt(*(self.node(branch, depth, stack, recursive) for branch in node.branches))
        if isinstance(node, Ref):
            return self.ref(node.name, depth, stack, recursive)
        if depth + 1 > self.limits.max_depth:
            if recursive:
                return EMPTY
            raise CompileLimitError(
                "depth", f"nesting depth exceeds the limit of {self.limits.max_depth}"
            )
        if isinstance(node, Array):
            return self.array(node, depth, stack, recursive)
        return self.obj(node, depth, stack, recursive)

    def ref(self, name: str, depth: int, stack: dict[str, int], recursive: bool) -> Regex:
        if name in stack:
            if stack[name] == depth:
                return EMPTY  # a cycle through no container adds nothing (least fixed point)
            recursive = True
        # The stack is part of the key: which references are open decides where cycles are cut.
        key = (name, depth, recursive, tuple(sorted(stack.items())))
        cached = self.memo.get(key)
        if cached is not None:
            return cached
        target = self.ir.root if name == ROOT_REF else self.ir.definitions[name]
        self.open_refs += 1
        if self.open_refs > self.limits.max_ref_depth:
            raise CompileLimitError(
                "ref_depth",
                f"$ref expansions nest deeper than the limit of {self.limits.max_ref_depth}",
            )
        try:
            result = self.node(target, depth, {**stack, name: depth}, recursive)
        finally:
            self.open_refs -= 1
        self.memo[key] = result
        return result

    def string(self, node: String) -> Regex:
        if node.max_length is not None and node.min_length > node.max_length:
            return EMPTY
        body = rep(JSON_CHAR, node.min_length, node.max_length)
        if node.pattern is not None:
            language = _pattern_language(node.pattern, self.limits)
            unconstrained = node.min_length == 0 and node.max_length is None
            body = language if unconstrained else inter(body, language)
        return cat(_QUOTE, body, _QUOTE)

    def array(self, node: Array, depth: int, stack: dict[str, int], recursive: bool) -> Regex:
        lo, hi = node.min_items, node.max_items
        if hi is not None and lo > hi:
            return EMPTY
        item = self.node(node.items, depth + 1, stack, recursive)
        if item is EMPTY or hi == 0:
            return literal("[]") if lo == 0 else EMPTY
        following = cat(_COMMA, item)
        if lo == 0:
            body = opt(cat(item, rep(following, 0, None if hi is None else hi - 1)))
        else:
            body = cat(item, rep(following, lo - 1, None if hi is None else hi - 1))
        return cat(literal("["), body, literal("]"))

    def obj(self, node: Object, depth: int, stack: dict[str, int], recursive: bool) -> Regex:
        members: list[tuple[Regex, bool]] = []
        for prop in node.properties:
            value = self.node(prop.schema, depth + 1, stack, recursive)
            if value is EMPTY:
                if prop.required:
                    return EMPTY
                continue
            key = literal(json.dumps(prop.name, ensure_ascii=False))
            members.append((cat(key, _COLON, value), prop.required))
        # first[i]: members i.. with nothing emitted yet; rest[i]: members i.. after an emission.
        first: Regex = EPS
        rest: Regex = EPS
        for member, required in reversed(members):
            if required:
                first, rest = cat(member, rest), cat(_COMMA, member, rest)
            else:
                first, rest = alt(cat(member, rest), first), cat(opt(cat(_COMMA, member)), rest)
        return cat(literal("{"), first, literal("}"))


def lower_schema(
    ir: SchemaIR, limits: CompileLimits = DEFAULT_LIMITS, budget: CompileBudget | None = None
) -> Regex:
    """Regex for the canonical serializations of every instance of ``ir`` (within depth)."""
    lowerer = _Lowerer(ir, limits, budget if budget is not None else CompileBudget(limits))
    return lowerer.node(ir.root, 0, {ROOT_REF: 0}, False)
