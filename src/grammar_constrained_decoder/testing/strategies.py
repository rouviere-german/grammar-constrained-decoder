"""Hypothesis strategies shared by the test suite and the benchmarks (dev-only module).

Nothing in the service imports this module; it needs the ``hypothesis`` dev dependency.

- :func:`regex_pairs` draws a byte-level regex AST together with an equivalent Python ``re``
  pattern over a small alphabet, for differential tests of the DFA pipeline.
- :func:`ecma_patterns` draws JSON Schema ``pattern`` strings from the constructs whose ECMA-262
  and Python ``re`` semantics coincide on the test alphabet (ASCII classes, no ``$``), so ``re``
  can serve as a trusted oracle.
- :func:`json_schemas` draws schemas in the supported subset, small enough that thousands can be
  compiled and checked against the derivative oracle in minutes.
"""

from __future__ import annotations

from typing import Any

from hypothesis import strategies as st

from grammar_constrained_decoder.automata.regex import (
    EPS,
    Regex,
    alt,
    byteset,
    cat,
    inter,
    literal,
    opt,
    rep,
    star,
)

__all__ = [
    "PATTERN_TEST_ALPHABET",
    "ecma_patterns",
    "intersections",
    "json_schemas",
    "regex_pairs",
    "text_samples",
]

RegexPair = tuple[Regex, str]
PATTERN_TEST_ALPHABET = 'abc019_- \t\nxyzé中😀"\\/'


def _leaf_pairs() -> st.SearchStrategy[RegexPair]:
    char = st.sampled_from("abc")
    return st.one_of(
        char.map(lambda c: (literal(c), c)),
        st.sets(char, min_size=1, max_size=3).map(
            lambda cs: (byteset(sum(1 << ord(c) for c in cs)), "[" + "".join(sorted(cs)) + "]")
        ),
        st.just((EPS, "(?:)")),
    )


def _extend_pairs(children: st.SearchStrategy[RegexPair]) -> st.SearchStrategy[RegexPair]:
    def repeat(pair: RegexPair, lo: int, extra: int | None) -> RegexPair:
        hi = None if extra is None else lo + extra
        bound = f"{{{lo},}}" if hi is None else f"{{{lo},{hi}}}"
        return rep(pair[0], lo, hi), f"(?:{pair[1]}){bound}"

    return st.one_of(
        st.tuples(children, children).map(lambda p: (cat(p[0][0], p[1][0]), p[0][1] + p[1][1])),
        st.tuples(children, children).map(
            lambda p: (alt(p[0][0], p[1][0]), f"(?:{p[0][1]}|{p[1][1]})")
        ),
        children.map(lambda p: (star(p[0]), f"(?:{p[1]})*")),
        children.map(lambda p: (opt(p[0]), f"(?:{p[1]})?")),
        st.builds(repeat, children, st.integers(0, 3), st.one_of(st.none(), st.integers(0, 3))),
    )


def regex_pairs(max_leaves: int = 8) -> st.SearchStrategy[RegexPair]:
    """(regex AST, equivalent Python pattern) over the alphabet ``abc``."""
    return st.recursive(_leaf_pairs(), _extend_pairs, max_leaves=max_leaves)


def intersections(max_leaves: int = 6) -> st.SearchStrategy[tuple[Regex, list[str]]]:
    """An intersection of two or three regexes, with the Python patterns of each operand."""
    return st.lists(regex_pairs(max_leaves), min_size=2, max_size=3).map(
        lambda pairs: (inter(*(p[0] for p in pairs)), [p[1] for p in pairs])
    )


def text_samples(alphabet: str = "abc", max_size: int = 8) -> st.SearchStrategy[str]:
    return st.text(alphabet=alphabet, max_size=max_size)


# -- JSON Schema patterns -----------------------------------------------------------------------

_PORTABLE_ATOMS = [
    "a", "b", "c", "0", "x", "_", "-", "é", "中", "[a-c]", "[^a]", "[0-9x]", "[a-c_]",
    r"\.", r"\-", r"\u00e9", r"\x61", "[ab]", "[^0-9]",
]  # fmt: skip
# \d \w \s and . agree with Python re only on ASCII-like inputs (re.ASCII, no CR or U+2028).
_ASCII_ATOMS = [*_PORTABLE_ATOMS, r"\d", r"\w", r"\W", r"\s", "."]


def _pattern_extend(children: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    quantifier = st.sampled_from(["*", "+", "?", "{2}", "{1,3}", "{0,2}", "{2,}", "*?", "+?"])
    return st.one_of(
        st.tuples(children, children).map("".join),
        st.tuples(children, children).map(lambda p: f"(?:{p[0]}|{p[1]})"),
        st.tuples(children, children).map(lambda p: f"({p[0]}|{p[1]})"),
        st.tuples(children, quantifier).map(lambda p: f"(?:{p[0]}){p[1]}"),
    )


def ecma_patterns(max_leaves: int = 5, *, portable: bool = False) -> st.SearchStrategy[str]:
    r"""Unanchored pattern bodies whose ECMA and Python semantics agree.

    With ``portable=False`` the atoms include ``\d \w \W \s .``, which agree with Python's
    ``re.ASCII`` mode on :data:`PATTERN_TEST_ALPHABET` (it has no Unicode digits or spaces, no
    carriage return and no U+2028). With ``portable=True`` only explicit characters and classes
    are used, so any Python-side generator (``hypothesis.from_regex``) agrees with ECMA semantics.
    """
    atoms = st.sampled_from(_PORTABLE_ATOMS if portable else _ASCII_ATOMS)
    return st.recursive(atoms, _pattern_extend, max_leaves=max_leaves)


# -- JSON Schemas in the supported subset --------------------------------------------------------

_NAMES = ("id", "name", "ok", "kind", "tags", "score", "a", "b", "note", "items")
_ENUM_VALUES = st.one_of(
    st.sampled_from(["red", "green", "blue", "x", "", "é", 'q"t']),
    st.integers(-20, 20),
    st.booleans(),
    st.none(),
    st.sampled_from([0.5, 1.25, -3.0]),
)


def _integer_schema() -> st.SearchStrategy[dict[str, Any]]:
    def build(low: int | None, width: int | None, exclusive: bool) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "integer"}
        if low is not None:
            schema["exclusiveMinimum" if exclusive else "minimum"] = low
        if width is not None:
            schema["maximum"] = (low if low is not None else 0) + width
        return schema

    return st.builds(
        build,
        st.one_of(st.none(), st.integers(-60, 60)),
        st.one_of(st.none(), st.integers(1, 120)),
        st.booleans(),
    )


def _number_schema() -> st.SearchStrategy[dict[str, Any]]:
    bound = st.integers(-2000, 2000).map(lambda value: value / 100)

    def build(low: float | None, high: float | None, low_ex: bool, high_ex: bool) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "number"}
        if low is not None:
            schema["exclusiveMinimum" if low_ex else "minimum"] = low
        if high is not None:
            if low is not None and high <= low:
                high = low + 1.5
            schema["exclusiveMaximum" if high_ex else "maximum"] = high
        return schema

    return st.builds(
        build,
        st.one_of(st.none(), bound),
        st.one_of(st.none(), bound),
        st.booleans(),
        st.booleans(),
    )


def _string_schema(with_patterns: bool) -> st.SearchStrategy[dict[str, Any]]:
    def build(min_length: int, extra: int | None, pattern: str | None) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "string"}
        if min_length:
            schema["minLength"] = min_length
        if extra is not None:
            schema["maxLength"] = min_length + extra
        if pattern is not None:
            schema["pattern"] = pattern
        return schema

    pattern = (
        st.one_of(
            st.none(),
            ecma_patterns(3, portable=True),
            ecma_patterns(3, portable=True).map(lambda p: "^" + p),
        )
        if with_patterns
        else st.none()
    )
    return st.builds(build, st.integers(0, 2), st.one_of(st.none(), st.integers(0, 5)), pattern)


def _leaf_schemas(with_patterns: bool) -> st.SearchStrategy[dict[str, Any]]:
    return st.one_of(
        st.just({"type": "null"}),
        st.just({"type": "boolean"}),
        _integer_schema(),
        _number_schema(),
        _string_schema(with_patterns),
        st.lists(_ENUM_VALUES, min_size=1, max_size=4).map(lambda values: {"enum": values}),
        _ENUM_VALUES.map(lambda value: {"const": value}),
        st.just({"type": ["string", "null"], "maxLength": 3}),
    )


def _container_schemas(
    children: st.SearchStrategy[dict[str, Any]],
) -> st.SearchStrategy[dict[str, Any]]:
    def array(items: dict[str, Any], min_items: int, extra: int | None) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": "array", "items": items}
        if min_items:
            schema["minItems"] = min_items
        if extra is not None:
            schema["maxItems"] = min_items + extra
        return schema

    def obj(members: list[tuple[str, dict[str, Any], bool]]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {name: sub for name, sub, _ in members},
            "required": [name for name, _, required in members if required],
            "additionalProperties": False,
        }

    member = st.tuples(st.sampled_from(_NAMES), children, st.booleans())
    return st.one_of(
        st.builds(array, children, st.integers(0, 1), st.one_of(st.none(), st.integers(0, 2))),
        st.lists(member, min_size=1, max_size=3, unique_by=lambda item: item[0]).map(obj),
        st.lists(children, min_size=2, max_size=2).map(lambda branches: {"anyOf": branches}),
    )


def json_schemas(
    *, with_patterns: bool = True, max_leaves: int = 4
) -> st.SearchStrategy[dict[str, Any]]:
    """Schemas in the supported subset (no ``$ref``; see tests for recursive schemas)."""
    return st.recursive(_leaf_schemas(with_patterns), _container_schemas, max_leaves=max_leaves)
