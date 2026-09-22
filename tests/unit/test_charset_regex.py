"""Byte sets, UTF-8 range encoding and the normalizing regex constructors."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from grammar_constrained_decoder.automata.charset import (
    ALL_BYTES,
    MAX_SCALAR,
    byte_mask,
    byte_range,
    complement_ranges,
    iter_bytes,
    mask_ranges,
    normalize_ranges,
    utf8_sequences,
)
from grammar_constrained_decoder.automata.regex import (
    EMPTY,
    EPS,
    Alt,
    And,
    Repeat,
    alt,
    byteset,
    cat,
    inter,
    iter_nodes,
    literal,
    masks_of,
    opt,
    plus,
    rep,
    size,
    star,
    to_pattern,
)

pytestmark = pytest.mark.unit

scalars = st.integers(0, MAX_SCALAR).filter(lambda c: not 0xD800 <= c <= 0xDFFF)


def _matches(sequence: list[tuple[int, int]], data: bytes) -> bool:
    return len(sequence) == len(data) and all(
        lo <= b <= hi for (lo, hi), b in zip(sequence, data, strict=True)
    )


@given(st.integers(0, MAX_SCALAR), st.integers(0, 0x3000), scalars)
def test_utf8_sequences_match_exactly_the_code_points_in_range(
    lo: int, width: int, probe: int
) -> None:
    hi = min(MAX_SCALAR, lo + width)
    sequences = utf8_sequences(lo, hi)
    encoded = chr(probe).encode("utf-8")
    hits = sum(_matches(seq, encoded) for seq in sequences)
    assert hits == (1 if lo <= probe <= hi else 0)
    for candidate in (lo, hi, (lo + hi) // 2):
        if not 0xD800 <= candidate <= 0xDFFF:
            assert sum(_matches(seq, chr(candidate).encode("utf-8")) for seq in sequences) == 1


def test_utf8_sequences_cover_all_scalars_and_never_surrogates() -> None:
    sequences = utf8_sequences(0, MAX_SCALAR)
    assert [(0, 127)] in sequences
    surrogate = b"\xed\xa0\x80"  # would encode U+D800
    assert not any(_matches(seq, surrogate) for seq in sequences)
    assert utf8_sequences(5, 4) == []
    with pytest.raises(ValueError, match="code points"):
        utf8_sequences(0, MAX_SCALAR + 1)


@given(st.lists(st.tuples(st.integers(0, 300), st.integers(0, 300))), st.integers(0, 400))
def test_complement_ranges_partition_the_scalars(ranges: list[tuple[int, int]], probe: int) -> None:
    ranges = [(min(a, b), max(a, b)) for a, b in ranges]
    inside = any(lo <= probe <= hi for lo, hi in ranges)
    outside = any(lo <= probe <= hi for lo, hi in complement_ranges(ranges))
    assert inside != outside


def test_byte_mask_helpers() -> None:
    mask = byte_range(0x30, 0x39) | byte_mask([0x41, 0x43])
    assert list(iter_bytes(mask))[:3] == [0x30, 0x31, 0x32]
    assert mask_ranges(mask) == [(0x30, 0x39), (0x41, 0x41), (0x43, 0x43)]
    assert byte_range(5, 4) == 0
    assert normalize_ranges([(5, 9), (1, 3), (4, 4), (20, 19)]) == [(1, 9)]
    with pytest.raises(ValueError, match="byte"):
        byte_range(0, 256)
    with pytest.raises(ValueError, match="byte"):
        byte_mask([300])


def test_nodes_are_interned_so_equality_is_identity() -> None:
    a, b = literal("a"), literal("b")
    assert cat(a, b) is cat(a, b)
    assert alt(a, b) is alt(b, a)
    assert alt(a, alt(b, a)) is alt(a, b)
    assert inter(star(a), star(alt(a, b))) is inter(star(alt(b, a)), star(a))


def test_smart_constructors_apply_the_similarity_rules() -> None:
    a = literal("a")
    assert cat(a, EMPTY) is EMPTY
    assert cat(EPS, a, EPS) is a
    assert cat() is EPS
    assert alt() is EMPTY
    assert alt(a, EMPTY) is a
    assert star(star(a)) is star(a)
    assert star(EPS) is EPS
    assert star(EMPTY) is EPS
    assert star(opt(a)) is star(a)
    assert rep(a, 0, None) is star(a)
    assert rep(a, 1, 1) is a
    assert rep(a, 0, 0) is EPS
    assert rep(EMPTY, 0, 3) is EPS
    assert rep(EMPTY, 1, 3) is EMPTY
    assert rep(opt(a), 2, 4) is rep(opt(a), 0, 4)
    assert alt(EPS, star(a)) is star(a)
    single = alt(byteset(1 << 97), byteset(1 << 98))
    assert single is byteset((1 << 97) | (1 << 98))
    assert inter(byteset(0b1100), byteset(0b0110)) is byteset(0b0100)
    assert inter(a, EMPTY) is EMPTY
    assert inter(EPS, star(a)) is EPS
    assert inter(EPS, a) is EMPTY
    assert byteset(0) is EMPTY
    with pytest.raises(ValueError, match="bounds"):
        rep(a, 3, 2)
    with pytest.raises(ValueError, match="zero operands"):
        inter()
    with pytest.raises(ValueError, match="mask"):
        byteset(ALL_BYTES + 1)


def test_nullability_is_computed_structurally() -> None:
    a = literal("a")
    assert star(a).nullable
    assert not plus(a).nullable
    assert opt(a).nullable
    assert not cat(opt(a), a).nullable
    assert alt(a, EPS).nullable
    assert not inter(star(a), a).nullable


def test_walkers_and_rendering() -> None:
    regex = cat(
        literal("ab"),
        rep(alt(literal("c"), literal("dd")), 1, 3),
        inter(star(literal("x")), plus(literal("x"))),
    )
    kinds = {type(node) for node in iter_nodes(regex)}
    assert {Alt, And, Repeat} <= kinds
    assert size(regex) == len(list(iter_nodes(regex)))
    assert masks_of(regex) >= {1 << ord("a"), 1 << ord("x")}
    rendered = to_pattern(regex)
    assert "ab" in rendered
    assert "{1,3}" in rendered
    assert "&" in rendered
    assert to_pattern(EMPTY) == "∅"
    assert to_pattern(EPS) == "ε"
    assert to_pattern(byteset(ALL_BYTES)) == "."
    assert "\\x00" in to_pattern(byteset(1))
    assert repr(literal("q")) == "Regex('q')"
