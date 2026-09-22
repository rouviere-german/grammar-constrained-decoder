"""Integer and decimal ranges lowered to regular languages over canonical decimal text."""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from grammar_constrained_decoder.automata.dfa import compile_regex
from grammar_constrained_decoder.schema.numeric import (
    NONNEG_DECIMAL,
    decimal_compare,
    integer_range,
    nonneg_integer_range,
    number_range,
)

pytestmark = pytest.mark.unit

bounds = st.one_of(st.none(), st.integers(-5_000, 5_000))
decimals = st.integers(-400_000, 400_000).map(lambda n: Decimal(n) / 1000)
NON_CANONICAL = ["01", "-0", "+1", "", "-", "00", "1.", ".5", "1e5", "-0.0", " 1", "1 "]


@given(bounds, bounds, st.lists(st.integers(-20_000, 20_000), max_size=30))
def test_integer_ranges_are_exact(low: int | None, high: int | None, probes: list[int]) -> None:
    dfa = compile_regex(integer_range(low, high))
    for value in [*probes, *(b for b in (low, high) if b is not None)]:
        expected = (low is None or value >= low) and (high is None or value <= high)
        assert dfa.accepts(str(value).encode()) == expected, (low, high, value)
    for text in [*NON_CANONICAL, "1.0"]:
        assert not dfa.accepts(text.encode()), text


def test_huge_and_unbounded_integers() -> None:
    dfa = compile_regex(integer_range(10**15, None))
    assert dfa.accepts(str(10**40).encode())
    assert not dfa.accepts(str(10**15 - 1).encode())
    negative = compile_regex(integer_range(None, -3))
    assert negative.accepts(b"-3")
    assert negative.accepts(b"-99999999999")
    assert not negative.accepts(b"-2")
    assert compile_regex(integer_range(5, 4)).is_empty
    assert compile_regex(nonneg_integer_range(7, 3)).is_empty
    with pytest.raises(ValueError, match="non-negative"):
        nonneg_integer_range(-1, 3)


@given(
    st.one_of(st.none(), decimals),
    st.booleans(),
    st.one_of(st.none(), decimals),
    st.booleans(),
    st.lists(decimals, max_size=25),
    st.lists(st.integers(0, 3), max_size=25),
)
def test_decimal_ranges_are_exact(  # noqa: PLR0917 - hypothesis passes positionally
    low: Decimal | None,
    low_exclusive: bool,
    high: Decimal | None,
    high_exclusive: bool,
    probes: list[Decimal],
    padding: list[int],
) -> None:
    dfa = compile_regex(number_range(low, low_exclusive, high, high_exclusive))
    candidates = [*probes, *(b for b in (low, high) if b is not None)]
    for index, value in enumerate(candidates):
        text = format(value, "f")
        if value == 0:
            text = "0"
        extra = padding[index % len(padding)] if padding else 0
        if extra:  # trailing zeros do not change the value
            text += ("" if "." in text else ".") + "0" * extra
        above = low is None or (value > low if low_exclusive else value >= low)
        below = high is None or (value < high if high_exclusive else value <= high)
        assert dfa.accepts(text.encode()) == (above and below), (low, high, text)
    for text in NON_CANONICAL:
        assert not dfa.accepts(text.encode()), text


@pytest.mark.parametrize(
    ("bound", "op", "accepted", "rejected"),
    [
        ("1.5", "lt", ["0", "1", "1.4999", "1.49"], ["1.5", "1.50", "2"]),
        ("1.5", "le", ["1.5", "1.500"], ["1.51", "2.0"]),
        ("1.5", "gt", ["1.51", "2", "10.0"], ["1.5", "1.4"]),
        ("1.5", "ge", ["1.5", "1.6"], ["1.49"]),
        ("0", "gt", ["0.001", "3"], ["0", "0.000"]),
        ("0", "le", ["0", "0.0"], ["0.1"]),
        ("0.05", "lt", ["0.04", "0.049", "0", "0.0"], ["0.05", "0.1"]),
    ],
)
def test_decimal_comparisons(bound: str, op: str, accepted: list[str], rejected: list[str]) -> None:
    dfa = compile_regex(decimal_compare(Decimal(bound), op))
    for text in accepted:
        assert dfa.accepts(text.encode()), text
    for text in rejected:
        assert not dfa.accepts(text.encode()), text


def test_decimal_compare_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="comparison"):
        decimal_compare(Decimal(1), "eq")
    with pytest.raises(ValueError, match="non-negative"):
        decimal_compare(Decimal(-1), "lt")


def test_unbounded_number_syntax() -> None:
    dfa = compile_regex(number_range(None, False, None, False))
    for text in ["0", "-1", "3.14", "-0.5", "1000000000000000000000.000001"]:
        assert dfa.accepts(text.encode()), text
    for text in NON_CANONICAL:
        assert not dfa.accepts(text.encode()), text
    assert compile_regex(NONNEG_DECIMAL).accepts(b"12.5")
