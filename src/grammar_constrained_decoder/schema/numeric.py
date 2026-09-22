"""Numeric ranges lowered to regular languages over canonical decimal text.

Integers are written without leading zeros, without ``+`` and without ``-0``. Numbers use the
fixed-point form ``-?(0|[1-9][0-9]*)(\\.[0-9]+)?``: no exponent, a fraction only when present, and
negative zero is never produced. Both are valid JSON number syntax.

Integer ranges use the classic digit-length decomposition: split ``[lo, hi]`` by number of digits,
then compare same-length digit strings lexicographically, which equals numeric order.

Decimal ranges compare a fixed-point string ``I.F`` with a bound ``B = Bi.Bf`` (``Bf`` without
trailing zeros): ``I.F < B`` iff ``I < Bi``, or ``I = Bi`` and the fraction digits compare lower
position by position (missing digits count as zeros). Each comparison is a small regular
language; a two-sided range is the intersection of a lower-bound and an upper-bound language,
which the compiler builds with a product automaton.
"""

from __future__ import annotations

from decimal import Decimal

from grammar_constrained_decoder.automata.charset import byte_range
from grammar_constrained_decoder.automata.regex import (
    EMPTY,
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
    "DIGIT",
    "NONNEG_DECIMAL",
    "decimal_compare",
    "integer_range",
    "nonneg_integer_range",
    "number_range",
]

DIGIT = byteset(byte_range(0x30, 0x39))
NONZERO_DIGIT = byteset(byte_range(0x31, 0x39))
_DOT = literal(".")
_MINUS = literal("-")


def _digits(lo: int, hi: int) -> Regex:
    if lo > hi:
        return EMPTY
    return byteset(byte_range(0x30 + lo, 0x30 + hi))


def _same_length(low: str, high: str) -> Regex:
    """Digit strings ``s`` with ``len(s) == len(low) == len(high)`` and ``low <= s <= high``."""
    if not low:
        return EPS
    width = len(low)
    if low == "0" * width and high == "9" * width:
        return rep(DIGIT, width, width)
    if low[0] == high[0]:
        return cat(literal(low[0]), _same_length(low[1:], high[1:]))
    rest = width - 1
    first, last = int(low[0]), int(high[0])
    return alt(
        cat(literal(low[0]), _same_length(low[1:], "9" * rest)),
        cat(_digits(first + 1, last - 1), rep(DIGIT, rest, rest)),
        cat(literal(high[0]), _same_length("0" * rest, high[1:])),
    )


def nonneg_integer_range(low: int, high: int | None) -> Regex:
    """Canonical decimal text of the integers in ``[low, high]`` (``high=None``: unbounded)."""
    if low < 0:
        raise ValueError("low must be non-negative")
    if high is not None and high < low:
        return EMPTY
    low_width = len(str(low))
    high_width = len(str(high)) if high is not None else low_width
    parts = []
    for width in range(low_width, high_width + 1):
        start = max(low, 10 ** (width - 1) if width > 1 else 0)
        end = 10**width - 1 if high is None else min(high, 10**width - 1)
        if start <= end:
            parts.append(_same_length(str(start), str(end)))
    if high is None:
        parts.append(cat(NONZERO_DIGIT, rep(DIGIT, low_width, None)))
    return alt(*parts)


def integer_range(low: int | None, high: int | None) -> Regex:
    """Canonical integers in ``[low, high]``; ``None`` leaves a side unbounded."""
    if low is not None and high is not None and low > high:
        return EMPTY
    parts = []
    if high is None or high >= 0:
        parts.append(nonneg_integer_range(max(low, 0) if low is not None else 0, high))
    if low is None or low < 0:
        magnitude_low = 1 if high is None or high >= 0 else -high
        magnitude_high = None if low is None else -low
        if magnitude_high is None or magnitude_low <= magnitude_high:
            parts.append(cat(_MINUS, nonneg_integer_range(magnitude_low, magnitude_high)))
    return alt(*parts)


NONNEG_INTEGER = nonneg_integer_range(0, None)
_FRACTION = cat(_DOT, rep(DIGIT, 1, None))
_ANY_SUFFIX = opt(_FRACTION)
NONNEG_DECIMAL = cat(NONNEG_INTEGER, _ANY_SUFFIX)


def _split(bound: Decimal) -> tuple[int, str]:
    if bound < 0:
        raise ValueError("bound must be non-negative")
    text = format(bound, "f")
    integer, _, fraction = text.partition(".")
    return int(integer), fraction.rstrip("0")


def _fraction_below(fraction: str) -> Regex:
    """Non-empty digit strings F with 0.F < 0.fraction (``fraction`` has no trailing zeros)."""
    parts = []
    for position, char in enumerate(fraction):
        prefix = literal(fraction[:position])
        digit = int(char)
        if digit > 0:
            parts.append(cat(prefix, _digits(0, digit - 1), star(DIGIT)))
        if position >= 1:
            parts.append(prefix)
    return alt(*parts)


def _suffix_equal(fraction: str) -> Regex:
    if not fraction:
        return opt(cat(_DOT, rep(literal("0"), 1, None)))
    return cat(literal("." + fraction), star(literal("0")))


def _suffix_below(fraction: str) -> Regex:
    if not fraction:
        return EMPTY
    return alt(EPS, cat(_DOT, _fraction_below(fraction)))


def _suffix_above(fraction: str) -> Regex:
    parts = []
    for position, char in enumerate(fraction):
        digit = int(char)
        if digit < 9:
            parts.append(cat(literal(fraction[:position]), _digits(digit + 1, 9), star(DIGIT)))
    parts.append(cat(literal(fraction), star(DIGIT), NONZERO_DIGIT, star(DIGIT)))
    return cat(_DOT, alt(*parts))


def decimal_compare(bound: Decimal, op: str) -> Regex:
    """Non-negative fixed-point strings whose value is ``op`` (lt, le, gt, ge) ``bound >= 0``."""
    integer, fraction = _split(bound)
    int_text = literal(str(integer))
    below = alt(
        cat(nonneg_integer_range(0, integer - 1), _ANY_SUFFIX) if integer > 0 else EMPTY,
        cat(int_text, _suffix_below(fraction)),
    )
    above = alt(
        cat(nonneg_integer_range(integer + 1, None), _ANY_SUFFIX),
        cat(int_text, _suffix_above(fraction)),
    )
    equal = cat(int_text, _suffix_equal(fraction))
    table = {"lt": below, "le": alt(below, equal), "gt": above, "ge": alt(above, equal)}
    if op not in table:
        raise ValueError(f"unknown comparison {op!r}")
    return table[op]


def number_range(
    low: Decimal | None,
    low_exclusive: bool,
    high: Decimal | None,
    high_exclusive: bool,
) -> Regex:
    """Fixed-point decimal text of the numbers in the given (half-)open or closed range."""
    parts = []
    if high is None or high > 0 or (high == 0 and not high_exclusive):
        constraints = []
        if low is not None and (low > 0 or (low == 0 and low_exclusive)):
            constraints.append(decimal_compare(low, "gt" if low_exclusive else "ge"))
        if high is not None:
            constraints.append(decimal_compare(high, "lt" if high_exclusive else "le"))
        parts.append(inter(*constraints) if constraints else NONNEG_DECIMAL)
    if low is None or low < 0:
        constraints = [decimal_compare(Decimal(0), "gt")]
        if low is not None:
            constraints.append(decimal_compare(-low, "lt" if low_exclusive else "le"))
        if high is not None and high < 0:
            constraints.append(decimal_compare(-high, "gt" if high_exclusive else "ge"))
        parts.append(cat(_MINUS, inter(*constraints)))
    return alt(*parts)
