"""Byte sets as 256-bit integers, and code point ranges lowered to UTF-8 byte-range sequences.

Automata in this package run over bytes, so every character class is eventually a set of byte
sequences. A byte set is a Python ``int`` whose bit ``b`` is set when byte ``b`` is a member, which
makes union, intersection and complement single integer operations.

:func:`utf8_sequences` splits a code point range into sequences of byte ranges such that a UTF-8
string of one code point matches exactly one sequence if and only if the code point is in the
range. The splitting follows the construction used by RE2 and Rust's ``regex-syntax``: split the
range at encoded-length boundaries, then at 6-bit continuation boundaries, until the first and last
code point of each piece differ only in positions where every byte value between them is valid.
Surrogates (U+D800-U+DFFF) are never produced because they are not Unicode scalar values.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

__all__ = [
    "ALL_BYTES",
    "MAX_SCALAR",
    "SURROGATES",
    "byte_mask",
    "byte_range",
    "complement_ranges",
    "iter_bytes",
    "mask_ranges",
    "normalize_ranges",
    "utf8_sequences",
]

ALL_BYTES = (1 << 256) - 1
MAX_SCALAR = 0x10FFFF
SURROGATES = (0xD800, 0xDFFF)
_ENCODED_LENGTH_LIMITS = (0x7F, 0x7FF, 0xFFFF, MAX_SCALAR)


def byte_range(lo: int, hi: int) -> int:
    """Mask of the bytes ``lo..hi`` inclusive (empty when ``lo > hi``)."""
    if not (0 <= lo <= 255 and 0 <= hi <= 255):
        raise ValueError("byte values must be in 0..255")
    if lo > hi:
        return 0
    return ((1 << (hi - lo + 1)) - 1) << lo


def byte_mask(values: Iterable[int]) -> int:
    """Mask of an iterable of byte values."""
    mask = 0
    for value in values:
        if not 0 <= value <= 255:
            raise ValueError("byte values must be in 0..255")
        mask |= 1 << value
    return mask


def iter_bytes(mask: int) -> Iterator[int]:
    """Members of a byte mask in increasing order."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def mask_ranges(mask: int) -> list[tuple[int, int]]:
    """Maximal runs of consecutive members, as inclusive ``(lo, hi)`` pairs."""
    ranges: list[tuple[int, int]] = []
    for value in iter_bytes(mask):
        if ranges and ranges[-1][1] == value - 1:
            ranges[-1] = (ranges[-1][0], value)
        else:
            ranges.append((value, value))
    return ranges


def normalize_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and merge overlapping or adjacent inclusive code point ranges."""
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(ranges):
        if lo > hi:
            continue
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def complement_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Unicode scalar values (surrogates excluded) that are not in ``ranges``."""
    result: list[tuple[int, int]] = []
    cursor = 0
    for lo, hi in normalize_ranges([*ranges, SURROGATES]):
        if lo > cursor:
            result.append((cursor, lo - 1))
        cursor = max(cursor, hi + 1)
    if cursor <= MAX_SCALAR:
        result.append((cursor, MAX_SCALAR))
    return result


def _without_surrogates(lo: int, hi: int) -> list[tuple[int, int]]:
    if hi < SURROGATES[0] or lo > SURROGATES[1]:
        return [(lo, hi)]
    pieces = []
    if lo < SURROGATES[0]:
        pieces.append((lo, SURROGATES[0] - 1))
    if hi > SURROGATES[1]:
        pieces.append((SURROGATES[1] + 1, hi))
    return pieces


def _encode(code_point: int) -> bytes:
    return chr(code_point).encode("utf-8")


def _split_same_length(lo: int, hi: int, out: list[list[tuple[int, int]]]) -> None:
    """Emit byte-range sequences for ``lo..hi`` whose code points share one encoded length."""
    stack = [(lo, hi)]
    while stack:
        start, end = stack.pop()
        length = len(_encode(start))
        for position in range(1, length):
            mask = (1 << (6 * position)) - 1
            if start & ~mask != end & ~mask:
                if start & mask:
                    stack.append(((start | mask) + 1, end))
                    stack.append((start, start | mask))
                    break
                if end & mask != mask:
                    stack.append((end & ~mask, end))
                    stack.append((start, (end & ~mask) - 1))
                    break
        else:
            first, last = _encode(start), _encode(end)
            out.append(list(zip(first, last, strict=True)))


def utf8_sequences(lo: int, hi: int) -> list[list[tuple[int, int]]]:
    """Byte-range sequences matching exactly the UTF-8 encodings of code points ``lo..hi``.

    Each sequence is a list of inclusive ``(byte_lo, byte_hi)`` pairs, one per encoded byte.
    Sequences are disjoint and returned in increasing code point order.
    """
    if not 0 <= lo <= MAX_SCALAR or not 0 <= hi <= MAX_SCALAR:
        raise ValueError("code points must be in 0..0x10FFFF")
    if lo > hi:
        return []
    sequences: list[list[tuple[int, int]]] = []
    for start, end in _without_surrogates(lo, hi):
        cursor = start
        for limit in _ENCODED_LENGTH_LIMITS:
            if cursor > end:
                break
            if cursor <= limit:
                piece_end = min(end, limit)
                pending: list[list[tuple[int, int]]] = []
                _split_same_length(cursor, piece_end, pending)
                sequences.extend(sorted(pending))
                cursor = piece_end + 1
    return sequences
