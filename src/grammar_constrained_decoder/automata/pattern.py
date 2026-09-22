"""In-repo parser for a safe subset of ECMA-262 regular expressions (JSON Schema ``pattern``).

Untrusted patterns are never handed to Python's ``re``: this parser turns them into the byte-level
regex AST, which compiles to a DFA that matches in linear time (no backtracking, so no ReDoS).

Semantics follow JSON Schema: a pattern is *searched* for, not fully matched, so every top-level
branch without ``^`` gets an any-string prefix and every branch without ``$`` an any-string suffix.
Characters are Unicode code points; the ``encoder`` callback turns each code point class into
bytes (raw UTF-8 for plain text, or the canonical JSON string encoding inside a document).

Supported: literals, ``.``, classes ``[...]``/``[^...]`` with ranges, ``\\d \\D \\w \\W \\s \\S``,
control and hex escapes (``\\t \\n \\r \\f \\v \\0 \\xHH \\uHHHH \\u{H...}``), escaped syntax
characters, groups ``(...)``/``(?:...)``, alternation, quantifiers ``* + ? {m} {m,} {m,n}``
(lazy forms accepted; laziness does not change the language), and ``^``/``$`` at the start/end of
a top-level branch.

Rejected with explicit errors: backreferences, lookaround, named groups, word boundaries,
Unicode property escapes, control-letter escapes, anchors in other positions, lone ``{ } ]``,
quantifiers without an operand and patterns or repetition counts over the configured limits.
"""

from __future__ import annotations

from collections.abc import Callable

from grammar_constrained_decoder.automata.charset import (
    MAX_SCALAR,
    SURROGATES,
    byte_range,
    complement_ranges,
    normalize_ranges,
    utf8_sequences,
)
from grammar_constrained_decoder.automata.limits import DEFAULT_LIMITS, CompileLimits, SchemaError
from grammar_constrained_decoder.automata.regex import (
    EPS,
    Regex,
    alt,
    byteset,
    cat,
    rep,
    star,
)

__all__ = [
    "ANY_CHAR",
    "DIGIT",
    "DOT",
    "SPACE",
    "WORD",
    "Encoder",
    "Ranges",
    "compile_pattern",
    "utf8_class",
]

Ranges = list[tuple[int, int]]
Encoder = Callable[[Ranges], Regex]

DIGIT: Ranges = [(0x30, 0x39)]
WORD: Ranges = [(0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A)]
SPACE: Ranges = [
    (0x09, 0x0D),
    (0x20, 0x20),
    (0xA0, 0xA0),
    (0x1680, 0x1680),
    (0x2000, 0x200A),
    (0x2028, 0x2029),
    (0x202F, 0x202F),
    (0x205F, 0x205F),
    (0x3000, 0x3000),
    (0xFEFF, 0xFEFF),
]
ANY_CHAR: Ranges = complement_ranges([])
DOT: Ranges = complement_ranges([(0x0A, 0x0A), (0x0D, 0x0D), (0x2028, 0x2029)])
_MAX_GROUP_DEPTH = 16
_SYNTAX = frozenset("^$\\.*+?()[]{}|/-")
_CONTROL_ESCAPES = {"t": 0x09, "n": 0x0A, "r": 0x0D, "f": 0x0C, "v": 0x0B}
_CLASS_ESCAPES: dict[str, tuple[Ranges, bool]] = {
    "d": (DIGIT, False),
    "D": (DIGIT, True),
    "w": (WORD, False),
    "W": (WORD, True),
    "s": (SPACE, False),
    "S": (SPACE, True),
}
_UNSUPPORTED_ESCAPES = {
    "b": "word boundaries are not supported",
    "B": "word boundaries are not supported",
    "k": "named backreferences are not supported",
    "p": "Unicode property escapes are not supported",
    "P": "Unicode property escapes are not supported",
    "c": "control-letter escapes are not supported",
}


def utf8_class(ranges: Ranges) -> Regex:
    """Raw UTF-8 encoding of a code point class."""
    branches = []
    for lo, hi in normalize_ranges(ranges):
        for sequence in utf8_sequences(lo, hi):
            branches.append(cat(*(byteset(byte_range(a, b)) for a, b in sequence)))
    return alt(*branches)


def _single(ranges: Ranges) -> bool:
    return len(ranges) == 1 and ranges[0][0] == ranges[0][1]


class _Parser:
    def __init__(self, pattern: str, encoder: Encoder, limits: CompileLimits, path: str) -> None:
        self.text = pattern
        self.pos = 0
        self.encoder = encoder
        self.limits = limits
        self.path = path
        self.depth = 0
        self._cache: dict[tuple[tuple[int, int], ...], Regex] = {}

    # -- helpers ---------------------------------------------------------------------------------

    def error(self, message: str) -> SchemaError:
        return SchemaError(self.path, f"pattern {message} (at offset {self.pos})", reason="pattern")

    def peek(self, offset: int = 0) -> str | None:
        index = self.pos + offset
        return self.text[index] if index < len(self.text) else None

    def take(self) -> str:
        char = self.peek()
        if char is None:
            raise self.error("ends unexpectedly")
        self.pos += 1
        return char

    def chars(self, ranges: Ranges) -> Regex:
        key = tuple(normalize_ranges(ranges))
        cached = self._cache.get(key)
        if cached is None:
            cached = self._cache[key] = self.encoder(list(key))
        return cached

    # -- grammar ---------------------------------------------------------------------------------

    def parse(self) -> Regex:
        any_string = star(self.chars(ANY_CHAR))
        branches = []
        while True:
            anchored_start = self.peek() == "^"
            if anchored_start:
                self.pos += 1
            body = self.sequence(top_level=True)
            anchored_end = self.peek() == "$"
            if anchored_end:
                self.pos += 1
                if self.peek() not in (None, "|"):
                    raise self.error("uses '$' before the end of a branch")
            prefix = EPS if anchored_start else any_string
            suffix = EPS if anchored_end else any_string
            branches.append(cat(prefix, body, suffix))
            if self.peek() != "|":
                break
            self.pos += 1
        if self.peek() is not None:
            raise self.error("has an unbalanced ')'")
        return alt(*branches)

    def alternation(self) -> Regex:
        branches = [self.sequence(top_level=False)]
        while self.peek() == "|":
            self.pos += 1
            branches.append(self.sequence(top_level=False))
        return alt(*branches)

    def sequence(self, *, top_level: bool) -> Regex:
        items: list[Regex] = []
        while True:
            char = self.peek()
            if char is None or char in "|)":
                break
            if char == "$":
                if top_level and self.peek(1) in (None, "|"):
                    break
                raise self.error("uses '$' before the end of a branch")
            if char == "^":
                raise self.error("uses '^' after the start of a branch")
            items.append(self.quantified())
        return cat(*items)

    def quantified(self) -> Regex:
        atom = self.atom()
        char = self.peek()
        if char is None or char not in "*+?{":
            return atom
        lo, hi = self.quantifier()
        if self.peek() == "?":
            self.pos += 1  # lazy quantifier: same language
        following = self.peek()
        if following is not None and following in "*+?{":
            raise self.error("has a quantifier with nothing to repeat")
        return rep(atom, lo, hi)

    def quantifier(self) -> tuple[int, int | None]:
        char = self.take()
        if char == "*":
            return 0, None
        if char == "+":
            return 1, None
        if char == "?":
            return 0, 1
        start = self.pos
        lo = self.number()
        hi: int | None = lo
        if self.peek() == ",":
            self.pos += 1
            hi = None if self.peek() == "}" else self.number()
        if self.peek() != "}" or self.pos == start:
            raise self.error("has a malformed '{m,n}' quantifier (escape a literal '{')")
        self.pos += 1
        if hi is not None and hi < lo:
            raise self.error("has a quantifier whose maximum is below its minimum")
        bound = hi if hi is not None else lo
        if bound > self.limits.max_repeat:
            raise SchemaError(
                self.path,
                f"pattern repetition count exceeds the limit of {self.limits.max_repeat}",
                reason="repeat_bound",
            )
        return lo, hi

    def number(self) -> int:
        start = self.pos
        while (char := self.peek()) is not None and char.isascii() and char.isdigit():
            self.pos += 1
            if self.pos - start > 6:
                raise self.error("has a repetition count that is too long")
        if self.pos == start:
            raise self.error("has a malformed '{m,n}' quantifier (escape a literal '{')")
        return int(self.text[start : self.pos])

    def atom(self) -> Regex:
        char = self.take()
        if char == "(":
            return self.group()
        if char == "[":
            return self.chars(self.char_class())
        if char == ".":
            return self.chars(DOT)
        if char == "\\":
            ranges = self.escape(in_class=False)
            return self.chars(ranges)
        if char in "*+?":
            raise self.error("has a quantifier with nothing to repeat")
        if char == "{":
            raise self.error("has a lone '{' (escape it as '\\{')")
        if char in "}]":
            raise self.error(f"has a lone '{char}' (escape it)")
        code = ord(char)
        return self.chars([(code, code)])

    def group(self) -> Regex:
        if self.peek() == "?":
            if self.peek(1) == ":":
                self.pos += 2
            else:
                raise self.error("uses lookaround or a named group, which are not supported")
        self.depth += 1
        if self.depth > _MAX_GROUP_DEPTH:
            raise SchemaError(
                self.path,
                f"pattern nests groups deeper than {_MAX_GROUP_DEPTH}",
                reason="pattern",
            )
        inner = self.alternation()
        if self.peek() != ")":
            raise self.error("has an unclosed group")
        self.pos += 1
        self.depth -= 1
        return inner

    def char_class(self) -> Ranges:
        negated = self.peek() == "^"
        if negated:
            self.pos += 1
        ranges: Ranges = []
        while True:
            char = self.peek()
            if char is None:
                raise self.error("has an unclosed character class")
            if char == "]":
                self.pos += 1
                break
            low_set = self.class_atom()
            if self.peek() == "-" and self.peek(1) not in (None, "]"):
                self.pos += 1
                high_set = self.class_atom()
                if not (_single(low_set) and _single(high_set)):
                    raise self.error("uses a class escape as a range endpoint")
                low, high = low_set[0][0], high_set[0][0]
                if low > high:
                    raise self.error("has a character range out of order")
                ranges.append((low, high))
            else:
                ranges.extend(low_set)
        return complement_ranges(ranges) if negated else normalize_ranges(ranges)

    def class_atom(self) -> Ranges:
        char = self.take()
        if char == "\\":
            if self.peek() == "b":
                self.pos += 1
                return [(0x08, 0x08)]
            if self.peek() == "-":
                self.pos += 1
                return [(0x2D, 0x2D)]
            return self.escape(in_class=True)
        code = ord(char)
        return [(code, code)]

    def escape(self, *, in_class: bool) -> Ranges:  # noqa: PLR0911 - one branch per escape family
        char = self.take()
        if char in _CLASS_ESCAPES:
            base, negate = _CLASS_ESCAPES[char]
            return complement_ranges(base) if negate else list(base)
        if char in _CONTROL_ESCAPES:
            code = _CONTROL_ESCAPES[char]
            return [(code, code)]
        if char == "0":
            if (nxt := self.peek()) is not None and nxt.isdigit():
                raise self.error("uses an octal escape, which is not supported")
            return [(0, 0)]
        if char.isdigit():
            raise self.error("uses a backreference, which is not supported")
        if char == "x":
            return self.code_point(self.hex_digits(2))
        if char == "u":
            if self.peek() == "{":
                self.pos += 1
                start = self.pos
                while (nxt := self.peek()) is not None and nxt != "}":
                    self.pos += 1
                digits = self.text[start : self.pos]
                if self.peek() != "}" or not digits or len(digits) > 6:
                    raise self.error("has a malformed \\u{...} escape")
                self.pos += 1
                return self.code_point(self.parse_hex(digits))
            return self.code_point(self.hex_digits(4))
        if char in _UNSUPPORTED_ESCAPES:
            raise self.error(f"uses '\\{char}': {_UNSUPPORTED_ESCAPES[char]}")
        if char in _SYNTAX:
            code = ord(char)
            return [(code, code)]
        del in_class
        raise self.error(f"uses the unknown escape '\\{char}'")

    def hex_digits(self, count: int) -> int:
        digits = self.text[self.pos : self.pos + count]
        if len(digits) != count:
            raise self.error("has a truncated hex escape")
        self.pos += count
        return self.parse_hex(digits)

    def parse_hex(self, digits: str) -> int:
        if not all(char in "0123456789abcdefABCDEF" for char in digits):
            raise self.error("has a malformed hex escape")
        return int(digits, 16)

    def code_point(self, value: int) -> Ranges:
        if value > MAX_SCALAR or SURROGATES[0] <= value <= SURROGATES[1]:
            raise self.error("escapes a value that is not a Unicode scalar value")
        return [(value, value)]


def compile_pattern(
    pattern: str,
    *,
    encoder: Encoder = utf8_class,
    limits: CompileLimits = DEFAULT_LIMITS,
    path: str = "/pattern",
) -> Regex:
    """Parse ``pattern`` into a regex matching every string in which the pattern is found."""
    if len(pattern) > limits.max_pattern_length:
        raise SchemaError(
            path,
            f"pattern is longer than {limits.max_pattern_length} characters",
            reason="pattern_length",
        )
    return _Parser(pattern, encoder, limits, path).parse()
