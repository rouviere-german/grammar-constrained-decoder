"""The in-repo ECMA pattern subset: differential tests against Python re (on trusted, generated
patterns only) and the rejection matrix for unsupported syntax."""

from __future__ import annotations

import json
import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from grammar_constrained_decoder.automata.dfa import compile_regex
from grammar_constrained_decoder.automata.limits import CompileLimits, SchemaError
from grammar_constrained_decoder.automata.pattern import compile_pattern
from grammar_constrained_decoder.schema.lower import json_string_class
from grammar_constrained_decoder.testing.strategies import (
    PATTERN_TEST_ALPHABET,
    ecma_patterns,
    text_samples,
)

pytestmark = pytest.mark.unit

words = st.lists(text_samples(PATTERN_TEST_ALPHABET, max_size=8), max_size=15)


@given(ecma_patterns(), words)
def test_full_match_semantics_agree_with_python_re(body: str, samples: list[str]) -> None:
    dfa = compile_regex(compile_pattern(f"^(?:{body})$"))
    oracle = re.compile(body, re.ASCII)
    for text in samples:
        assert dfa.accepts(text.encode("utf-8")) == (oracle.fullmatch(text) is not None), (
            body,
            text,
        )


@given(ecma_patterns(), words)
def test_search_semantics_agree_with_python_re(body: str, samples: list[str]) -> None:
    dfa = compile_regex(compile_pattern(body))
    oracle = re.compile(body, re.ASCII)
    for text in samples:
        assert dfa.accepts(text.encode("utf-8")) == (oracle.search(text) is not None), (body, text)


@given(ecma_patterns(), words)
def test_patterns_inside_json_strings_match_the_decoded_value(
    body: str, samples: list[str]
) -> None:
    dfa = compile_regex(compile_pattern(f"^(?:{body})$", encoder=json_string_class))
    oracle = re.compile(body, re.ASCII)
    for text in samples:
        encoded = json.dumps(text, ensure_ascii=False)[1:-1].encode("utf-8")
        assert dfa.accepts(encoded) == (oracle.fullmatch(text) is not None), (body, text)


@pytest.mark.parametrize(
    ("pattern", "accepted", "rejected"),
    [
        (r"^\d{3}$", ["123"], ["12", "1234", "١٢٣"]),
        (r"^\u{1F600}+$", ["😀😀"], ["", "x"]),
        (r"^[à-ÿ]$", ["é"], ["e"]),
        (r"^\x41\t\n$", ["A\t\n"], ["A"]),
        (r"^[^\s]+$", ["abc"], ["a b", "a\u3000b"]),
        (r"^.$", ["a", "中"], ["\n", "\r", "\u2028"]),
        (r"^[-a]$", ["-", "a"], ["b"]),
        (r"^[a-]$", ["-", "a"], ["b"]),
        (r"^[\b]$", ["\b"], ["b"]),
        (r"^[\-\]]$", ["-", "]"], ["a"]),
        (r"^a{2,}?$", ["aa", "aaaa"], ["a"]),
        (r"^(?:ab|cd)\/$", ["ab/", "cd/"], ["ab"]),
        (r"^[]$", [], ["", "a"]),
        (r"^[^]$", ["a", "\n"], ["", "ab"]),
        (r"^x|y$", ["x", "xz", "zy"], ["z"]),
        (r"^\0$", ["\x00"], ["0"]),
    ],
)
def test_supported_constructs(pattern: str, accepted: list[str], rejected: list[str]) -> None:
    dfa = compile_regex(compile_pattern(pattern))
    for text in accepted:
        assert dfa.accepts(text.encode("utf-8")), text
    for text in rejected:
        assert not dfa.accepts(text.encode("utf-8")), text


@pytest.mark.parametrize(
    ("pattern", "message"),
    [
        (r"(a)\1", "backreference"),
        (r"(?=a)", "lookaround"),
        (r"(?!a)", "lookaround"),
        (r"(?<=a)b", "lookaround"),
        (r"(?<name>a)", "lookaround or a named group"),
        (r"\bword", "word boundaries"),
        (r"\p{L}", "property"),
        (r"\cJ", "control-letter"),
        (r"\k<n>", "named backreferences"),
        (r"a^b", "'^' after the start"),
        (r"a$b", "'$' before the end"),
        (r"(a$)", "'$' before the end"),
        (r"*a", "nothing to repeat"),
        (r"a**", "nothing to repeat"),
        (r"a{", "malformed"),
        (r"a{2,1}", "below its minimum"),
        (r"{", "lone '{'"),
        (r"a}", "lone '}'"),
        (r"a]", "lone ']'"),
        (r"(ab", "unclosed group"),
        (r"ab)", "unbalanced"),
        (r"[ab", "unclosed character class"),
        (r"[z-a]", "out of order"),
        (r"[\d-z]", "range endpoint"),
        (r"\ud800", "not a Unicode scalar"),
        (r"\u{110000}", "not a Unicode scalar"),
        (r"\u{}", "malformed"),
        (r"\x4", "truncated"),
        (r"\xZZ", "malformed hex"),
        (r"\q", "unknown escape"),
        (r"\01", "octal"),
        ("a\\", "ends unexpectedly"),
    ],
)
def test_unsupported_syntax_is_rejected_explicitly(pattern: str, message: str) -> None:
    with pytest.raises(SchemaError, match=re.escape(message)) as info:
        compile_pattern(pattern)
    assert info.value.reason == "pattern"


def test_pattern_limits() -> None:
    limits = CompileLimits(max_pattern_length=10, max_repeat=5)
    with pytest.raises(SchemaError) as long_info:
        compile_pattern("a" * 11, limits=limits)
    assert long_info.value.reason == "pattern_length"
    with pytest.raises(SchemaError) as repeat_info:
        compile_pattern("a{6}", limits=limits)
    assert repeat_info.value.reason == "repeat_bound"
    with pytest.raises(SchemaError, match="nests groups"):
        compile_pattern("(" * 17 + "a" + ")" * 17)
    with pytest.raises(SchemaError, match="too long"):
        compile_pattern("a{1234567}")


def test_python_re_is_never_used_to_compile_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Python re was called on a schema pattern")

    monkeypatch.setattr(re, "compile", forbidden)
    monkeypatch.setattr(re, "search", forbidden)
    monkeypatch.setattr(re, "fullmatch", forbidden)
    dfa = compile_regex(compile_pattern("^(a+)+[0-9]{2}$"))
    assert dfa.accepts(b"aaa12")
