"""Output validation: no reference is ever fetched, pattern compilation is bounded, and every
failure (unsupported pattern, timeout, lone surrogate, runaway recursion) is a typed invalid
outcome instead of an exception."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from grammar_constrained_decoder.automata.limits import CompileLimits
from grammar_constrained_decoder.schema.canonical import pattern_search
from grammar_constrained_decoder.schema.validate import validate_output

pytestmark = pytest.mark.unit


def test_remote_references_are_never_fetched(schema_server: tuple[str, list[str]]) -> None:
    base, hits = schema_server
    outcome = validate_output({"$ref": f"{base}/evil.json"}, "1")
    assert hits == []
    assert outcome.valid is False
    assert outcome.error is not None
    assert "reference" in outcome.error


def test_file_references_are_never_read(tmp_path: Path) -> None:
    target = tmp_path / "integer.json"
    target.write_text(json.dumps({"type": "integer"}), encoding="utf-8")
    present = validate_output({"$ref": target.as_uri()}, "1")
    missing = validate_output({"$ref": (tmp_path / "missing.json").as_uri()}, "1")
    # Same answer whether or not the file exists and whatever it contains: no read, no oracle.
    assert present == missing
    assert present.valid is False


def test_unsupported_patterns_are_invalid_outcomes_not_exceptions() -> None:
    outcome = validate_output({"type": "string", "pattern": "(?=a)a"}, '"a"')
    assert outcome.parsed is True
    assert outcome.valid is False
    assert outcome.error is not None
    assert "pattern" in outcome.error


def test_pattern_compilation_honours_the_callers_limits() -> None:
    schema = {"type": "string", "pattern": "^(((a|b){0,60}){0,60}){0,60}$"}
    limits = CompileLimits(timeout_seconds=0.3)
    started = time.perf_counter()
    outcome = validate_output(schema, '"ab"', limits=limits)
    elapsed = time.perf_counter() - started
    assert outcome.valid is False
    assert outcome.error in {
        f"pattern rejected: {reason}"
        for reason in ("timeout", "subset_size", "nfa_states", "dfa_states")
    }
    assert elapsed < 2.0


def test_lone_surrogates_in_outputs_are_rejected_without_crashing() -> None:
    assert pattern_search("a", "a\ud800") is False
    outcome = validate_output({"type": "string", "pattern": "a"}, '"a\\ud800"')
    assert outcome.valid is False


def test_runaway_recursion_is_reported_not_raised() -> None:
    schema = {
        "$defs": {"a": {"type": "array", "items": {"$ref": "#/$defs/a"}, "maxItems": 1}},
        "$ref": "#/$defs/a",
    }
    text = "[" * 900 + "]" * 900
    outcome = validate_output(schema, text)
    assert outcome.valid is False
    assert outcome.error is not None


def test_valid_and_invalid_documents() -> None:
    schema = {
        "type": "object",
        "properties": {"code": {"type": "string", "pattern": "^[A-Z]{2}$"}},
        "required": ["code"],
    }
    assert validate_output(schema, '{"code":"AB"}').valid is True
    wrong = validate_output(schema, '{"code":"ab"}')
    assert wrong.valid is False
    assert wrong.error == "keyword pattern failed at /code"
    assert validate_output(schema, "{").parsed is False
