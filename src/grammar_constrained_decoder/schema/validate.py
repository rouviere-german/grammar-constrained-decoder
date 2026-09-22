"""Output validation with ``jsonschema`` where ``pattern`` is checked by the in-repo DFA.

``jsonschema`` implements ``pattern`` with Python's backtracking ``re``, which must never see an
untrusted pattern (ReDoS). :func:`safe_validator` extends the Draft 2020-12 validator and replaces
the ``pattern`` keyword with a DFA search compiled by ``automata/pattern.py``. Every other keyword
our subset accepts is regex-free. Schemas reach this module only after ``parse_schema`` accepted
them, so ``format``, ``patternProperties`` and remote ``$ref`` are already excluded.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator, ValidationError, validators
from jsonschema.protocols import Validator

from grammar_constrained_decoder.core.canonical import canonical_json
from grammar_constrained_decoder.schema.canonical import pattern_search

__all__ = ["SafeValidator", "ValidationOutcome", "safe_validator", "validate_output"]


def _dfa_pattern(
    validator: Validator, pattern: Any, instance: Any, schema: Any
) -> Iterator[ValidationError]:
    del schema
    if validator.is_type(instance, "string") and not pattern_search(str(pattern), instance):
        yield ValidationError("string does not match the pattern")


SafeValidator = validators.extend(  # type: ignore[no-untyped-call]
    Draft202012Validator, {"pattern": _dfa_pattern}
)


@lru_cache(maxsize=128)
def _cached(schema_text: str) -> Validator:
    validator: Validator = SafeValidator(json.loads(schema_text))
    return validator


def safe_validator(schema: dict[str, Any]) -> Validator:
    return _cached(canonical_json(schema))


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    parsed: bool
    valid: bool
    error: str | None


def validate_output(schema: dict[str, Any], text: str) -> ValidationOutcome:
    """Parse ``text`` as JSON and validate it; errors are short and never echo the value."""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return ValidationOutcome(False, False, "not valid JSON")
    error = next(iter(safe_validator(schema).iter_errors(value)), None)
    if error is None:
        return ValidationOutcome(True, True, None)
    location = "/".join(str(part) for part in error.absolute_path)[:80]
    return ValidationOutcome(True, False, f"keyword {error.validator} failed at /{location}")
