"""Output validation with ``jsonschema`` where ``pattern`` is checked by the in-repo DFA.

``jsonschema`` implements ``pattern`` with Python's backtracking ``re``, which must never see an
untrusted pattern (ReDoS). :func:`safe_validator` extends the Draft 2020-12 validator and replaces
the ``pattern`` keyword with a DFA search compiled by ``automata/pattern.py`` under the caller's
:class:`CompileLimits`. Every other keyword our subset accepts is regex-free.

The service only validates against schemas that ``parse_schema`` accepted (which excludes
``format``, ``patternProperties`` and non-local ``$ref``), but this module does not rely on it:

- validators are built with a ``referencing`` registry whose ``retrieve`` always refuses, so a
  remote (``http://``) or local-file (``file://``) reference is never fetched or read;
- an unresolvable reference, a pattern that fails to compile within the limits (unsupported
  syntax, timeout, state caps) and runaway recursion all become ``valid=False`` outcomes with a
  short reason, never an exception.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from jsonschema import Draft202012Validator, ValidationError, validators
from jsonschema.exceptions import SchemaError as JSONSchemaError
from jsonschema.protocols import Validator
from referencing import Registry
from referencing.exceptions import Unresolvable

from grammar_constrained_decoder.automata.limits import DEFAULT_LIMITS, CompileError, CompileLimits
from grammar_constrained_decoder.core.canonical import CanonicalJSONError, canonical_json
from grammar_constrained_decoder.schema.canonical import pattern_search

__all__ = [
    "NO_REMOTE_REGISTRY",
    "SafeValidator",
    "ValidationOutcome",
    "safe_validator",
    "validate_output",
]


# A bare ``referencing`` registry refuses to retrieve any URI (its default ``retrieve`` raises
# ``NoSuchResource``). ``jsonschema``'s own default registry instead fetches unknown references
# with ``urllib`` (``http://`` and ``file://``), which is why every validator here gets this one.
# tests/unit/test_validate.py pins the behavior.
NO_REMOTE_REGISTRY: Registry[Any] = Registry()
_extend = cast(Callable[..., type[Validator]], validators.extend)


@lru_cache(maxsize=8)
def _validator_class(limits: CompileLimits) -> type[Validator]:
    def dfa_pattern(
        validator: Validator, pattern: Any, instance: Any, schema: Any
    ) -> Iterator[ValidationError]:
        del schema
        if validator.is_type(instance, "string") and not pattern_search(
            str(pattern), instance, limits
        ):
            yield ValidationError("string does not match the pattern")

    return _extend(Draft202012Validator, {"pattern": dfa_pattern})


SafeValidator = _validator_class(DEFAULT_LIMITS)


@lru_cache(maxsize=128)
def _cached(schema_text: str, limits: CompileLimits) -> Validator:
    return _validator_class(limits)(json.loads(schema_text), registry=NO_REMOTE_REGISTRY)


def safe_validator(schema: dict[str, Any], limits: CompileLimits = DEFAULT_LIMITS) -> Validator:
    return _cached(canonical_json(schema), limits)


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    parsed: bool
    valid: bool
    error: str | None


def _failure_reason(exc: Exception) -> str | None:
    """Short, value-free reason for an exception raised while evaluating a schema."""
    if isinstance(exc, CompileError):
        return f"pattern rejected: {exc.reason}"
    if isinstance(exc, Unresolvable):
        return "schema reference could not be resolved"
    if isinstance(exc, JSONSchemaError | CanonicalJSONError):
        return "schema is not valid"
    if isinstance(exc, RecursionError):
        return "validation exceeded the recursion limit"
    return None


def validate_output(
    schema: dict[str, Any], text: str, *, limits: CompileLimits = DEFAULT_LIMITS
) -> ValidationOutcome:
    """Parse ``text`` as JSON and validate it; errors are short and never echo the value."""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return ValidationOutcome(False, False, "not valid JSON")
    try:
        error = next(iter(safe_validator(schema, limits).iter_errors(value)), None)
    except (CompileError, Unresolvable, JSONSchemaError, CanonicalJSONError, RecursionError) as exc:
        return ValidationOutcome(True, False, _failure_reason(exc))
    if error is None:
        return ValidationOutcome(True, True, None)
    location = "/".join(str(part) for part in error.absolute_path)[:80]
    return ValidationOutcome(True, False, f"keyword {error.validator} failed at /{location}")
