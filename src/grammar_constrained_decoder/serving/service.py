"""Generation service: request limits, schema resolution and compilation, decoding, validation.

Everything here is synchronous and CPU-bound; the API runs it in worker threads. The service never
logs prompts or outputs: log events carry lengths, truncated SHA-256 digests, the schema label and
timings only.

Every client-supplied schema (inline on ``/v1/generate`` or being registered) goes through
:meth:`GenerationService.compile_schema` before any token is generated, whether or not decoding is
constrained: the JSON document check, the size cap, the subset parser (no remote ``$ref``, no
``format``/``patternProperties``), lowering and compilation under the service's limits, and a
bounded compile of every ``pattern`` that output validation will use. Any failure is a typed
``400``/``413`` with a metric, never a ``500``. Rejections are remembered per schema digest for a
few minutes, so replaying a schema that timed out costs a dictionary lookup, not another compile.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from grammar_constrained_decoder.automata.limits import CompileBudget, CompileError, CompileLimits
from grammar_constrained_decoder.core.canonical import (
    CanonicalJSONError,
    canonical_json,
    sha256_hex,
)
from grammar_constrained_decoder.core.config import Settings
from grammar_constrained_decoder.core.logging import get_logger
from grammar_constrained_decoder.decoding.compiler import CompiledGrammar, GrammarCompiler
from grammar_constrained_decoder.decoding.decoder import Decoder, StepEvent
from grammar_constrained_decoder.schema.canonical import pattern_dfa
from grammar_constrained_decoder.schema.ir import (
    SchemaIR,
    String,
    check_json_document,
    iter_children,
)
from grammar_constrained_decoder.schema.validate import validate_output
from grammar_constrained_decoder.serving.metrics import DecoderMetrics, PromptDriftMonitor
from grammar_constrained_decoder.serving.schema_registry import (
    SchemaIntegrityError,
    SchemaNotFoundError,
    SchemaRegistry,
)
from grammar_constrained_decoder.training.pipeline import LoadedModel

__all__ = ["GenerationService", "Prepared", "ServiceError", "schema_digest"]

_log = get_logger("grammar_constrained_decoder.service")


_REJECTION_CACHE_SIZE = 256
_REJECTION_TTL_SECONDS = 300.0


class ServiceError(Exception):
    """A client-visible failure with an HTTP status and a stable code. Messages are fixed text."""

    def __init__(self, status: int, code: str, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.reason = reason if reason is not None else code


def schema_digest(schema: Any) -> str:
    """SHA-256 of the canonical JSON of ``schema``; never raises for a decoded request body.

    Values that canonical JSON refuses (lone surrogates, NaN) are hashed in an ASCII-escaped form,
    so a rejected schema still gets a stable digest for the audit log and the rejection cache.
    """
    try:
        return sha256_hex(canonical_json(schema))
    except (CanonicalJSONError, ValueError, RecursionError):
        pass
    try:
        text = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError, RecursionError):
        text = f"unencodable:{type(schema).__name__}"
    return sha256_hex("fallback:" + text)


@dataclass(frozen=True)
class GenerationParams:
    prompt: str
    schema_id: str | None
    inline_schema: dict[str, Any] | None
    constrained: bool
    strategy: Literal["greedy", "top_p", "beam"]
    max_tokens: int
    temperature: float
    top_p: float
    beam_width: int
    seed: int | None


@dataclass(frozen=True)
class Prepared:
    params: GenerationParams
    prompt_ids: list[int]
    schema: dict[str, Any] | None
    schema_label: str
    grammar: CompiledGrammar | None


def _digest(text: str) -> str:
    return sha256_hex(text)[:16]


def _patterns(ir: SchemaIR) -> list[str]:
    """Distinct ``pattern`` values in the schema, sorted (iterative walk of the IR)."""
    found: set[str] = set()
    stack = [ir.root, *ir.definitions.values()]
    while stack:
        node = stack.pop()
        if isinstance(node, String) and node.pattern is not None:
            found.add(node.pattern)
        stack.extend(iter_children(node))
    return sorted(found)


def _rejection(exc: CompileError) -> ServiceError:
    return ServiceError(
        400, "schema_rejected", f"{exc.reason}: {str(exc)[:300]}", reason=exc.reason
    )


class GenerationService:
    def __init__(
        self,
        settings: Settings,
        model: LoadedModel,
        *,
        schemas: SchemaRegistry,
        metrics: DecoderMetrics,
    ) -> None:
        self.settings = settings
        self.model = model
        self.schemas = schemas
        self.metrics = metrics
        self.limits = CompileLimits(
            max_depth=settings.compile_max_depth,
            max_dfa_states=settings.compile_max_dfa_states,
            timeout_seconds=settings.compile_timeout_seconds,
        )
        self.compiler = GrammarCompiler(
            model.tokenizer.vocab,
            limits=self.limits,
            cache_dir=settings.data_dir / "index-cache",
            observer=metrics,
        )
        self.decoder = Decoder(model.lm, model.tokenizer.vocab)
        self.drift = PromptDriftMonitor(model.prompt_lengths)
        self.clock: Callable[[], float] = time.monotonic
        self._rejected: OrderedDict[str, tuple[float, ServiceError]] = OrderedDict()
        self._rejected_lock = threading.Lock()

    # -- schemas ---------------------------------------------------------------------------------

    def compile_schema(self, schema: dict[str, Any], *, client_supplied: bool) -> CompiledGrammar:
        """Compile a schema, or raise a typed :class:`ServiceError` (``400``/``413``).

        Client-supplied schemas (inline, or being registered) are fully vetted: JSON document
        check, size cap, parse, lower, compile, and a bounded compile of their validation
        patterns; rejections are cached by digest. Builtin and already-registered schemas were
        vetted before and are only compiled (through the grammar cache).
        """
        if not client_supplied:
            try:
                return self.compiler.compile(schema)
            except CompileError as exc:
                raise _rejection(exc) from exc
        digest = schema_digest(schema)
        remembered = self._remembered(digest)
        if remembered is not None:
            self.metrics.on_rejected(remembered.reason)
            raise remembered
        try:
            return self._vet(schema)
        except ServiceError as exc:
            self._remember(digest, exc)
            raise
        except CompileError as exc:
            error = _rejection(exc)
            self._remember(digest, error)
            raise error from exc
        except Exception as exc:  # noqa: BLE001 - any other failure is a rejection, not a 500
            _log.error("schema_vetting_failed", error_type=type(exc).__name__)
            self.metrics.on_rejected("internal")
            error = ServiceError(
                400, "schema_rejected", "internal: the schema could not be processed",
                reason="internal",
            )  # fmt: skip
            self._remember(digest, error)
            raise error from None

    def _vet(self, schema: dict[str, Any]) -> CompiledGrammar:
        try:
            check_json_document(schema, self.limits)
        except CompileError as exc:
            self.metrics.on_rejected(exc.reason)
            raise
        size = len(canonical_json(schema).encode("utf-8"))
        if size > self.settings.max_inline_schema_bytes:
            self.metrics.on_rejected("schema_too_large")
            raise ServiceError(413, "schema_too_large", "The schema exceeds the size limit.")
        compiled = self.compiler.compile(schema)
        # Output validation compiles each pattern again (raw UTF-8 instead of JSON-escaped);
        # doing it now, under one shared budget, bounds that work and turns a failure into a 400.
        budget = CompileBudget(self.limits)
        try:
            for pattern in _patterns(compiled.ir):
                pattern_dfa(pattern, self.limits, budget)
        except CompileError as exc:
            self.metrics.on_rejected(exc.reason)
            raise
        return compiled

    def _remembered(self, digest: str) -> ServiceError | None:
        now = self.clock()
        with self._rejected_lock:
            entry = self._rejected.get(digest)
            if entry is None:
                return None
            expires, error = entry
            if expires <= now:
                del self._rejected[digest]
                return None
            self._rejected.move_to_end(digest)
            return error

    def _remember(self, digest: str, error: ServiceError) -> None:
        with self._rejected_lock:
            self._rejected[digest] = (self.clock() + _REJECTION_TTL_SECONDS, error)
            self._rejected.move_to_end(digest)
            while len(self._rejected) > _REJECTION_CACHE_SIZE:
                self._rejected.popitem(last=False)

    def resolve(
        self, schema_id: str | None, inline: dict[str, Any] | None
    ) -> tuple[dict[str, Any] | None, str]:
        if inline is not None:
            return inline, "inline:" + schema_digest(inline)[:12]
        if schema_id is None:
            return None, "none"
        try:
            record = self.schemas.get(schema_id)
        except SchemaNotFoundError as exc:
            raise ServiceError(
                404, "schema_not_found", "No schema is registered under that id."
            ) from exc
        except SchemaIntegrityError as exc:
            _log.error("schema_integrity_failure", schema_id=schema_id)
            raise ServiceError(
                500, "schema_integrity_error", "The stored schema failed verification."
            ) from exc
        return record.schema, schema_id

    # -- generation ------------------------------------------------------------------------------

    def prepare(self, params: GenerationParams) -> Prepared:
        cfg = self.settings
        if len(params.prompt) > cfg.max_prompt_chars:
            raise ServiceError(400, "prompt_too_long", "The prompt exceeds the character limit.")
        if params.max_tokens > cfg.max_tokens_cap:
            raise ServiceError(400, "max_tokens_too_large", "max_tokens exceeds the server cap.")
        prompt_ids = self.model.tokenizer.encode(params.prompt)
        if len(prompt_ids) > cfg.max_prompt_tokens:
            raise ServiceError(400, "prompt_too_long", "The prompt exceeds the token limit.")
        schema, label = self.resolve(params.schema_id, params.inline_schema)
        inline = params.inline_schema is not None
        compiled = None
        if schema is not None and (params.constrained or inline):
            # An inline schema is vetted even for unconstrained decoding: the output is still
            # validated against it, so it must pass the same parser and limits.
            compiled = self.compile_schema(schema, client_supplied=inline)
        grammar = None
        if params.constrained:
            if compiled is None:
                raise ServiceError(400, "schema_required", "Constrained decoding needs a schema.")
            grammar = compiled
            if grammar.min_tokens > params.max_tokens:
                raise ServiceError(
                    400,
                    "max_tokens_too_small",
                    f"This schema needs at least {grammar.min_tokens} tokens.",
                )
        psi = self.drift.observe(len(prompt_ids))
        if psi is not None:
            self.metrics.prompt_length_psi.set(psi)
        return Prepared(params, prompt_ids, schema, label, grammar)

    def stream(self, prepared: Prepared) -> Iterator[StepEvent]:
        params = prepared.params
        strategy: Literal["greedy", "top_p"] = "top_p" if params.strategy == "top_p" else "greedy"
        return self.decoder.stream(
            prepared.prompt_ids,
            grammar=prepared.grammar.index if prepared.grammar is not None else None,
            max_tokens=params.max_tokens,
            strategy=strategy,
            temperature=params.temperature,
            top_p=params.top_p,
            seed=params.seed,
        )

    def generate(self, prepared: Prepared) -> dict[str, Any]:
        params = prepared.params
        generation = self.decoder.generate(
            prepared.prompt_ids,
            grammar=prepared.grammar.index if prepared.grammar is not None else None,
            max_tokens=params.max_tokens,
            strategy=params.strategy,
            temperature=params.temperature,
            top_p=params.top_p,
            seed=params.seed,
            beam_width=params.beam_width,
        )
        return self.finish(
            prepared,
            text=generation.text,
            tokens=len(generation.token_ids),
            stop_reason=generation.stop_reason,
            seconds=generation.seconds,
            mask_seconds=generation.mask_seconds,
        )

    def finish(
        self,
        prepared: Prepared,
        *,
        text: str,
        tokens: int,
        stop_reason: str,
        seconds: float,
        mask_seconds: list[float],
    ) -> dict[str, Any]:
        """Validate the output, record metrics and a privacy-preserving log line."""
        valid: bool | None = None
        error: str | None = None
        if prepared.schema is not None:
            outcome = validate_output(prepared.schema, text, limits=self.limits)
            valid, error = outcome.valid, outcome.error
            self.metrics.observe_output(constrained=prepared.grammar is not None, valid=valid)
        tokens_per_second = tokens / seconds if seconds > 0 else 0.0
        if tokens:
            self.metrics.tokens_per_second.observe(tokens_per_second)
        for value in mask_seconds:
            self.metrics.mask_latency_seconds.observe(value)
        mask_p95 = float(np.percentile(mask_seconds, 95)) if mask_seconds else 0.0
        _log.info(
            "generation_finished",
            schema=prepared.schema_label,
            constrained=prepared.grammar is not None,
            strategy=prepared.params.strategy,
            prompt_chars=len(prepared.params.prompt),
            prompt_sha256=_digest(prepared.params.prompt),
            output_chars=len(text),
            output_sha256=_digest(text),
            tokens=tokens,
            valid=valid,
            stop_reason=stop_reason,
            seconds=round(seconds, 4),
        )
        return {
            "text": text,
            "tokens": tokens,
            "stop_reason": stop_reason,
            "valid": valid,
            "validation_error": error,
            "schema": prepared.schema_label,
            "constrained": prepared.grammar is not None,
            "tokens_per_second": round(tokens_per_second, 1),
            "mask_p95_ms": round(mask_p95 * 1000, 4),
        }

    def timed_stream(self, prepared: Prepared) -> Iterator[tuple[StepEvent, float]]:
        started = time.perf_counter()
        for event in self.stream(prepared):
            yield event, time.perf_counter() - started
