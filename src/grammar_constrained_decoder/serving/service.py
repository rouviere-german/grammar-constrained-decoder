"""Generation service: request limits, schema resolution and compilation, decoding, validation.

Everything here is synchronous and CPU-bound; the API runs it in worker threads. The service never
logs prompts or outputs: log events carry lengths, truncated SHA-256 digests, the schema label and
timings only.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from grammar_constrained_decoder.automata.limits import CompileError, CompileLimits
from grammar_constrained_decoder.core.canonical import canonical_json, sha256_hex
from grammar_constrained_decoder.core.config import Settings
from grammar_constrained_decoder.core.logging import get_logger
from grammar_constrained_decoder.decoding.compiler import CompiledGrammar, GrammarCompiler
from grammar_constrained_decoder.decoding.decoder import Decoder, StepEvent
from grammar_constrained_decoder.schema.validate import validate_output
from grammar_constrained_decoder.serving.metrics import DecoderMetrics, PromptDriftMonitor
from grammar_constrained_decoder.serving.schema_registry import (
    SchemaIntegrityError,
    SchemaNotFoundError,
    SchemaRegistry,
)
from grammar_constrained_decoder.training.pipeline import LoadedModel

__all__ = ["GenerationService", "Prepared", "ServiceError"]

_log = get_logger("grammar_constrained_decoder.service")


class ServiceError(Exception):
    """A client-visible failure with an HTTP status and a stable code. Messages are fixed text."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


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

    # -- schemas ---------------------------------------------------------------------------------

    def compile_schema(self, schema: dict[str, Any], *, client_supplied: bool) -> CompiledGrammar:
        """Compile a schema. Client-supplied schemas (inline or being registered) are size-capped;
        builtin and already-registered schemas were accepted before."""
        if client_supplied:
            text = canonical_json(schema)
            if len(text.encode("utf-8")) > self.settings.max_inline_schema_bytes:
                self.metrics.on_rejected("schema_too_large")
                raise ServiceError(413, "schema_too_large", "The schema exceeds the size limit.")
        try:
            return self.compiler.compile(schema)
        except CompileError as exc:
            raise ServiceError(400, "schema_rejected", f"{exc.reason}: {str(exc)[:300]}") from exc

    def resolve(
        self, schema_id: str | None, inline: dict[str, Any] | None
    ) -> tuple[dict[str, Any] | None, str]:
        if inline is not None:
            return inline, "inline:" + sha256_hex(canonical_json(inline))[:12]
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
        grammar = None
        if params.constrained:
            if schema is None:
                raise ServiceError(400, "schema_required", "Constrained decoding needs a schema.")
            grammar = self.compile_schema(schema, client_supplied=params.inline_schema is not None)
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
            outcome = validate_output(prepared.schema, text)
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
