"""HTTP API: ``/v1/generate`` (JSON or SSE), schema registry routes, health and readiness.

Built on the hardened core app (request ids, security headers, rate limiting, body limits, error
handling). Route protection:

- ``POST /v1/generate`` and ``GET /v1/schemas[/{id}]`` need a JWT with the ``generate`` scope;
- ``POST /v1/schemas`` needs the ``schemas:admin`` scope and is audited;
- ``/metrics`` needs an operator API key (template behavior); ``/healthz`` and ``/readyz`` are open.

Generations also pass a non-blocking concurrency cap (global and per client). CPU-bound work
(compilation, decoding steps) runs in worker threads so the event loop keeps serving.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterator
from functools import partial
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Request
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import CollectorRegistry
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sse_starlette import EventSourceResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from grammar_constrained_decoder.core.app import create_app
from grammar_constrained_decoder.core.canonical import canonical_json, sha256_hex
from grammar_constrained_decoder.core.config import Settings, get_settings
from grammar_constrained_decoder.core.errors import AppError
from grammar_constrained_decoder.core.logging import get_logger
from grammar_constrained_decoder.core.security import (
    SCOPE_ADMIN,
    SCOPE_GENERATE,
    TokenPrincipal,
    require_scope,
)
from grammar_constrained_decoder.decoding.decoder import StepEvent
from grammar_constrained_decoder.serving.concurrency import ConcurrencyLimiter, Lease
from grammar_constrained_decoder.serving.metrics import create_decoder_metrics
from grammar_constrained_decoder.serving.schema_registry import (
    SchemaConflictError,
    SchemaIntegrityError,
    SchemaNotFoundError,
    SchemaRegistry,
)
from grammar_constrained_decoder.serving.service import (
    GenerationParams,
    GenerationService,
    Prepared,
    ServiceError,
)
from grammar_constrained_decoder.training.corpus import load_bundled_schemas
from grammar_constrained_decoder.training.pipeline import (
    LoadedModel,
    ModelIntegrityError,
    load_model,
)

__all__ = ["GenerateRequest", "RegisterSchemaRequest", "build_app"]

_log = get_logger("grammar_constrained_decoder.api")
_SCHEMA_ID = r"^[a-z0-9][a-z0-9_-]{0,63}$"
GeneratePrincipal = Annotated[TokenPrincipal, Depends(require_scope(SCOPE_GENERATE))]
AdminPrincipal = Annotated[TokenPrincipal, Depends(require_scope(SCOPE_ADMIN))]


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    prompt: str = Field(default="", max_length=100_000)
    schema_id: str | None = Field(default=None, pattern=_SCHEMA_ID)
    json_schema: dict[str, Any] | None = Field(default=None, alias="schema")
    constrained: bool = True
    strategy: Literal["greedy", "top_p", "beam"] = "greedy"
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=1.0, gt=0, le=5)
    top_p: float = Field(default=0.95, gt=0, le=1)
    beam_width: int = Field(default=4, ge=1, le=8)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    stream: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> GenerateRequest:
        if self.schema_id is not None and self.json_schema is not None:
            raise ValueError("give either schema_id or schema, not both")
        if self.stream and self.strategy == "beam":
            raise ValueError("beam search cannot be streamed")
        return self

    def params(self) -> GenerationParams:
        return GenerationParams(
            prompt=self.prompt,
            schema_id=self.schema_id,
            inline_schema=self.json_schema,
            constrained=self.constrained,
            strategy=self.strategy,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            beam_width=self.beam_width,
            seed=self.seed,
        )


class RegisterSchemaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_id: str = Field(pattern=_SCHEMA_ID)
    json_schema: dict[str, Any] = Field(alias="schema")


def _app_error(exc: ServiceError) -> AppError:
    return AppError(exc.status, exc.code, exc.message)


def _service(request: Request) -> GenerationService:
    service: GenerationService | None = request.app.state.service
    if service is None:
        raise AppError(503, "model_unavailable", "The model is not loaded.")
    return service


def _next_event(iterator: Iterator[StepEvent]) -> StepEvent | None:
    return next(iterator, None)


async def _stream_events(
    service: GenerationService, prepared: Prepared, lease: Lease
) -> AsyncIterator[dict[str, str]]:
    """SSE events: one ``token`` per generated token, then ``done`` with the validation result."""
    tokens = 0
    text_parts: list[str] = []
    masks: list[float] = []
    stop_reason = "max_tokens"
    started = time.perf_counter()
    try:
        iterator = service.stream(prepared)
        while True:
            event = await run_in_threadpool(_next_event, iterator)
            if event is None:
                break
            masks.append(event.mask_seconds)
            if event.eos:
                stop_reason = "eos"
                if event.text:
                    text_parts.append(event.text)
                break
            tokens += 1
            text_parts.append(event.text)
            yield {"event": "token", "data": json.dumps({"index": event.index, "text": event.text})}
        summary = service.finish(
            prepared,
            text="".join(text_parts),
            tokens=tokens,
            stop_reason=stop_reason,
            seconds=time.perf_counter() - started,
            mask_seconds=masks,
        )
        summary.pop("text")
        yield {"event": "done", "data": json.dumps(summary)}
    finally:
        lease.release()


def build_app(
    settings: Settings | None = None,
    *,
    model: LoadedModel | None = None,
    load: bool = True,
    registry: CollectorRegistry | None = None,
    tracer_provider: TracerProvider | None = None,
) -> FastAPI:
    """Create the service app. With ``model=None`` and ``load=True`` the model is loaded from
    ``settings.model_dir``; a model that fails verification leaves the app unready (503)."""
    cfg = settings if settings is not None else get_settings()
    if model is None and load:
        try:
            model = load_model(cfg.model_dir)
        except ModelIntegrityError as exc:
            _log.error("model_load_failed", error_type=type(exc).__name__)
            model = None
    schemas = SchemaRegistry(
        cfg.data_dir / "schema-registry",
        builtins={item.id: item.schema for item in load_bundled_schemas()},
    )
    state: dict[str, Any] = {}
    app = create_app(
        cfg,
        readiness_checks={
            "model": lambda: state.get("service") is not None,
            "schema_registry": schemas.verify,
        },
        registry=registry,
        tracer_provider=tracer_provider,
    )
    metrics = create_decoder_metrics(app.state.metrics.registry)
    service = (
        GenerationService(cfg, model, schemas=schemas, metrics=metrics)
        if model is not None
        else None
    )
    state["service"] = service
    app.state.service = service
    app.state.schema_registry = schemas
    app.state.decoder_metrics = metrics
    app.state.concurrency = ConcurrencyLimiter(
        cfg.max_concurrent_generations, cfg.max_concurrent_per_client
    )
    _install_generation_route(app)
    _install_schema_routes(app, schemas)
    return app


def _install_generation_route(app: FastAPI) -> None:
    @app.post("/v1/generate")
    async def generate(
        body: GenerateRequest, principal: GeneratePrincipal, request: Request
    ) -> Any:
        """Generate text for a prompt, constrained to a schema unless ``constrained`` is false."""
        active = _service(request)
        limiter: ConcurrencyLimiter = request.app.state.concurrency
        lease = limiter.acquire(principal.client_key)
        if lease is None:
            raise AppError(429, "too_many_generations", "Too many concurrent generations.")
        try:
            prepared = await run_in_threadpool(active.prepare, body.params())
        except ServiceError as exc:
            lease.release()
            raise _app_error(exc) from None
        except BaseException:
            lease.release()
            raise
        if body.stream:
            # The generator releases the lease when it finishes or is cancelled; the background
            # task covers a response that is never started (release is idempotent).
            return EventSourceResponse(
                _stream_events(active, prepared, lease),
                ping=15,
                background=BackgroundTask(lease.release),
            )
        try:
            return await run_in_threadpool(active.generate, prepared)
        finally:
            lease.release()


def _install_schema_routes(app: FastAPI, schemas: SchemaRegistry) -> None:
    @app.post("/v1/schemas", status_code=201)
    async def register_schema(
        body: RegisterSchemaRequest, principal: AdminPrincipal, request: Request
    ) -> dict[str, Any]:
        """Compile and register a schema (admin scope). Rejections are audited too."""
        active = _service(request)
        digest = sha256_hex(canonical_json(body.json_schema))
        try:
            compiled = await run_in_threadpool(
                partial(active.compile_schema, body.json_schema, client_supplied=True)
            )
        except ServiceError as exc:
            schemas.record_rejection(
                actor=principal.subject, schema_id=body.schema_id, digest=digest, reason=exc.code
            )
            raise _app_error(exc) from None
        stats = {
            "dfa_states": compiled.stats.dfa_states,
            "index_entries": compiled.stats.index_entries,
        }
        try:
            record = schemas.register(
                body.schema_id, body.json_schema, actor=principal.subject, stats=stats
            )
        except SchemaConflictError as exc:
            raise AppError(409, "schema_conflict", "That schema id is taken.") from exc
        return {"schema_id": record.schema_id, "sha256": record.sha256, **stats}

    @app.get("/v1/schemas")
    async def list_schemas(principal: GeneratePrincipal) -> dict[str, Any]:
        del principal
        return {
            "schemas": [
                {"schema_id": item.schema_id, "sha256": item.sha256, "builtin": item.builtin}
                for item in schemas.list()
            ]
        }

    @app.get("/v1/schemas/{schema_id}")
    async def get_schema(schema_id: str, principal: GeneratePrincipal) -> dict[str, Any]:
        del principal
        try:
            record = schemas.get(schema_id)
        except SchemaNotFoundError as exc:
            raise AppError(
                404, "schema_not_found", "No schema is registered under that id."
            ) from exc
        except SchemaIntegrityError as exc:
            raise AppError(
                500, "schema_integrity_error", "The stored schema failed verification."
            ) from exc
        return {
            "schema_id": record.schema_id,
            "sha256": record.sha256,
            "builtin": record.builtin,
            "schema": record.schema,
        }
