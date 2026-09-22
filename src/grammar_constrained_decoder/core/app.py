"""FastAPI application factory with the baseline hardening applied."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import CollectorRegistry
from starlette.datastructures import Headers

from grammar_constrained_decoder import __version__
from grammar_constrained_decoder.core.config import Settings, get_settings
from grammar_constrained_decoder.core.errors import install_error_handlers
from grammar_constrained_decoder.core.logging import get_logger
from grammar_constrained_decoder.core.metrics import create_http_metrics, render_metrics
from grammar_constrained_decoder.core.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
)
from grammar_constrained_decoder.core.ratelimit import RateLimiter
from grammar_constrained_decoder.core.security import (
    ApiKeyVerifier,
    JwtVerifier,
    extract_api_key,
    extract_bearer,
    require_api_key,
)
from grammar_constrained_decoder.core.tracing import build_tracer_provider

__all__ = ["ReadinessCheck", "create_app"]

ReadinessCheck = Callable[[], bool | Awaitable[bool]]
_log = get_logger("grammar_constrained_decoder.app")


def _identity(api_keys: ApiKeyVerifier, tokens: JwtVerifier) -> Callable[[Headers], str | None]:
    """Rate-limit key of a caller with valid credentials: API key id first, then JWT subject."""

    def identify(headers: Headers) -> str | None:
        key = api_keys.verify(extract_api_key(headers))
        if key is not None:
            return f"key:{key.key_id}"
        principal = tokens.verify(extract_bearer(headers))
        return principal.client_key if principal is not None else None

    return identify


async def _run_check(name: str, check: ReadinessCheck) -> bool:
    try:
        result = check()
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
    except Exception as exc:  # noqa: BLE001 - a failing probe must report not-ready, not crash
        _log.warning("readiness_check_failed", check=name, error_type=type(exc).__name__)
        return False


def create_app(
    settings: Settings | None = None,
    *,
    readiness_checks: Mapping[str, ReadinessCheck] | None = None,
    registry: CollectorRegistry | None = None,
    tracer_provider: TracerProvider | None = None,
) -> FastAPI:
    """Build the app. All state hangs off ``app.state``; nothing global is mutated."""
    cfg = settings if settings is not None else get_settings()
    checks: dict[str, ReadinessCheck] = dict(readiness_checks or {})
    provider = tracer_provider or build_tracer_provider(cfg.service_name, cfg.tracing_exporter)
    metrics = create_http_metrics(registry)
    verifier = ApiKeyVerifier(cfg.api_key_sha256)
    jwt_verifier = JwtVerifier(
        cfg.jwt_secret.get_secret_value() if cfg.jwt_secret is not None else None,
        issuer=cfg.jwt_issuer,
        audience=cfg.jwt_audience,
        leeway_seconds=cfg.jwt_leeway_seconds,
        max_ttl_seconds=cfg.jwt_max_ttl_seconds,
    )
    limiter = RateLimiter(
        cfg.rate_limit_per_second, cfg.rate_limit_burst, max_keys=cfg.rate_limit_max_clients
    )

    app = FastAPI(
        title=cfg.service_name,
        version=__version__,
        docs_url="/docs" if cfg.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if cfg.enable_docs else None,
    )
    app.state.settings = cfg
    app.state.metrics = metrics
    app.state.api_key_verifier = verifier
    app.state.jwt_verifier = jwt_verifier
    app.state.rate_limiter = limiter
    app.state.tracer_provider = provider
    install_error_handlers(app)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        """Liveness: the process is serving requests. Checks no dependencies."""
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz() -> JSONResponse:
        """Readiness: every registered dependency check passes. Reports names, not errors."""
        results = {name: await _run_check(name, check) for name, check in checks.items()}
        ready = all(results.values())
        body: dict[str, Any] = {
            "status": "ready" if ready else "not_ready",
            "checks": {name: "ok" if ok else "fail" for name, ok in results.items()},
        }
        return JSONResponse(body, status_code=200 if ready else 503)

    metrics_deps = [Depends(require_api_key)] if cfg.metrics_require_auth else []

    @app.get("/metrics", include_in_schema=False, dependencies=metrics_deps)
    async def metrics_endpoint(request: Request) -> Response:
        payload, content_type = render_metrics(request.app.state.metrics.registry)
        return Response(content=payload, media_type=content_type)

    # add_middleware prepends, so the last one added is the outermost.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=cfg.max_body_bytes, metrics=metrics)
    app.add_middleware(
        RateLimitMiddleware,
        limiter=limiter,
        identify=_identity(verifier, jwt_verifier),
        metrics=metrics,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_allow_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization", "X-Request-ID"],
        max_age=600,
    )
    app.add_middleware(
        RequestContextMiddleware,
        tracer=provider.get_tracer("grammar_constrained_decoder.http"),
        metrics=metrics,
        docs_enabled=cfg.enable_docs,
    )
    return app
