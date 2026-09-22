"""Authentication: API keys (operators, ``/metrics``) and scoped JWTs (the generation API).

API keys are high-entropy random tokens, so a plain SHA-256 digest is an adequate at-rest form (a
slow password hash buys nothing against a 256-bit secret). The presented key is hashed first and
the digest is compared against *every* configured digest with ``hmac.compare_digest`` and no
early exit, so response timing reveals neither key prefixes nor which configured key matched.

JWTs are verified with PyJWT pinned to HS256 (``alg: none`` and algorithm confusion are refused),
with required ``exp``, ``iat``, ``sub``, ``iss`` and ``aud`` claims, a bounded lifetime and a
space-separated ``scope`` claim. Scopes gate the routes: ``generate`` for decoding and reading
schemas, ``schemas:admin`` for registering schemas. With no secret configured every JWT is
rejected (fail closed).
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any

import jwt
from fastapi import Depends, Request

from grammar_constrained_decoder.core.errors import AppError

__all__ = [
    "API_KEY_HEADER",
    "SCOPE_ADMIN",
    "SCOPE_GENERATE",
    "ApiKeyVerifier",
    "AuthenticatedPrincipal",
    "JwtVerifier",
    "Principal",
    "TokenPrincipal",
    "extract_api_key",
    "extract_bearer",
    "generate_api_key",
    "hash_api_key",
    "issue_token",
    "require_api_key",
    "require_scope",
]

SCOPE_GENERATE = "generate"
SCOPE_ADMIN = "schemas:admin"
_ALGORITHM = "HS256"
_SUBJECT = re.compile(r"^[A-Za-z0-9._:@-]{1,128}$")
_SCOPE = re.compile(r"^[a-z:]{1,32}$")
_MAX_TOKEN_LENGTH = 4096

API_KEY_HEADER = "X-API-Key"
_MAX_KEY_LENGTH = 512


def hash_api_key(raw_key: str) -> str:
    """Lowercase hex SHA-256 of an API key, as stored in ``APP_API_KEY_SHA256``."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Return a new (raw_key, sha256_hex) pair. Show the raw key once; store only the digest."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_api_key(raw)


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated caller, identified by a non-secret prefix of its key digest."""

    key_id: str


class ApiKeyVerifier:
    def __init__(self, sha256_hex_digests: Iterable[str]) -> None:
        self._digests = tuple(bytes.fromhex(item) for item in sha256_hex_digests)

    @property
    def configured(self) -> bool:
        return bool(self._digests)

    def verify(self, presented: str | None) -> Principal | None:
        """Return the principal for a valid key, else None. Fails closed with no keys set."""
        if not presented or len(presented) > _MAX_KEY_LENGTH:
            return None
        digest = hashlib.sha256(presented.encode("utf-8")).digest()
        matched: bytes | None = None
        for candidate in self._digests:
            if hmac.compare_digest(digest, candidate):
                matched = candidate
        return Principal(key_id=matched.hex()[:12]) if matched is not None else None


def extract_api_key(headers: Mapping[str, str]) -> str | None:
    """Read the key from ``X-API-Key`` or ``Authorization: Bearer <key>``.

    Starlette's ``Headers`` is case-insensitive; plain mappings must use lowercase names.
    """
    value = headers.get("x-api-key")
    if value:
        return value
    auth = headers.get("authorization")
    if auth and auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    return None


def require_api_key(request: Request) -> Principal:
    """FastAPI dependency: authenticate the caller or raise a uniform 401."""
    verifier: ApiKeyVerifier = request.app.state.api_key_verifier
    principal = verifier.verify(extract_api_key(request.headers))
    if principal is None:
        request.app.state.metrics.auth_failures_total.inc()
        raise AppError(
            401,
            "unauthorized",
            "A valid API key is required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return principal


AuthenticatedPrincipal = Annotated[Principal, Depends(require_api_key)]


@dataclass(frozen=True, slots=True)
class TokenPrincipal:
    """A caller authenticated by JWT. ``client_key`` is a non-reversible id for rate limiting."""

    subject: str
    scopes: frozenset[str]

    @property
    def client_key(self) -> str:
        return "sub:" + hashlib.sha256(self.subject.encode("utf-8")).hexdigest()[:24]


def extract_bearer(headers: Mapping[str, str]) -> str | None:
    auth = headers.get("authorization")
    if auth and auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    return None


def issue_token(
    secret: str,
    *,
    subject: str,
    scopes: Iterable[str],
    issuer: str,
    audience: str,
    ttl_seconds: int = 3600,
    now: Callable[[], float] = time.time,
) -> str:
    """Mint an HS256 token (used by the CLI, the demo and tests)."""
    scope_list = sorted(set(scopes))
    if not _SUBJECT.fullmatch(subject) or not all(_SCOPE.fullmatch(s) for s in scope_list):
        raise ValueError("invalid subject or scope")
    issued = int(now())
    claims: dict[str, Any] = {
        "sub": subject,
        "scope": " ".join(scope_list),
        "iss": issuer,
        "aud": audience,
        "iat": issued,
        "nbf": issued,
        "exp": issued + ttl_seconds,
    }
    return jwt.encode(claims, secret, algorithm=_ALGORITHM)


class JwtVerifier:
    """HS256 verification with required claims and a maximum lifetime."""

    def __init__(
        self,
        secret: str | None,
        *,
        issuer: str,
        audience: str,
        leeway_seconds: int = 30,
        max_ttl_seconds: int = 86_400,
    ) -> None:
        self._secret = secret
        self.issuer = issuer
        self.audience = audience
        self.leeway = leeway_seconds
        self.max_ttl = max_ttl_seconds

    @property
    def configured(self) -> bool:
        return bool(self._secret)

    def verify(self, token: str | None) -> TokenPrincipal | None:
        if not self._secret or not token or len(token) > _MAX_TOKEN_LENGTH:
            return None
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=[_ALGORITHM],
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.leeway,
                options={"require": ["exp", "iat", "sub", "iss", "aud"]},
            )
        except jwt.PyJWTError:
            return None
        subject, scope = claims.get("sub"), claims.get("scope", "")
        issued, expires = claims.get("iat"), claims.get("exp")
        if (
            not isinstance(subject, str)
            or not _SUBJECT.fullmatch(subject)
            or not isinstance(scope, str)
            or not isinstance(issued, int)
            or not isinstance(expires, int)
            or expires - issued > self.max_ttl
        ):
            return None
        scopes = frozenset(part for part in scope.split() if _SCOPE.fullmatch(part))
        return TokenPrincipal(subject=subject, scopes=scopes)


def require_scope(scope: str) -> Callable[[Request], TokenPrincipal]:
    """FastAPI dependency factory: a valid JWT that carries ``scope``.

    Missing or invalid tokens get 401; valid tokens without the scope get 403.
    """

    def dependency(request: Request) -> TokenPrincipal:
        verifier: JwtVerifier = request.app.state.jwt_verifier
        principal = verifier.verify(extract_bearer(request.headers))
        if principal is None:
            request.app.state.metrics.auth_failures_total.inc()
            raise AppError(
                401,
                "unauthorized",
                "A valid bearer token is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if scope not in principal.scopes:
            raise AppError(403, "forbidden", "The token does not grant this operation.")
        return principal

    return dependency
