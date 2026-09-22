"""JWT verification, concurrency caps and decoder metrics."""

from __future__ import annotations

import time

import jwt
import numpy as np
import pytest
from prometheus_client import CollectorRegistry, generate_latest

from grammar_constrained_decoder.core.security import (
    SCOPE_ADMIN,
    SCOPE_GENERATE,
    JwtVerifier,
    extract_bearer,
    issue_token,
)
from grammar_constrained_decoder.serving.concurrency import ConcurrencyLimiter
from grammar_constrained_decoder.serving.metrics import PromptDriftMonitor, create_decoder_metrics

pytestmark = pytest.mark.unit

SECRET = "s" * 48
ISSUER, AUDIENCE = "gcd-test", "gcd-api"


def verifier(**overrides: int) -> JwtVerifier:
    return JwtVerifier(SECRET, issuer=ISSUER, audience=AUDIENCE, **overrides)


def token(**overrides: object) -> str:
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": "alice",
        "scope": f"{SCOPE_GENERATE} {SCOPE_ADMIN}",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + 600,
    }
    claims.update(overrides)
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, SECRET, algorithm="HS256")


def test_issued_tokens_verify_with_their_scopes() -> None:
    raw = issue_token(
        SECRET, subject="bob", scopes=[SCOPE_GENERATE], issuer=ISSUER, audience=AUDIENCE
    )
    principal = verifier().verify(raw)
    assert principal is not None
    assert principal.subject == "bob"
    assert principal.scopes == frozenset({SCOPE_GENERATE})
    assert principal.client_key.startswith("sub:")
    assert "bob" not in principal.client_key
    with pytest.raises(ValueError, match="invalid subject"):
        issue_token(SECRET, subject="bad subject", scopes=[], issuer=ISSUER, audience=AUDIENCE)


@pytest.mark.parametrize(
    "overrides",
    [
        {"exp": int(time.time()) - 3600},
        {"aud": "someone-else"},
        {"iss": "someone-else"},
        {"sub": None},
        {"exp": None},
        {"iat": None},
        {"sub": "has space"},
        {"exp": int(time.time()) + 10 * 86_400},  # lifetime above the maximum
    ],
)
def test_invalid_claims_are_rejected(overrides: dict[str, object]) -> None:
    assert verifier().verify(token(**overrides)) is None


def test_algorithm_confusion_and_unsigned_tokens_are_rejected() -> None:
    now = int(time.time())
    claims = {"sub": "a", "iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + 60}
    unsigned = jwt.encode(claims, key="", algorithm="none")
    assert verifier().verify(unsigned) is None
    other_alg = jwt.encode(claims, SECRET, algorithm="HS512")
    assert verifier().verify(other_alg) is None
    wrong_key = jwt.encode(claims, "x" * 48, algorithm="HS256")
    assert verifier().verify(wrong_key) is None
    assert verifier().verify("not.a.jwt") is None
    assert verifier().verify("x" * 5000) is None


def test_no_secret_fails_closed() -> None:
    closed = JwtVerifier(None, issuer=ISSUER, audience=AUDIENCE)
    assert not closed.configured
    assert closed.verify(token()) is None


def test_unknown_scope_strings_are_dropped() -> None:
    principal = verifier().verify(token(scope="generate DROP-TABLE schemas:admin"))
    assert principal is not None
    assert principal.scopes == frozenset({SCOPE_GENERATE, SCOPE_ADMIN})


def test_bearer_extraction() -> None:
    assert extract_bearer({"authorization": "Bearer abc"}) == "abc"
    assert extract_bearer({"authorization": "Basic abc"}) is None
    assert extract_bearer({}) is None


def test_concurrency_limits_are_global_and_per_client() -> None:
    limiter = ConcurrencyLimiter(global_limit=3, per_client_limit=2)
    a1, a2 = limiter.acquire("a"), limiter.acquire("a")
    assert a1 is not None
    assert a2 is not None
    assert limiter.acquire("a") is None
    b1 = limiter.acquire("b")
    assert b1 is not None
    assert limiter.acquire("c") is None  # global cap
    a1.release()
    a1.release()  # idempotent
    assert limiter.active == 2
    assert limiter.acquire("c") is not None
    with pytest.raises(ValueError, match="positive"):
        ConcurrencyLimiter(0, 1)


def test_decoder_metrics_track_ratios_and_reasons() -> None:
    registry = CollectorRegistry()
    metrics = create_decoder_metrics(registry)
    metrics.on_compiled(0.2, cache_hit=False)
    metrics.on_compiled(0.01, cache_hit=True)
    metrics.on_compiled(0.01, cache_hit=True)
    metrics.on_rejected("timeout")
    metrics.on_rejected("made-up-reason")
    metrics.observe_output(constrained=True, valid=True)
    metrics.observe_output(constrained=False, valid=False)
    metrics.observe_output(constrained=False, valid=True)
    text = generate_latest(registry).decode()
    assert "app_index_cache_hit_ratio 0.6666666666666666" in text
    assert 'app_compile_rejections_total{reason="timeout"} 1.0' in text
    assert 'app_compile_rejections_total{reason="other"} 1.0' in text
    assert 'app_output_validity_ratio{mode="constrained"} 1.0' in text
    assert 'app_output_validity_ratio{mode="unconstrained"} 0.5' in text
    for name in ("app_tokens_per_second", "app_mask_latency_seconds", "app_compile_seconds",
                 "app_prompt_length_psi"):  # fmt: skip
        assert name in text


def test_prompt_drift_monitor_flags_shifted_prompt_lengths() -> None:
    rng = np.random.default_rng(0)
    reference = rng.normal(20, 3, size=2_000).round()
    monitor = PromptDriftMonitor(reference, window=400, min_samples=50, every=25)
    values = [monitor.observe(int(v)) for v in rng.normal(20, 3, size=400).round()]
    stable = [v for v in values if v is not None]
    assert stable[-1] < 0.1
    assert monitor.last_psi == stable[-1]
    shifted = [monitor.observe(int(v)) for v in rng.normal(60, 3, size=400).round()]
    assert max(v for v in shifted if v is not None) > 0.2
    with pytest.raises(ValueError, match="empty"):
        PromptDriftMonitor([])
