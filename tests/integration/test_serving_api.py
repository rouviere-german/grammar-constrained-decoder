"""HTTP API: JWT scopes, limits, SSE streaming, audited schema registry, logging hygiene."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conftest import Trained
from grammar_constrained_decoder.core.audit import verify_file
from grammar_constrained_decoder.core.config import Settings
from grammar_constrained_decoder.core.logging import configure_logging
from grammar_constrained_decoder.core.security import (
    SCOPE_ADMIN,
    SCOPE_GENERATE,
    hash_api_key,
    issue_token,
)
from grammar_constrained_decoder.serving.api import build_app

pytestmark = pytest.mark.integration

SECRET = "integration-secret-" + "k" * 32
API_KEY = "operator-key-" + "x" * 30
WEATHER = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "minLength": 1, "maxLength": 16},
        "rain": {"type": "boolean"},
    },
    "required": ["city", "rain"],
    "additionalProperties": False,
}


def settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "jwt_secret": SECRET,
        "data_dir": tmp_path / "data",
        "model_dir": tmp_path / "no-model",
        "environment": "test",
        "api_key_sha256": [hash_api_key(API_KEY)],
        "rate_limit_burst": 1000,
        "rate_limit_per_second": 1000.0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def bearer(subject: str = "alice", scopes: tuple[str, ...] = (SCOPE_GENERATE,)) -> dict[str, str]:
    raw = issue_token(
        SECRET,
        subject=subject,
        scopes=scopes,
        issuer="grammar-constrained-decoder",
        audience="gcd-api",
    )
    return {"Authorization": f"Bearer {raw}"}


@pytest.fixture
def app(trained: Trained, tmp_path: Path) -> FastAPI:
    return build_app(settings(tmp_path), model=trained.model)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_health_and_readiness(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {"model": "ok", "schema_registry": "ok"}


def test_missing_or_tampered_model_leaves_the_service_unready(
    trained: Trained, tmp_path: Path
) -> None:
    app = build_app(settings(tmp_path))  # model_dir does not exist
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/readyz").status_code == 503
        response = client.post("/v1/generate", json={"schema_id": "invoice"}, headers=bearer())
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"


def test_generation_requires_a_valid_token_with_the_generate_scope(client: TestClient) -> None:
    body = {"prompt": "Q: x\ninvoice:\n", "schema_id": "invoice", "max_tokens": 200}
    assert client.post("/v1/generate", json=body).status_code == 401
    wrong = client.post("/v1/generate", json=body, headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    assert wrong.headers["www-authenticate"] == "Bearer"
    admin_only = client.post("/v1/generate", json=body, headers=bearer(scopes=(SCOPE_ADMIN,)))
    assert admin_only.status_code == 403
    ok = client.post("/v1/generate", json=body, headers=bearer())
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["valid"] is True
    assert payload["constrained"] is True
    assert payload["stop_reason"] == "eos"
    assert json.loads(payload["text"])["invoice_no"].startswith("INV-")


@pytest.mark.parametrize("strategy", ["greedy", "top_p", "beam"])
def test_every_strategy_returns_schema_valid_output(client: TestClient, strategy: str) -> None:
    body = {"prompt": "Q: weather\nweather:\n", "schema": WEATHER, "strategy": strategy,
            "max_tokens": 60, "seed": 3, "beam_width": 3}  # fmt: skip
    response = client.post("/v1/generate", json=body, headers=bearer())
    assert response.status_code == 200, response.text
    assert response.json()["valid"] is True


def test_unconstrained_generation_is_reported_honestly(client: TestClient) -> None:
    body = {"prompt": "Q: weather\nweather:\n", "schema": WEATHER, "constrained": False,
            "max_tokens": 40, "strategy": "top_p", "seed": 1}  # fmt: skip
    response = client.post("/v1/generate", json=body, headers=bearer())
    assert response.status_code == 200
    assert response.json()["constrained"] is False
    assert response.json()["valid"] in (True, False)
    no_schema = client.post(
        "/v1/generate",
        json={"prompt": "hi", "constrained": False, "max_tokens": 5},
        headers=bearer(),
    )
    assert no_schema.json()["valid"] is None
    needs_schema = client.post("/v1/generate", json={"prompt": "hi"}, headers=bearer())
    assert needs_schema.status_code == 400
    assert needs_schema.json()["error"]["code"] == "schema_required"


def test_sse_streams_tokens_then_a_done_event(client: TestClient) -> None:
    body = {"prompt": "Q: weather\nweather:\n", "schema": WEATHER, "stream": True, "max_tokens": 60}
    events: list[tuple[str, str]] = []
    with client.stream("POST", "/v1/generate", json=body, headers=bearer()) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        name = ""
        for line in response.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((name, line.split(":", 1)[1].strip()))
    tokens = [json.loads(data)["text"] for kind, data in events if kind == "token"]
    done = [json.loads(data) for kind, data in events if kind == "done"]
    assert len(done) == 1
    assert done[0]["valid"] is True
    assert done[0]["tokens"] == len(tokens)
    assert json.loads("".join(tokens))["rain"] in (True, False)
    beam_stream = client.post("/v1/generate", json={**body, "strategy": "beam"}, headers=bearer())
    assert beam_stream.status_code == 422


def test_request_caps(trained: Trained, tmp_path: Path) -> None:
    app = build_app(
        settings(tmp_path, max_tokens_cap=50, max_prompt_chars=100, max_inline_schema_bytes=200),
        model=trained.model,
    )
    with TestClient(app) as client:
        too_many = client.post(
            "/v1/generate", json={"schema": WEATHER, "max_tokens": 51}, headers=bearer()
        )
        assert too_many.json()["error"]["code"] == "max_tokens_too_large"
        long_prompt = client.post(
            "/v1/generate", json={"schema_id": "geo_point", "prompt": "x" * 101}, headers=bearer()
        )
        assert long_prompt.json()["error"]["code"] == "prompt_too_long"
        big_schema = {"enum": ["v" * 50 for _ in range(1)] + [f"x{i}" for i in range(40)]}
        too_big = client.post(
            "/v1/generate", json={"schema": big_schema, "max_tokens": 50}, headers=bearer()
        )
        assert too_big.status_code == 413
        tiny_budget = client.post(
            "/v1/generate", json={"schema_id": "invoice", "max_tokens": 5}, headers=bearer()
        )
        assert tiny_budget.json()["error"]["code"] == "max_tokens_too_small"
        unknown = client.post(
            "/v1/generate", json={"schema_id": "nope", "max_tokens": 5}, headers=bearer()
        )
        assert unknown.status_code == 404
        both = client.post(
            "/v1/generate", json={"schema_id": "invoice", "schema": WEATHER}, headers=bearer()
        )
        assert both.status_code == 422


def test_prompt_token_cap(trained: Trained, tmp_path: Path) -> None:
    app = build_app(settings(tmp_path, max_prompt_tokens=3), model=trained.model)
    with TestClient(app) as client:
        response = client.post(
            "/v1/generate",
            json={"schema_id": "geo_point", "prompt": "one two three four five"},
            headers=bearer(),
        )
        assert response.json()["error"]["code"] == "prompt_too_long"


def test_concurrency_cap_returns_429(app: FastAPI, client: TestClient) -> None:
    limiter = app.state.concurrency
    held = [limiter.acquire(f"sub:someone-{i}") for i in range(limiter.global_limit)]
    assert all(lease is not None for lease in held)
    response = client.post(
        "/v1/generate", json={"schema": WEATHER, "max_tokens": 40}, headers=bearer()
    )
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "too_many_generations"
    for lease in held:
        assert lease is not None
        lease.release()
    assert (
        client.post(
            "/v1/generate", json={"schema": WEATHER, "max_tokens": 40}, headers=bearer()
        ).status_code
        == 200
    )
    assert limiter.active == 0


def test_rate_limit_is_per_authenticated_subject(trained: Trained, tmp_path: Path) -> None:
    app = build_app(
        settings(tmp_path, rate_limit_burst=2, rate_limit_per_second=0.001), model=trained.model
    )
    with TestClient(app) as client:
        alice, bob = bearer("alice"), bearer("bob")
        codes = [client.get("/v1/schemas", headers=alice).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        assert client.get("/v1/schemas", headers=bob).status_code == 200


def test_schema_registration_needs_admin_and_is_audited(
    app: FastAPI, client: TestClient, tmp_path: Path
) -> None:
    body = {"schema_id": "weather", "schema": WEATHER}
    assert client.post("/v1/schemas", json=body, headers=bearer()).status_code == 403
    created = client.post("/v1/schemas", json=body, headers=bearer("root", (SCOPE_ADMIN,)))
    assert created.status_code == 201
    assert created.json()["dfa_states"] > 0
    again = client.post("/v1/schemas", json=body, headers=bearer("root", (SCOPE_ADMIN,)))
    assert again.status_code == 201  # idempotent for identical content
    changed = client.post(
        "/v1/schemas", json={"schema_id": "weather", "schema": {"type": "null"}},
        headers=bearer("root", (SCOPE_ADMIN,)),
    )  # fmt: skip
    assert changed.status_code == 409
    builtin = client.post(
        "/v1/schemas", json={"schema_id": "invoice", "schema": {"type": "null"}},
        headers=bearer("root", (SCOPE_ADMIN,)),
    )  # fmt: skip
    assert builtin.status_code == 409
    rejected = client.post(
        "/v1/schemas", json={"schema_id": "bad", "schema": {"type": "string", "format": "email"}},
        headers=bearer("root", (SCOPE_ADMIN,)),
    )  # fmt: skip
    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "schema_rejected"
    assert "'format'" in rejected.json()["error"]["message"]

    audit_path = tmp_path / "data" / "schema-registry" / "registry_audit.jsonl"
    result = verify_file(audit_path)
    assert result.ok
    entries = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert [entry["action"] for entry in entries] == ["schema.register", "schema.reject"]
    assert entries[0]["actor"] == "root"
    assert "properties" not in json.dumps(entries)  # hashes and ids only, never schema bodies

    listed = client.get("/v1/schemas", headers=bearer()).json()["schemas"]
    assert {"schema_id": "weather", "sha256": created.json()["sha256"], "builtin": False} in listed
    assert sum(1 for item in listed if item["builtin"]) == 30
    fetched = client.get("/v1/schemas/weather", headers=bearer()).json()
    assert fetched["schema"] == WEATHER
    assert client.get("/v1/schemas/missing", headers=bearer()).status_code == 404
    generated = client.post(
        "/v1/generate", json={"schema_id": "weather", "max_tokens": 60}, headers=bearer()
    )
    assert generated.json()["valid"] is True


def test_a_schema_edited_on_disk_is_refused(
    app: FastAPI, client: TestClient, tmp_path: Path
) -> None:
    admin = bearer("root", (SCOPE_ADMIN,))
    assert (
        client.post(
            "/v1/schemas", json={"schema_id": "weather", "schema": WEATHER}, headers=admin
        ).status_code
        == 201
    )
    stored = tmp_path / "data" / "schema-registry" / "schemas" / "weather.json"
    loosened = dict(WEATHER, required=[])
    stored.write_text(json.dumps(loosened), encoding="utf-8")
    response = client.get("/v1/schemas/weather", headers=bearer())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "schema_integrity_error"
    generate = client.post(
        "/v1/generate", json={"schema_id": "weather", "max_tokens": 40}, headers=bearer()
    )
    assert generate.status_code == 500
    assert client.get("/readyz").status_code == 503
    assert all(
        item["schema_id"] != "weather"
        for item in client.get("/v1/schemas", headers=bearer()).json()["schemas"]
    )


def test_prompts_and_outputs_never_reach_the_logs(
    client: TestClient, restore_logging: None
) -> None:
    stream = io.StringIO()
    configure_logging("INFO", json_output=True, stream=stream)
    marker = "CONFIDENTIAL-PROMPT-7731"
    body = {"prompt": f"Q: {marker}\nweather:\n", "schema": WEATHER, "max_tokens": 60}
    response = client.post("/v1/generate", json=body, headers=bearer())
    assert response.status_code == 200
    output = response.json()["text"]
    logs = stream.getvalue()
    assert marker not in logs
    assert output not in logs
    finished = [json.loads(line) for line in logs.splitlines() if "generation_finished" in line]
    assert len(finished) == 1
    event = finished[0]
    assert event["prompt_chars"] == len(str(body["prompt"]))
    assert re.fullmatch(r"[0-9a-f]{16}", event["prompt_sha256"])
    assert event["schema"].startswith("inline:")


def test_metrics_expose_decoder_series(client: TestClient) -> None:
    client.post("/v1/generate", json={"schema": WEATHER, "max_tokens": 60}, headers=bearer())
    client.post(
        "/v1/generate", json={"schema": {"type": "string", "format": "x"}}, headers=bearer()
    )
    assert client.get("/metrics").status_code == 401
    text = client.get("/metrics", headers={"X-API-Key": API_KEY}).text
    for name in (
        "app_tokens_per_second_count 1.0",
        "app_mask_latency_seconds_count",
        "app_index_cache_hit_ratio",
        "app_compile_seconds_count",
        'app_compile_rejections_total{reason="unsupported"} 1.0',
        'app_output_validity_ratio{mode="constrained"} 1.0',
        "app_prompt_length_psi",
    ):
        assert name in text, name


def test_python_re_never_sees_a_client_pattern(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = {"type": "string", "pattern": "^(a|aa)+X$", "maxLength": 8}
    seen: list[str] = []
    real_compile = vars(re)["_compile"]  # what re.compile, re.search and re.match all call

    def guarded(pattern: Any, flags: Any) -> Any:
        if isinstance(pattern, str) and "(a|aa)" in pattern:
            seen.append(pattern)
        return real_compile(pattern, flags)

    monkeypatch.setattr(re, "_compile", guarded)
    response = client.post(
        "/v1/generate", json={"schema": schema, "max_tokens": 20}, headers=bearer()
    )
    assert response.status_code == 200
    assert response.json()["valid"] is True
    assert seen == []
