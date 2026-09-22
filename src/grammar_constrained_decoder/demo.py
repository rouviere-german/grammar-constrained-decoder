"""Offline end-to-end demo: train, compile, decode with and without constraints, stream via SSE.

``uv run python tasks.py demo`` (or ``python -m grammar_constrained_decoder demo``):

1. trains the corpus, tokenizer and 5-gram model into ``<workdir>/model`` and loads it back
   through the verified registry path;
2. for three held-out schemas, compiles the grammar and decodes greedy and top-p samples with and
   without the mask, plus the "retry up to 3 times" baseline, and reports validity and mask
   overhead;
3. starts the API in-process (FastAPI TestClient, no sockets), mints scoped JWTs, shows the auth
   checks, registers a schema as admin, streams one generation over SSE and verifies the
   registry audit chain;
4. measures distribution distortion on three tiny schemas.

Needs the dev environment (``uv sync`` installs it) for the in-process HTTP client.
"""

from __future__ import annotations

import json
import secrets
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from grammar_constrained_decoder.core.audit import verify_file
from grammar_constrained_decoder.core.config import Settings
from grammar_constrained_decoder.core.security import SCOPE_ADMIN, SCOPE_GENERATE, issue_token
from grammar_constrained_decoder.decoding.compiler import GrammarCompiler
from grammar_constrained_decoder.decoding.decoder import Decoder
from grammar_constrained_decoder.decoding.distortion import distortion_report
from grammar_constrained_decoder.schema.validate import validate_output
from grammar_constrained_decoder.training.corpus import (
    BundledSchema,
    load_bundled_schemas,
    load_distortion_schemas,
    prompt_for,
)
from grammar_constrained_decoder.training.pipeline import LoadedModel, load_model, train_pipeline

__all__ = ["DEMO_SCHEMAS", "run_demo"]

DEMO_SCHEMAS = ("invoice", "category_tree", "feature_flag")
DEMO_TINY = ("color", "digit", "ok_flag")
MAX_TOKENS = 160


def _pick(schemas: list[BundledSchema], names: tuple[str, ...]) -> list[BundledSchema]:
    by_id = {item.id: item for item in schemas}
    return [by_id[name] for name in names]


def _schema_study(
    model: LoadedModel,
    compiler: GrammarCompiler,
    schema: BundledSchema,
    samples: int,
    say: Callable[[str], None],
) -> dict[str, Any]:
    decoder = Decoder(model.lm, model.tokenizer.vocab)
    grammar = compiler.compile(schema.schema)
    prompt = model.tokenizer.encode(prompt_for(schema))

    def valid(text: str) -> bool:
        return validate_output(schema.schema, text).valid

    greedy_c = decoder.generate(prompt, grammar=grammar.index, max_tokens=MAX_TOKENS)
    greedy_u = decoder.generate(prompt, grammar=None, max_tokens=MAX_TOKENS)
    constrained = [
        decoder.generate(
            prompt,
            grammar=grammar.index,
            max_tokens=MAX_TOKENS,
            strategy="top_p",
            top_p=0.95,
            seed=s,
        )
        for s in range(samples)
    ]
    free = [
        decoder.generate(
            prompt, grammar=None, max_tokens=MAX_TOKENS, strategy="top_p", top_p=0.95, seed=s
        )
        for s in range(samples)
    ]
    retried = 0
    for s in range(samples):
        for attempt in range(4):
            out = decoder.generate(
                prompt,
                grammar=None,
                max_tokens=MAX_TOKENS,
                strategy="top_p",
                top_p=0.95,
                seed=10_000 + 4 * s + attempt,
            )
            if valid(out.text):
                retried += 1
                break
    masks = np.array([m for g in constrained for m in g.mask_seconds]) * 1e6
    result = {
        "dfa_states": grammar.stats.dfa_states,
        "index_entries": grammar.stats.index_entries,
        "compile_seconds": round(grammar.stats.seconds, 3),
        "greedy_constrained_valid": valid(greedy_c.text),
        "greedy_unconstrained_valid": valid(greedy_u.text),
        "top_p_constrained_valid_ratio": sum(valid(g.text) for g in constrained) / samples,
        "top_p_unconstrained_valid_ratio": sum(valid(g.text) for g in free) / samples,
        "unconstrained_3_retries_valid_ratio": retried / samples,
        "mask_p50_us": round(float(np.percentile(masks, 50)), 1),
        "mask_p95_us": round(float(np.percentile(masks, 95)), 1),
    }
    say(
        f"\n== {schema.id}: {grammar.stats.dfa_states} DFA states, "
        f"{grammar.stats.index_entries} index entries, compiled in {grammar.stats.seconds:.2f}s"
    )
    say(
        f"  constrained greedy   [{'valid' if result['greedy_constrained_valid'] else 'INVALID'}] "
        f"{greedy_c.text[:110]}"
    )
    say(
        f"  unconstrained greedy "
        f"[{'valid' if result['greedy_unconstrained_valid'] else 'INVALID'}] "
        f"{greedy_u.text[:110]!r}"
    )
    say(
        f"  top-p validity over {samples} samples: constrained "
        f"{result['top_p_constrained_valid_ratio']:.0%}, unconstrained "
        f"{result['top_p_unconstrained_valid_ratio']:.0%}, unconstrained + 3 retries "
        f"{result['unconstrained_3_retries_valid_ratio']:.0%}"
    )
    say(
        f"  mask overhead per token: p50 {result['mask_p50_us']} us, p95 {result['mask_p95_us']} us"
    )
    return result


def _http_demo(model: LoadedModel, workdir: Path, say: Callable[[str], None]) -> dict[str, Any]:
    from fastapi.testclient import TestClient  # noqa: PLC0415 - dev dependency, demo only

    from grammar_constrained_decoder.serving.api import build_app  # noqa: PLC0415

    secret = secrets.token_urlsafe(48)
    settings = Settings(
        _env_file=None,
        jwt_secret=secret,
        data_dir=workdir / "service",
        environment="test",
    )
    app = build_app(settings, model=model)
    user = issue_token(
        secret,
        subject="demo-user",
        scopes=[SCOPE_GENERATE],
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
    )
    admin = issue_token(
        secret,
        subject="demo-admin",
        scopes=[SCOPE_GENERATE, SCOPE_ADMIN],
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
    )
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "minLength": 1, "maxLength": 20},
            "rain": {"type": "boolean"},
            "mm": {"type": "number", "minimum": 0, "maximum": 500},
        },
        "required": ["city", "rain"],
        "additionalProperties": False,
    }
    with TestClient(app) as client:
        ready = client.get("/readyz").status_code
        anonymous = client.post("/v1/generate", json={"prompt": "x", "schema_id": "invoice"})
        forbidden = client.post(
            "/v1/schemas",
            json={"schema_id": "rain_report", "schema": schema},
            headers={"Authorization": f"Bearer {user}"},
        )
        registered = client.post(
            "/v1/schemas",
            json={"schema_id": "rain_report", "schema": schema},
            headers={"Authorization": f"Bearer {admin}"},
        )
        rejected = client.post(
            "/v1/schemas",
            json={"schema_id": "bad", "schema": {"type": "string", "format": "email"}},
            headers={"Authorization": f"Bearer {admin}"},
        )
        events: list[tuple[str, str]] = []
        with client.stream(
            "POST",
            "/v1/generate",
            json={
                "prompt": "Q: rain report\nrain_report:\n",
                "schema_id": "rain_report",
                "stream": True,
                "max_tokens": 80,
            },
            headers={"Authorization": f"Bearer {user}"},
        ) as response:
            event_name = ""
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    events.append((event_name, line.split(":", 1)[1].strip()))
    tokens = [json.loads(data)["text"] for name, data in events if name == "token"]
    done = next(json.loads(data) for name, data in events if name == "done")
    audit = verify_file(workdir / "service" / "schema-registry" / "registry_audit.jsonl")
    say("\n== HTTP API (in-process)")
    say(
        f"  /readyz -> {ready}; no token -> {anonymous.status_code}; "
        f"register with generate scope -> {forbidden.status_code}; "
        f"register as admin -> {registered.status_code}; unsupported keyword -> "
        f"{rejected.status_code} ({rejected.json()['error']['code']})"
    )
    say(f"  SSE stream: {len(tokens)} token events -> {''.join(tokens)[:100]}")
    say(
        f"  done event: valid={done['valid']} stop={done['stop_reason']} "
        f"tokens/s={done['tokens_per_second']}"
    )
    say(f"  registry audit chain: ok={audit.ok}, {audit.entries} entries")
    return {
        "readyz": ready,
        "anonymous_status": anonymous.status_code,
        "forbidden_status": forbidden.status_code,
        "register_status": registered.status_code,
        "reject_status": rejected.status_code,
        "sse_token_events": len(tokens),
        "sse_valid": done["valid"],
        "audit_ok": audit.ok,
        "audit_entries": audit.entries,
    }


def run_demo(workdir: Path, *, samples: int = 20, out: TextIO | None = None) -> dict[str, Any]:
    stream = out if out is not None else sys.stdout

    def say(text: str) -> None:
        stream.write(text + "\n")
        stream.flush()

    started = time.perf_counter()
    trained = train_pipeline(workdir / "model")
    model = load_model(workdir / "model" / "registry")
    say(
        f"trained in {trained.seconds['total']:.1f}s: "
        f"{int(trained.metrics['train_tokens'])} tokens, "
        f"vocab {int(trained.metrics['vocab_size'])}, held-out perplexity "
        f"{trained.metrics['heldout_perplexity']:.1f}; model version {trained.version.digest[:16]}"
    )
    compiler = GrammarCompiler(model.tokenizer.vocab, cache_dir=workdir / "index-cache")
    schemas = _pick(load_bundled_schemas(), DEMO_SCHEMAS)
    studies = {item.id: _schema_study(model, compiler, item, samples, say) for item in schemas}
    http = _http_demo(model, workdir, say)
    say("\n== distortion (exact, tiny schemas)")
    distortion: dict[str, Any] = {}
    for tiny in _pick(load_distortion_schemas(), DEMO_TINY):
        grammar = compiler.compile(tiny.schema)
        prompt = model.tokenizer.encode(prompt_for(tiny))
        report = distortion_report(
            model.lm, grammar.index, model.tokenizer.vocab, prompt, samples=200
        )
        distortion[tiny.id] = report.as_dict()
        say(
            f"  {tiny.id}: {report.n_strings} strings / {report.n_sequences} token sequences, "
            f"valid mass {report.valid_mass:.3f}, KL(string) {report.kl_string:.4f} nats, "
            f"greedy {report.greedy_text!r} is mode: {report.greedy_is_conditional_mode}"
        )
    summary: dict[str, Any] = {
        "model_version": trained.version.digest,
        "train_seconds": round(trained.seconds["total"], 2),
        "heldout_perplexity": trained.metrics["heldout_perplexity"],
        "schemas": studies,
        "http": http,
        "distortion": distortion,
        "seconds": round(time.perf_counter() - started, 1),
    }
    return summary
