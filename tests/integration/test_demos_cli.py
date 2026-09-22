from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace as otel_trace

from conftest import Trained
from grammar_constrained_decoder import cli
from grammar_constrained_decoder.core import tracing
from grammar_constrained_decoder.core.audit import AuditLog
from grammar_constrained_decoder.core.security import hash_api_key
from grammar_constrained_decoder.demo import run_demo

pytestmark = pytest.mark.integration


def test_demo_runs_end_to_end_offline(tmp_path: Path) -> None:
    out = __import__("io").StringIO()
    summary = run_demo(tmp_path, samples=3, out=out)
    for study in summary["schemas"].values():
        assert study["greedy_constrained_valid"] is True
        assert study["top_p_constrained_valid_ratio"] == 1.0
    http = summary["http"]
    assert (http["anonymous_status"], http["forbidden_status"], http["register_status"]) == (
        401,
        403,
        201,
    )
    assert http["reject_status"] == 400
    assert http["sse_valid"] is True
    assert http["sse_token_events"] > 0
    assert http["audit_ok"] is True
    assert set(summary["distortion"]) == {"color", "digit", "ok_flag"}
    assert "SSE stream" in out.getvalue()


@pytest.mark.usefixtures("restore_logging")
def test_cli_demo_prints_a_json_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import grammar_constrained_decoder.demo as demo_module

    monkeypatch.setattr(demo_module, "run_demo", lambda workdir, samples: {"seconds": 1.0})
    assert cli.main(["demo", "--workdir", str(tmp_path), "--samples", "2"]) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["demo"] == "grammar-constrained-decoder"


def test_cli_train_and_compile(
    trained: Trained, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(
        json.dumps({"type": "array", "items": {"type": "boolean"}}), encoding="utf-8"
    )
    assert (
        cli.main(["compile", str(schema_file), "--model-dir", str(trained.root / "registry")]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["dfa_states"] > 0
    schema_file.write_text(json.dumps({"type": "string", "format": "uri"}), encoding="utf-8")
    assert (
        cli.main(["compile", str(schema_file), "--model-dir", str(trained.root / "registry")]) == 1
    )
    assert json.loads(capsys.readouterr().out)["reason"] == "unsupported"


def test_cli_train_writes_a_registry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from grammar_constrained_decoder.training import pipeline

    real = pipeline.train_pipeline
    monkeypatch.setattr(
        pipeline,
        "train_pipeline",
        lambda workdir, config: real(workdir, pipeline.TrainConfig(
            seed=config.seed, docs_per_schema=5, eval_docs_per_schema=2, free_text_docs=20,
            vocab_size=300,
        )),
    )  # fmt: skip
    assert cli.main(["train", "--workdir", str(tmp_path / "model"), "--seed", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["model_version"]) == 64
    assert (tmp_path / "model" / "registry").is_dir()


def test_cli_issue_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from grammar_constrained_decoder.core.config import get_settings

    get_settings.cache_clear()
    try:
        assert cli.main(["issue-token", "--sub", "alice", "--scope", "generate"]) == 2
        monkeypatch.setenv("APP_JWT_SECRET", "cli-secret-" + "z" * 40)
        get_settings.cache_clear()
        assert (
            cli.main(["issue-token", "--sub", "alice", "--scope", "generate", "--ttl", "60"]) == 0
        )
        token = capsys.readouterr().out.strip()
        assert token.count(".") == 2
    finally:
        get_settings.cache_clear()


def test_cli_generate_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate-api-key"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["api_key_sha256"] == hash_api_key(payload["api_key"])


def test_cli_audit_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("admin", "schema.register", {"schema_id": "x"})
    assert cli.main(["audit", "verify", str(path), "--expect-head", log.head]) == 0
    assert cli.main(["audit", "verify", str(path), "--expect-head", "0" * 64]) == 1
    capsys.readouterr()


@pytest.mark.usefixtures("restore_logging")
def test_cli_serve_configures_uvicorn_safely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import uvicorn

    from grammar_constrained_decoder.core.config import get_settings

    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APP_MODEL_DIR", str(tmp_path / "missing"))
    get_settings.cache_clear()
    try:
        assert cli.main(["serve", "--port", "8123"]) == 0
    finally:
        get_settings.cache_clear()
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False


def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["grammar_constrained_decoder", "generate-api-key"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("grammar_constrained_decoder", run_name="__main__")
    assert exit_info.value.code == 0
    assert "api_key_sha256" in capsys.readouterr().out
