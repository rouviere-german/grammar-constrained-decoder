"""Command line entry point.

Run as ``grammar-constrained-decoder <command>`` or ``python -m grammar_constrained_decoder``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

__all__ = ["main"]


def _serve(host: str | None, port: int | None) -> int:
    import uvicorn  # noqa: PLC0415 - only needed when serving

    from grammar_constrained_decoder.core.config import get_settings  # noqa: PLC0415
    from grammar_constrained_decoder.core.logging import configure_logging  # noqa: PLC0415
    from grammar_constrained_decoder.core.tracing import (  # noqa: PLC0415
        install_global_tracer_provider,
    )
    from grammar_constrained_decoder.serving.api import build_app  # noqa: PLC0415

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    app = build_app(settings)
    install_global_tracer_provider(app.state.tracer_provider)
    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )
    return 0


def _train(workdir: Path, seed: int) -> int:
    from grammar_constrained_decoder.training.pipeline import (  # noqa: PLC0415
        TrainConfig,
        train_pipeline,
    )

    result = train_pipeline(workdir, TrainConfig(seed=seed))
    report = {
        "model_version": result.version.digest,
        "metrics": result.metrics,
        "seconds": {name: round(value, 3) for name, value in result.seconds.items()},
        "registry": str(workdir / "registry"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _issue_token(subject: str, scopes: list[str], ttl: int) -> int:
    from grammar_constrained_decoder.core.config import get_settings  # noqa: PLC0415
    from grammar_constrained_decoder.core.security import issue_token  # noqa: PLC0415

    settings = get_settings()
    if settings.jwt_secret is None:
        print("APP_JWT_SECRET is not set; refusing to mint a token.", file=sys.stderr)
        return 2
    token = issue_token(
        settings.jwt_secret.get_secret_value(),
        subject=subject,
        scopes=scopes,
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        ttl_seconds=min(ttl, settings.jwt_max_ttl_seconds),
    )
    print(token)
    return 0


def _compile(path: Path, model_dir: Path) -> int:
    from grammar_constrained_decoder.automata.limits import CompileError  # noqa: PLC0415
    from grammar_constrained_decoder.decoding.compiler import GrammarCompiler  # noqa: PLC0415
    from grammar_constrained_decoder.training.pipeline import load_model  # noqa: PLC0415

    model = load_model(model_dir)
    compiler = GrammarCompiler(model.tokenizer.vocab)
    try:
        compiled = compiler.compile(json.loads(path.read_text(encoding="utf-8")))
    except CompileError as exc:
        print(json.dumps({"ok": False, "reason": exc.reason, "error": str(exc)}))
        return 1
    report = {"ok": True, "schema_sha256": compiled.schema_sha256, **asdict(compiled.stats)}
    print(json.dumps(report, sort_keys=True))
    return 0


def _demo(workdir: Path, samples: int) -> int:
    from grammar_constrained_decoder.core.logging import configure_logging  # noqa: PLC0415
    from grammar_constrained_decoder.demo import run_demo  # noqa: PLC0415

    configure_logging("WARNING", json_output=True, stream=sys.stderr)
    summary = run_demo(workdir, samples=samples)
    print(
        "\n"
        + json.dumps(
            {"demo": "grammar-constrained-decoder", **summary},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911 - one exit per command
    parser = argparse.ArgumentParser(prog="grammar-constrained-decoder")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    train = sub.add_parser("train", help="generate the corpus, train tokenizer and LM, register")
    train.add_argument("--workdir", type=Path, default=Path("var/model"))
    train.add_argument("--seed", type=int, default=0)

    sub.add_parser("generate-api-key", help="print a new API key and its SHA-256 digest")

    token = sub.add_parser("issue-token", help="mint a scoped JWT with APP_JWT_SECRET")
    token.add_argument("--sub", required=True)
    token.add_argument("--scope", action="append", default=[], required=True)
    token.add_argument("--ttl", type=int, default=3600)

    compile_cmd = sub.add_parser("compile", help="compile a schema file and print statistics")
    compile_cmd.add_argument("schema", type=Path)
    compile_cmd.add_argument("--model-dir", type=Path, default=Path("var/model/registry"))

    audit = sub.add_parser("audit", help="audit log tools")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify", help="verify a hash-chained audit log")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-head", default=None)

    demo = sub.add_parser("demo", help="run the offline end-to-end demo")
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))
    demo.add_argument("--samples", type=int, default=20)

    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args.host, args.port)
    if args.command == "train":
        return _train(args.workdir, args.seed)
    if args.command == "generate-api-key":
        from grammar_constrained_decoder.core.security import generate_api_key  # noqa: PLC0415

        raw, digest = generate_api_key()
        print(json.dumps({"api_key": raw, "api_key_sha256": digest}, indent=2))
        print("Store only api_key_sha256 in APP_API_KEY_SHA256.", file=sys.stderr)
        return 0
    if args.command == "issue-token":
        return _issue_token(args.sub, args.scope, args.ttl)
    if args.command == "compile":
        return _compile(args.schema, args.model_dir)
    if args.command == "audit":
        from grammar_constrained_decoder.core.audit import main as audit_main  # noqa: PLC0415

        forwarded = ["verify", str(args.path)]
        if args.expect_head:
            forwarded += ["--expect-head", args.expect_head]
        return audit_main(forwarded)
    return _demo(args.workdir, args.samples)


if __name__ == "__main__":
    raise SystemExit(main())
