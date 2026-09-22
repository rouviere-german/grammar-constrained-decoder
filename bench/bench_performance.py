"""Mask overhead per token, compile time vs schema size, and stress rejections.

- **Mask overhead**: constrained greedy and top-p decoding of every held-out schema (10 samples
  each); the time to fetch the token mask and apply it to the logits is measured per token and
  compared with the time of one language-model step.
- **Compile time vs size**: every bundled schema compiled cold (fresh compiler, no cache), with
  canonical schema bytes, DFA states and token-index entries; then warm (memory cache) and from
  the disk cache.
- **Stress**: adversarial schemas (deep nesting, huge enums, exponential and nested-counter
  patterns, huge bounds ...) must be rejected with a typed reason within the compile timeout.

Usage: uv run python bench/bench_performance.py [--check]
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from _common import percentile, report_and_check, train_model, write_results
from grammar_constrained_decoder.automata.limits import CompileError, CompileLimits
from grammar_constrained_decoder.core.canonical import canonical_json
from grammar_constrained_decoder.decoding.compiler import GrammarCompiler
from grammar_constrained_decoder.decoding.decoder import Decoder
from grammar_constrained_decoder.training.corpus import load_bundled_schemas, prompt_for

STRESS_TIMEOUT = 2.0


def _deep(levels: int) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "null"}
    for _ in range(levels):
        schema = {"type": "array", "items": schema}
    return schema


def _ref_chain(links: int) -> dict[str, Any]:
    definitions: dict[str, Any] = {f"d{i}": {"$ref": f"#/$defs/d{i + 1}"} for i in range(links)}
    definitions[f"d{links}"] = {"type": "null"}
    return {"$defs": definitions, "$ref": "#/$defs/d0"}


STRESS: list[tuple[str, dict[str, Any]]] = [
    ("deep_nesting_200", _deep(200)),
    ("enum_10000", {"enum": [f"v{i}" for i in range(10_000)]}),
    ("pattern_5000_chars", {"type": "string", "pattern": "a" * 5_000}),
    ("exponential_dfa", {"type": "string", "pattern": "^(a|b)*a(a|b){20}$"}),
    ("nested_counters", {"type": "string", "pattern": "^((a{0,90}){0,90}){0,90}$"}),
    ("max_length_1e9", {"type": "string", "maxLength": 10**9}),
    ("bound_1e40", {"type": "integer", "maximum": 10**40}),
    (
        "object_500_properties",
        {"type": "object", "properties": {f"p{i}": {"type": "null"} for i in range(500)}},
    ),
    ("any_of_5000", {"anyOf": [{"type": "null"}] * 5_000}),
    ("backreference", {"type": "string", "pattern": "(a+)\\1"}),
    ("ref_chain_600", _ref_chain(600)),
    ("lone_surrogate_enum", {"enum": ["ok", chr(0xD800)]}),
]


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - one linear report
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        model = train_model(Path(tmp))
        vocab = model.tokenizer.vocab
        decoder = Decoder(model.lm, vocab)
        schemas = load_bundled_schemas()

        compile_rows: list[dict[str, Any]] = []
        for item in schemas:
            started = time.perf_counter()
            compiled = GrammarCompiler(vocab).compile(item.schema)
            cold = time.perf_counter() - started
            compile_rows.append(
                {
                    "schema": item.id,
                    "bytes": len(canonical_json(item.schema)),
                    "dfa_states": compiled.stats.dfa_states,
                    "index_entries": compiled.stats.index_entries,
                    "cold_seconds": round(cold, 3),
                }
            )
        cache_dir = Path(tmp) / "cache"
        warm = GrammarCompiler(vocab, cache_dir=cache_dir)
        invoice = next(item for item in schemas if item.id == "invoice")
        warm.compile(invoice.schema)
        started = time.perf_counter()
        warm.compile(invoice.schema)
        memory_hit = time.perf_counter() - started
        started = time.perf_counter()
        GrammarCompiler(vocab, cache_dir=cache_dir).compile(invoice.schema)
        disk_hit = time.perf_counter() - started

        mask_seconds: list[float] = []
        step_seconds: list[float] = []
        compiler = GrammarCompiler(vocab)
        for item in (entry for entry in schemas if entry.split == "eval"):
            grammar = compiler.compile(item.schema)
            prompt = model.tokenizer.encode(prompt_for(item))
            for seed in range(10):
                strategy: Literal["greedy", "top_p"] = "greedy" if seed == 0 else "top_p"
                out = decoder.generate(
                    prompt, grammar=grammar.index, max_tokens=256, strategy=strategy, seed=seed
                )
                mask_seconds.extend(out.mask_seconds)
            for _ in range(200):
                started = time.perf_counter()
                model.lm.logprobs(prompt)
                step_seconds.append(time.perf_counter() - started)

    stress_rows: list[dict[str, Any]] = []
    limits = CompileLimits(timeout_seconds=STRESS_TIMEOUT)
    for name, schema in STRESS:
        started = time.perf_counter()
        try:
            GrammarCompiler(vocab, limits=limits).compile(schema)
            reason = "accepted"
        except CompileError as exc:
            reason = exc.reason
        stress_rows.append(
            {"case": name, "reason": reason, "seconds": round(time.perf_counter() - started, 3)}
        )

    print("| schema | canonical bytes | DFA states | index entries | cold compile (s) |")
    print("|---|---:|---:|---:|---:|")
    for row in sorted(compile_rows, key=lambda r: r["bytes"]):
        print(
            f"| {row['schema']} | {row['bytes']} | {row['dfa_states']} | "
            f"{row['index_entries']} | {row['cold_seconds']} |"
        )
    print("\n| stress case | rejection reason | seconds |")
    print("|---|---|---:|")
    for row in stress_rows:
        print(f"| {row['case']} | {row['reason']} | {row['seconds']} |")

    colds = [row["cold_seconds"] for row in compile_rows]
    metrics = {
        "mask_tokens_measured": float(len(mask_seconds)),
        "mask_p50_us": round(percentile(mask_seconds, 50) * 1e6, 1),
        "mask_p95_us": round(percentile(mask_seconds, 95) * 1e6, 1),
        "mask_p99_us": round(percentile(mask_seconds, 99) * 1e6, 1),
        "lm_step_p50_us": round(percentile(step_seconds, 50) * 1e6, 1),
        "compile_cold_median_s": round(percentile(colds, 50), 3),
        "compile_cold_max_s": round(max(colds), 3),
        "compile_memory_hit_ms": round(memory_hit * 1e3, 3),
        "compile_disk_hit_ms": round(disk_hit * 1e3, 1),
        "stress_cases": float(len(stress_rows)),
        "stress_accepted": float(sum(row["reason"] == "accepted" for row in stress_rows)),
        "stress_max_seconds": max(row["seconds"] for row in stress_rows),
    }
    write_results(
        "performance", {"metrics": metrics, "compile": compile_rows, "stress": stress_rows}
    )
    return report_and_check("performance", metrics, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
