"""Micro-benchmarks for the hot paths, with CI regression thresholds.

Usage:
  uv run python bench/bench_baseline.py          # run, write bench/results/baseline.json
  uv run python bench/bench_baseline.py --check   # also exit 1 if bench/thresholds.json is exceeded

Thresholds are deliberately loose (several times the numbers measured on a laptop) because shared
CI runners are noisy; they catch order-of-magnitude regressions, not small drifts.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from _common import check_thresholds, train_model, write_results
from grammar_constrained_decoder.automata.dfa import compile_regex
from grammar_constrained_decoder.automata.pattern import compile_pattern
from grammar_constrained_decoder.core.ratelimit import RateLimiter
from grammar_constrained_decoder.core.redaction import redact_text
from grammar_constrained_decoder.decoding.compiler import GrammarCompiler
from grammar_constrained_decoder.decoding.decoder import GrammarCursor
from grammar_constrained_decoder.ml.drift import DriftMonitor
from grammar_constrained_decoder.training.corpus import load_bundled_schemas, prompt_for


def _timed(function: Callable[[], object], runs: int, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    micros = np.asarray(samples, dtype=np.float64) / 1_000.0
    return {
        "runs": runs,
        "p50_us": round(float(np.percentile(micros, 50)), 1),
        "p95_us": round(float(np.percentile(micros, 95)), 1),
    }


def run() -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    line = "user jane.doe@example.com paid; " + "lorem ipsum dolor sit amet " * 35
    results["redact_text_1kb"] = _timed(lambda: redact_text(line), runs=2_000)
    limiter = RateLimiter(rate=1e9, burst=1_000_000_000)
    results["rate_limiter_check"] = _timed(lambda: limiter.check("client"), runs=20_000)
    rng = np.random.default_rng(0)
    monitor = DriftMonitor({"x": rng.normal(size=10_000)})
    current = {"x": rng.normal(loc=0.1, size=10_000)}
    results["drift_check_10k_vs_10k"] = _timed(lambda: monitor.check(current), runs=30, warmup=2)

    dfa = compile_regex(compile_pattern("^(a+)+[a-z0-9._]+@[a-z]+$"))
    evil = b"a" * 10_000 + b"!"
    results["dfa_match_10kb_redos_input"] = _timed(lambda: dfa.accepts(evil), runs=200)

    with tempfile.TemporaryDirectory() as tmp:
        model = train_model(Path(tmp))
    schema = next(item for item in load_bundled_schemas() if item.id == "invoice")
    compiler = GrammarCompiler(model.tokenizer.vocab)
    grammar = compiler.compile(schema.schema)
    prompt = model.tokenizer.encode(prompt_for(schema))
    results["lm_logprobs_step"] = _timed(lambda: model.lm.logprobs(prompt), runs=2_000)
    cursor = GrammarCursor(grammar.index, 256)
    logp = model.lm.logprobs(prompt)

    def masked_step() -> None:
        np.where(cursor.mask(), logp, -np.inf)

    results["mask_and_apply_step"] = _timed(masked_step, runs=5_000)
    results["compile_invoice_memory_hit"] = _timed(
        lambda: compiler.compile(schema.schema), runs=200
    )
    results["compile_invoice_cold"] = _timed(
        lambda: GrammarCompiler(model.tokenizer.vocab).compile(schema.schema), runs=3, warmup=1
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="hot-path micro-benchmarks")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    results = run()
    write_results("baseline", {"results": results})
    print("| benchmark | runs | p50 (us) | p95 (us) |")
    print("|---|---:|---:|---:|")
    for name, stats in results.items():
        print(f"| {name} | {stats['runs']:.0f} | {stats['p50_us']} | {stats['p95_us']} |")
    if args.check:
        flat = {f"{name}.p95_us": stats["p95_us"] for name, stats in results.items()}
        problems = check_thresholds("baseline", flat)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
