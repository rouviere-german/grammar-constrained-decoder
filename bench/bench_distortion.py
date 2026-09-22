"""Distribution distortion of masked sampling on 20 tiny schemas with enumerable languages.

For each schema the constrained token language is enumerated exactly (every tokenization of every
valid string) under the trained 5-gram model with the prompt ``"Q: <description>\\n<id>:\\n"``:

- ``P(. | valid)``: the model's distribution restricted to valid sequences, renormalized globally
  (what an ideal constrained sampler would draw from);
- ``Q``: the distribution masked, locally renormalized sampling actually draws from.

Reported per schema: KL(P(. | valid) || Q) over strings and over token sequences (nats), the
probability mass the unconstrained model puts on valid sequences, whether greedy decoding returns
the conditional mode, and the total variation between 300 real decoder samples and the exact Q
(a check of the decoder implementation, expected to be small but non-zero from sampling noise).

Usage: uv run python bench/bench_distortion.py [--check] [--samples 300]
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from _common import report_and_check, train_model, write_results
from grammar_constrained_decoder.decoding.compiler import GrammarCompiler
from grammar_constrained_decoder.decoding.distortion import distortion_report
from grammar_constrained_decoder.training.corpus import load_distortion_schemas, prompt_for


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        model = train_model(Path(tmp))
    compiler = GrammarCompiler(model.tokenizer.vocab)
    rows: list[dict[str, Any]] = []
    for index, tiny in enumerate(load_distortion_schemas()):
        grammar = compiler.compile(tiny.schema)
        prompt = model.tokenizer.encode(prompt_for(tiny))
        report = distortion_report(
            model.lm, grammar.index, model.tokenizer.vocab, prompt, samples=args.samples,
            seed=10_000 * index,
        )  # fmt: skip
        rows.append({"schema": tiny.id, **report.as_dict()})
    print(
        "| schema | strings | token seqs | valid mass | KL string | KL token "
        "| greedy = mode | TV samples |"
    )
    print("|---|---:|---:|---:|---:|---:|---|---:|")
    for row in rows:
        print(
            f"| {row['schema']} | {row['n_strings']} | {row['n_sequences']} | "
            f"{row['valid_mass']:.2e} | {row['kl_string']:.4f} | {row['kl_token']:.4f} | "
            f"{'yes' if row['greedy_is_conditional_mode'] else 'no'} | "
            f"{row['sampled_tv_to_masked']:.3f} |"
        )
    kl_string = np.array([row["kl_string"] for row in rows])
    kl_token = np.array([row["kl_token"] for row in rows])
    metrics = {
        "schemas": float(len(rows)),
        "kl_string_mean": round(float(kl_string.mean()), 4),
        "kl_string_median": round(float(np.median(kl_string)), 4),
        "kl_string_max": round(float(kl_string.max()), 4),
        "kl_token_mean": round(float(kl_token.mean()), 4),
        "greedy_is_mode_ratio": round(
            sum(r["greedy_is_conditional_mode"] for r in rows) / len(rows), 3
        ),
        "sampled_tv_max": round(max(r["sampled_tv_to_masked"] for r in rows), 3),
        "kl_string_exceeds_kl_token": float(
            sum(r["kl_string"] > r["kl_token"] + 1e-9 for r in rows)
        ),
    }
    write_results("distortion", {"metrics": metrics, "per_schema": rows})
    return report_and_check("distortion", metrics, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
