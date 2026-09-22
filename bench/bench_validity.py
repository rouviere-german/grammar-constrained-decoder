"""Validity of generated JSON: constrained vs unconstrained vs unconstrained with 3 retries.

Protocol (fixed seeds, CPU only, a few minutes):

- the model is trained from scratch (18 train schemas); generation targets the 12 held-out
  schemas the model never saw, 1,000 generations in total (84 or 83 per schema);
- every mode samples with nucleus sampling (top_p 0.95, temperature 1.0) from the same prompt
  ``"Q: <description>\\n<schema id>:\\n"`` with ``max_tokens`` 256;
- **constrained**: masked decoding with the compiled grammar;
- **unconstrained**: the same sampler without a mask;
- **unconstrained + 3 retries**: up to four unconstrained attempts, stopping at the first valid
  one (tokens of failed attempts count as wasted);
- validity = ``json.loads`` succeeds and ``jsonschema`` (Draft 2020-12) accepts the value;
- greedy decoding (one per schema) and 20 samples per *training* schema are reported for context.

Usage: uv run python bench/bench_validity.py [--check] [--total 1000]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from _common import report_and_check, train_model, write_results
from grammar_constrained_decoder.decoding.compiler import GrammarCompiler
from grammar_constrained_decoder.decoding.decoder import Decoder
from grammar_constrained_decoder.training.corpus import load_bundled_schemas, prompt_for

MAX_TOKENS = 256
TOP_P = 0.95


def _valid(validator: Draft202012Validator, text: str) -> bool:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return False
    return bool(validator.is_valid(value))


def run(total: int) -> tuple[dict[str, float], list[dict[str, Any]]]:  # noqa: PLR0915
    with tempfile.TemporaryDirectory() as tmp:
        model = train_model(Path(tmp))
    tokenizer = model.tokenizer
    decoder = Decoder(model.lm, tokenizer.vocab)
    compiler = GrammarCompiler(tokenizer.vocab)
    schemas = load_bundled_schemas()
    held_out = [item for item in schemas if item.split == "eval"]
    train = [item for item in schemas if item.split == "train"]
    base, extra = divmod(total, len(held_out))
    rows: list[dict[str, Any]] = []
    totals = {"constrained": 0, "unconstrained": 0, "retry": 0, "n": 0}
    tokens = {"constrained": 0, "unconstrained": 0, "retry": 0}
    started = time.perf_counter()
    for index, schema in enumerate(held_out):
        count = base + (1 if index < extra else 0)
        validator = Draft202012Validator(schema.schema)
        grammar = compiler.compile(schema.schema)
        prompt = tokenizer.encode(prompt_for(schema))
        row: dict[str, Any] = {"schema": schema.id, "n": count}
        c_valid = u_valid = r_valid = 0
        for draw in range(count):
            seed = 1_000 * index + draw
            constrained = decoder.generate(
                prompt,
                grammar=grammar.index,
                max_tokens=MAX_TOKENS,
                strategy="top_p",
                top_p=TOP_P,
                seed=seed,
            )
            c_valid += _valid(validator, constrained.text)
            tokens["constrained"] += len(constrained.token_ids)
            free = decoder.generate(
                prompt,
                grammar=None,
                max_tokens=MAX_TOKENS,
                strategy="top_p",
                top_p=TOP_P,
                seed=seed,
            )
            u_valid += _valid(validator, free.text)
            tokens["unconstrained"] += len(free.token_ids)
            attempt_text, spent = free.text, len(free.token_ids)
            for retry in range(1, 4):
                if _valid(validator, attempt_text):
                    break
                again = decoder.generate(
                    prompt, grammar=None, max_tokens=MAX_TOKENS, strategy="top_p", top_p=TOP_P,
                    seed=seed + 100_000 * retry,
                )  # fmt: skip
                attempt_text = again.text
                spent += len(again.token_ids)
            r_valid += _valid(validator, attempt_text)
            tokens["retry"] += spent
        greedy_c = decoder.generate(prompt, grammar=grammar.index, max_tokens=MAX_TOKENS)
        greedy_u = decoder.generate(prompt, grammar=None, max_tokens=MAX_TOKENS)
        row.update(
            constrained=c_valid / count,
            unconstrained=u_valid / count,
            unconstrained_3_retries=r_valid / count,
            greedy_constrained=_valid(validator, greedy_c.text),
            greedy_unconstrained=_valid(validator, greedy_u.text),
            dfa_states=grammar.stats.dfa_states,
        )
        rows.append(row)
        totals["constrained"] += c_valid
        totals["unconstrained"] += u_valid
        totals["retry"] += r_valid
        totals["n"] += count
    held_out_seconds = time.perf_counter() - started

    seen_valid = seen_total = 0
    for index, schema in enumerate(train):
        validator = Draft202012Validator(schema.schema)
        prompt = tokenizer.encode(prompt_for(schema))
        for draw in range(20):
            out = decoder.generate(
                prompt, grammar=None, max_tokens=MAX_TOKENS, strategy="top_p", top_p=TOP_P,
                seed=500_000 + 100 * index + draw,
            )  # fmt: skip
            seen_valid += _valid(validator, out.text)
            seen_total += 1

    n = totals["n"]
    metrics = {
        "generations": float(n),
        "constrained_valid_ratio": round(totals["constrained"] / n, 4),
        "unconstrained_valid_ratio": round(totals["unconstrained"] / n, 4),
        "unconstrained_3_retries_valid_ratio": round(totals["retry"] / n, 4),
        "constrained_tokens_per_generation": round(tokens["constrained"] / n, 1),
        "unconstrained_tokens_per_generation": round(tokens["unconstrained"] / n, 1),
        "retry_tokens_per_generation": round(tokens["retry"] / n, 1),
        "greedy_constrained_valid_schemas": float(sum(r["greedy_constrained"] for r in rows)),
        "greedy_unconstrained_valid_schemas": float(sum(r["greedy_unconstrained"] for r in rows)),
        "seen_schema_unconstrained_valid_ratio": round(seen_valid / seen_total, 4),
        "held_out_seconds": round(held_out_seconds, 1),
    }
    return metrics, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--total", type=int, default=1_000)
    args = parser.parse_args(argv)
    metrics, rows = run(args.total)
    write_results("validity", {"metrics": metrics, "per_schema": rows})
    print("| held-out schema | n | constrained | unconstrained | + 3 retries | greedy c/u |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows:
        greedy_c = "ok" if row["greedy_constrained"] else "x"
        greedy = f"{greedy_c}/{'ok' if row['greedy_unconstrained'] else 'x'}"
        print(
            f"| {row['schema']} | {row['n']} | {row['constrained']:.1%} | "
            f"{row['unconstrained']:.1%} | {row['unconstrained_3_retries']:.1%} | {greedy} |"
        )
    return report_and_check("validity", metrics, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
