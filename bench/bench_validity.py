"""Validity of generated JSON: constrained vs unconstrained vs unconstrained with 3 retries.

Protocol (fixed seeds, CPU only, about a minute):

- the model is trained from scratch (18 train schemas); generation targets the 12 held-out
  schemas the model never saw, 1,000 generations in total (84 or 83 per schema);
- every mode samples with nucleus sampling (top_p 0.95, temperature 1.0) from the same prompt
  ``"Q: <description>\\n<schema id>:\\n"`` with ``max_tokens`` 256;
- **constrained**: masked decoding with the compiled grammar;
- **unconstrained**: the same sampler without a mask;
- **unconstrained + 3 retries**: up to four unconstrained attempts, stopping at the first valid
  one (tokens of failed attempts count as wasted);
- validity = ``json.loads`` succeeds and ``jsonschema`` (Draft 2020-12, stock validator, whose
  ``pattern`` uses Python ``re`` on these trusted bundled patterns) accepts the value. Every
  output is also checked with the service's ``SafeValidator`` (ECMA patterns by DFA); the two
  must agree on every constrained output (``constrained_validator_disagreements``);
- partial credit for unconstrained output: the share that parses as JSON, the share of the
  schema's top-level required keys present, and the viable prefix (bytes before the output
  leaves every valid document, as a share of its length);
- a second, stronger baseline: for each held-out schema a **single-schema 5-gram** trained only
  on 120 sampled documents of that schema (the same tokenizer), 20 samples per schema, with the
  same three modes. It has seen the target format, so it separates "the model forgot which
  schema it is writing" from "the model cannot keep a document valid";
- greedy decoding (one per schema) and 20 samples per *training* schema are reported for context.

Usage: uv run python bench/bench_validity.py [--check] [--total 1000]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from _common import report_and_check, train_model, write_results
from grammar_constrained_decoder.decoding.compiler import CompiledGrammar, GrammarCompiler
from grammar_constrained_decoder.decoding.decoder import Decoder, Generation
from grammar_constrained_decoder.lm.kneser_ney import KneserNeyLM
from grammar_constrained_decoder.lm.tokenizer import BPETokenizer
from grammar_constrained_decoder.schema.validate import safe_validator
from grammar_constrained_decoder.training.corpus import (
    BundledSchema,
    generate_corpus,
    load_bundled_schemas,
    prompt_for,
)

MAX_TOKENS = 256
TOP_P = 0.95
SINGLE_SCHEMA_DOCS = 120
SINGLE_SCHEMA_SAMPLES = 20


class Judge:
    """Validity (stock and safe validators) and partial credit for one schema."""

    def __init__(self, schema: BundledSchema, grammar: CompiledGrammar) -> None:
        self.stock = Draft202012Validator(schema.schema)
        self.safe = safe_validator(schema.schema)
        self.grammar = grammar
        required = schema.schema.get("required", [])
        self.required = [str(name) for name in required] if isinstance(required, list) else []

    def valid(self, text: str) -> bool:
        return bool(self.judge(text)["valid"])

    def judge(self, text: str) -> dict[str, Any]:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value, parsed = None, False
        else:
            parsed = True
        stock = parsed and bool(self.stock.is_valid(value))
        safe = parsed and bool(self.safe.is_valid(value))
        if self.required and isinstance(value, dict):
            required = sum(name in value for name in self.required) / len(self.required)
        else:
            required = 1.0 if parsed and isinstance(value, dict) else 0.0
        data = text.encode("utf-8")
        dfa, state, viable = self.grammar.dfa, self.grammar.dfa.start, 0
        for byte in data:
            state = dfa.step(state, byte)
            if state == dfa.dead:
                break
            viable += 1
        return {
            "valid": stock,
            "disagree": stock != safe,
            "parsed": parsed,
            "required": required,
            "viable": viable / len(data) if data else 0.0,
        }


@dataclass
class Tally:
    n: int = 0
    valid: int = 0
    parsed: int = 0
    disagreements: int = 0
    required: float = 0.0
    viable: float = 0.0
    tokens: int = 0

    def merge(self, other: Tally) -> None:
        for name in ("n", "valid", "parsed", "disagreements", "required", "viable", "tokens"):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def add(self, verdict: dict[str, Any], tokens: int) -> None:
        self.n += 1
        self.valid += verdict["valid"]
        self.parsed += verdict["parsed"]
        self.disagreements += verdict["disagree"]
        self.required += verdict["required"]
        self.viable += verdict["viable"]
        self.tokens += tokens

    def ratio(self, name: str) -> float:
        value = {"valid": self.valid, "parsed": self.parsed}.get(name)
        total = float(value) if value is not None else float(getattr(self, name))
        return round(total / max(self.n, 1), 4)


def sample(
    decoder: Decoder, prompt: Sequence[int], grammar: CompiledGrammar | None, seed: int
) -> Generation:
    return decoder.generate(
        prompt,
        grammar=grammar.index if grammar is not None else None,
        max_tokens=MAX_TOKENS,
        strategy="top_p",
        top_p=TOP_P,
        seed=seed,
    )


def three_modes(
    decoder: Decoder,
    judge: Judge,
    prompt: Sequence[int],
    grammar: CompiledGrammar,
    *,
    seed: int,
    tallies: tuple[Tally, Tally, Tally],
) -> None:
    """One constrained, one unconstrained and one retried-unconstrained sample."""
    constrained, free, retry = tallies
    out = sample(decoder, prompt, grammar, seed)
    constrained.add(judge.judge(out.text), len(out.token_ids))
    out = sample(decoder, prompt, None, seed)
    verdict = judge.judge(out.text)
    free.add(verdict, len(out.token_ids))
    spent = len(out.token_ids)
    for attempt in range(1, 4):
        if verdict["valid"]:
            break
        out = sample(decoder, prompt, None, seed + 100_000 * attempt)
        verdict = judge.judge(out.text)
        spent += len(out.token_ids)
    retry.add(verdict, spent)


def single_schema_decoder(tokenizer: BPETokenizer, schema: BundledSchema, seed: int) -> Decoder:
    records = generate_corpus(
        [schema], docs_per_schema=SINGLE_SCHEMA_DOCS, free_text_docs=0, seed=seed
    )
    eos = tokenizer.vocab.eos_id
    documents = [[*tokenizer.encode(record.text), eos] for record in records]
    lm = KneserNeyLM.fit(documents, vocab_size=tokenizer.vocab.size, order=5, bos_id=eos)
    return Decoder(lm, tokenizer.vocab)


def run(total: int) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
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
    weak = (Tally(), Tally(), Tally())
    started = time.perf_counter()
    greedy_c = greedy_u = 0
    for index, schema in enumerate(held_out):
        count = base + (1 if index < extra else 0)
        grammar = compiler.compile(schema.schema)
        judge = Judge(schema, grammar)
        prompt = tokenizer.encode(prompt_for(schema))
        local = (Tally(), Tally(), Tally())
        for draw in range(count):
            three_modes(decoder, judge, prompt, grammar, seed=1_000 * index + draw, tallies=local)
        for mine, overall in zip(local, weak, strict=True):
            overall.merge(mine)
        g_c = judge.valid(sample_greedy(decoder, prompt, grammar).text)
        g_u = judge.valid(sample_greedy(decoder, prompt, None).text)
        greedy_c += g_c
        greedy_u += g_u
        rows.append(
            {
                "schema": schema.id,
                "n": count,
                "constrained": local[0].ratio("valid"),
                "unconstrained": local[1].ratio("valid"),
                "unconstrained_3_retries": local[2].ratio("valid"),
                "unconstrained_parsed": local[1].ratio("parsed"),
                "unconstrained_viable_prefix": local[1].ratio("viable"),
                "greedy_constrained": g_c,
                "greedy_unconstrained": g_u,
                "dfa_states": grammar.stats.dfa_states,
            }
        )
    held_out_seconds = time.perf_counter() - started

    mark = time.perf_counter()
    single = (Tally(), Tally(), Tally())
    single_rows: list[dict[str, Any]] = []
    for index, schema in enumerate(held_out):
        grammar = compiler.compile(schema.schema)
        judge = Judge(schema, grammar)
        prompt = tokenizer.encode(prompt_for(schema))
        specialist = single_schema_decoder(tokenizer, schema, seed=20_000 + index)
        local = (Tally(), Tally(), Tally())
        for draw in range(SINGLE_SCHEMA_SAMPLES):
            seed = 700_000 + 1_000 * index + draw
            three_modes(specialist, judge, prompt, grammar, seed=seed, tallies=local)
        for mine, overall in zip(local, single, strict=True):
            overall.merge(mine)
        single_rows.append(
            {
                "schema": schema.id,
                "n": SINGLE_SCHEMA_SAMPLES,
                "constrained": local[0].ratio("valid"),
                "unconstrained": local[1].ratio("valid"),
                "unconstrained_3_retries": local[2].ratio("valid"),
                "unconstrained_parsed": local[1].ratio("parsed"),
                "unconstrained_viable_prefix": local[1].ratio("viable"),
            }
        )
    single_seconds = time.perf_counter() - mark

    seen_valid = seen_total = 0
    for index, schema in enumerate(train):
        judge = Judge(schema, compiler.compile(schema.schema))
        prompt = tokenizer.encode(prompt_for(schema))
        for draw in range(20):
            out = sample(decoder, prompt, None, 500_000 + 100 * index + draw)
            seen_valid += judge.valid(out.text)
            seen_total += 1

    constrained, free, retry = weak
    s_constrained, s_free, s_retry = single
    n = constrained.n
    metrics = {
        "generations": float(n),
        "constrained_valid_ratio": constrained.ratio("valid"),
        "unconstrained_valid_ratio": free.ratio("valid"),
        "unconstrained_3_retries_valid_ratio": retry.ratio("valid"),
        "unconstrained_json_parse_ratio": free.ratio("parsed"),
        "unconstrained_required_keys_ratio": free.ratio("required"),
        "unconstrained_viable_prefix_ratio": free.ratio("viable"),
        "constrained_tokens_per_generation": round(constrained.tokens / n, 1),
        "unconstrained_tokens_per_generation": round(free.tokens / n, 1),
        "retry_tokens_per_generation": round(retry.tokens / n, 1),
        "greedy_constrained_valid_schemas": float(greedy_c),
        "greedy_unconstrained_valid_schemas": float(greedy_u),
        "seen_schema_unconstrained_valid_ratio": round(seen_valid / seen_total, 4),
        "constrained_validator_disagreements": float(
            constrained.disagreements + s_constrained.disagreements
        ),
        "validator_disagreements": float(sum(tally.disagreements for tally in (*weak, *single))),
        "single_schema_generations": float(s_constrained.n),
        "single_schema_constrained_valid_ratio": s_constrained.ratio("valid"),
        "single_schema_unconstrained_valid_ratio": s_free.ratio("valid"),
        "single_schema_3_retries_valid_ratio": s_retry.ratio("valid"),
        "single_schema_unconstrained_json_parse_ratio": s_free.ratio("parsed"),
        "single_schema_unconstrained_required_keys_ratio": s_free.ratio("required"),
        "single_schema_unconstrained_viable_prefix_ratio": s_free.ratio("viable"),
        "single_schema_retry_tokens_per_generation": round(s_retry.tokens / max(s_retry.n, 1), 1),
        "held_out_seconds": round(held_out_seconds, 1),
        "single_schema_seconds": round(single_seconds, 1),
    }
    return metrics, rows, single_rows


def sample_greedy(
    decoder: Decoder, prompt: Sequence[int], grammar: CompiledGrammar | None
) -> Generation:
    return decoder.generate(
        prompt, grammar=grammar.index if grammar is not None else None, max_tokens=MAX_TOKENS
    )


def print_table(title: str, rows: list[dict[str, Any]], *, greedy: bool) -> None:
    print(f"\n{title}")
    header = "| held-out schema | n | constrained | unconstrained | + 3 retries | parses | viable |"
    print(header + (" greedy c/u |" if greedy else ""))
    print("|---|---:|---:|---:|---:|---:|---:|" + ("---|" if greedy else ""))
    for row in rows:
        line = (
            f"| {row['schema']} | {row['n']} | {row['constrained']:.1%} | "
            f"{row['unconstrained']:.1%} | {row['unconstrained_3_retries']:.1%} | "
            f"{row['unconstrained_parsed']:.1%} | {row['unconstrained_viable_prefix']:.1%} |"
        )
        if greedy:
            c = "ok" if row["greedy_constrained"] else "x"
            u = "ok" if row["greedy_unconstrained"] else "x"
            line += f" {c}/{u} |"
        print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--total", type=int, default=1_000)
    args = parser.parse_args(argv)
    metrics, rows, single_rows = run(args.total)
    write_results(
        "validity", {"metrics": metrics, "per_schema": rows, "single_schema": single_rows}
    )
    print_table("Shipped 5-gram (never saw these schemas):", rows, greedy=True)
    print_table("Single-schema 5-gram (trained on 120 documents of the schema):", single_rows,
                greedy=False)  # fmt: skip
    print()
    return report_and_check("validity", metrics, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
