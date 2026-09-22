"""Mask exactness, completeness, dead ends and DFA-vs-re agreement.

Checks (fixed seeds via hypothesis ``derandomize``, CPU only):

1. **Mask exactness** on ``--schemas`` (default 2,000) hypothesis-generated schemas in the subset,
   compiled against the trained 1,024-token vocabulary. For each schema the Brzozowski-derivative
   oracle walks the product of the compiled DFA and the derivative graph (a bisimulation, i.e. a
   language-equivalence proof), then recomputes, at every DFA state reachable at a token boundary,
   the set of tokens whose bytes leave the derivative non-empty, plus each token's successor, and
   compares them with the precomputed token index. Mismatches must be 0.
2. **Completeness**: for each schema, up to 3 instances drawn by ``hypothesis-jsonschema`` (one
   derandomized run per schema); the canonical serialization of every instance must be accepted
   by the compiled DFA.
3. **Dead ends**: every bundled and tiny schema compiles with the dead-end proof enabled.
4. **DFA vs re**: ``--patterns`` generated patterns in the ASCII-compatible subset, 40 random
   strings each, full-match and search semantics compared with Python ``re`` (``re.ASCII``) used
   as a trusted oracle on patterns this script generated itself.

Usage: uv run python bench/bench_exactness.py [--check] [--schemas 2000] [--patterns 500]
"""

from __future__ import annotations

import argparse
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st
from hypothesis.errors import HypothesisException
from hypothesis_jsonschema import from_schema

from _common import report_and_check, train_model, write_results
from grammar_constrained_decoder.automata.dfa import compile_regex
from grammar_constrained_decoder.automata.limits import CompileError
from grammar_constrained_decoder.automata.oracle import DerivativeOracle
from grammar_constrained_decoder.automata.pattern import compile_pattern
from grammar_constrained_decoder.decoding.compiler import GrammarCompiler
from grammar_constrained_decoder.lm.tokenizer import ByteVocabulary
from grammar_constrained_decoder.schema.canonical import canonical_dumps
from grammar_constrained_decoder.testing.strategies import (
    PATTERN_TEST_ALPHABET,
    ecma_patterns,
    json_schemas,
)
from grammar_constrained_decoder.training.corpus import (
    load_bundled_schemas,
    load_distortion_schemas,
)

_RUN = settings(
    derandomize=True,
    database=None,
    deadline=None,
    phases=[Phase.generate],
    suppress_health_check=list(HealthCheck),
)


def collect_schemas(n_schemas: int) -> list[dict[str, Any]]:
    """The first ``n_schemas`` schemas hypothesis generates (derandomized, so always the same)."""
    collected: list[dict[str, Any]] = []

    @settings(_RUN, max_examples=n_schemas)
    @given(json_schemas())
    def collect(schema: dict[str, Any]) -> None:
        collected.append(schema)

    collect()
    return collected[:n_schemas]


def instances_for(schema: dict[str, Any], count: int = 3) -> list[Any] | None:
    """Up to ``count`` instances from hypothesis-jsonschema, or None if it cannot generate any."""
    found: list[Any] = []

    @settings(_RUN, max_examples=count)
    @given(from_schema(schema))
    def draw(value: Any) -> None:
        found.append(value)

    try:
        draw()
    except HypothesisException:
        return None
    return found


def exactness(vocab: ByteVocabulary, n_schemas: int) -> dict[str, float]:
    compiler = GrammarCompiler(vocab, memory_entries=0)
    stats: dict[str, float] = {
        "schemas": 0,
        "rejected_empty_language": 0,
        "dfa_states": 0,
        "token_reachable_states": 0,
        "token_decisions": 0,
        "allowed_tokens": 0,
        "equivalence_pairs": 0,
        "mask_mismatches": 0,
        "equivalence_mismatches": 0,
        "instances": 0,
        "completeness_failures": 0,
        "instance_generation_skipped": 0,
    }
    failures: list[str] = []
    for schema in collect_schemas(n_schemas):
        try:
            compiled = compiler.compile(schema)
        except CompileError as exc:
            if exc.reason != "empty_language":
                failures.append(f"unexpected rejection {exc.reason}: {schema}")
            stats["rejected_empty_language"] += 1
            continue
        stats["schemas"] += 1
        stats["dfa_states"] += compiled.dfa.n_states
        oracle = DerivativeOracle(compiled.regex)
        equivalence = oracle.check_equivalence(compiled.dfa)
        tokens = oracle.check_token_index(
            compiled.index, vocab.token_bytes, equivalence, eos_id=vocab.eos_id
        )
        stats["equivalence_pairs"] += equivalence.pairs_checked
        stats["token_reachable_states"] += tokens.states_checked
        stats["token_decisions"] += tokens.tokens_checked
        stats["allowed_tokens"] += tokens.allowed_total
        if not equivalence.ok:
            stats["equivalence_mismatches"] += 1
            failures.append(f"equivalence {equivalence.mismatches[:2]}: {schema}")
        if not tokens.ok:
            stats["mask_mismatches"] += len(tokens.mismatches)
            failures.append(f"mask {tokens.mismatches[:2]}: {schema}")
        instances = instances_for(schema)
        if instances is None:
            stats["instance_generation_skipped"] += 1
            continue
        for instance in instances:
            stats["instances"] += 1
            text = canonical_dumps(compiled.ir, instance)
            if not compiled.dfa.accepts(text.encode("utf-8")):
                stats["completeness_failures"] += 1
                failures.append(f"completeness {text!r}: {schema}")
    for line in failures[:10]:
        print("FAILURE:", line[:300])
    return stats


def dead_ends(vocab: ByteVocabulary) -> dict[str, float]:
    compiler = GrammarCompiler(vocab)
    schemas = [*load_bundled_schemas(), *load_distortion_schemas()]
    failures = 0
    for item in schemas:
        try:
            compiler.compile(item.schema)
        except CompileError:
            failures += 1
    return {"dead_end_schemas_checked": float(len(schemas)), "dead_end_failures": float(failures)}


def pattern_agreement(n_patterns: int) -> dict[str, float]:
    comparisons = disagreements = 0

    @settings(_RUN, max_examples=n_patterns)
    @given(
        ecma_patterns(),
        st.lists(st.text(PATTERN_TEST_ALPHABET, max_size=10), min_size=40, max_size=40),
    )
    def check(body: str, words: list[str]) -> None:
        nonlocal comparisons, disagreements
        full = compile_regex(compile_pattern(f"^(?:{body})$"))
        search = compile_regex(compile_pattern(body))
        oracle = re.compile(body, re.ASCII)
        for word in words:
            data = word.encode("utf-8")
            comparisons += 2
            disagreements += full.accepts(data) != (oracle.fullmatch(word) is not None)
            disagreements += search.accepts(data) != (oracle.search(word) is not None)

    check()
    return {"re_comparisons": float(comparisons), "re_disagreements": float(disagreements)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--schemas", type=int, default=2_000)
    parser.add_argument("--patterns", type=int, default=500)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        model = train_model(Path(tmp))
    vocab = model.tokenizer.vocab
    mark = time.perf_counter()
    metrics = exactness(vocab, args.schemas)
    metrics["exactness_seconds"] = round(time.perf_counter() - mark, 1)
    metrics.update(dead_ends(vocab))
    metrics.update(pattern_agreement(args.patterns))
    metrics["mean_dfa_states"] = round(metrics["dfa_states"] / max(metrics["schemas"], 1), 1)
    metrics["total_seconds"] = round(time.perf_counter() - started, 1)
    metrics = {name: float(np.round(value, 1)) for name, value in metrics.items()}
    write_results("exactness", {"metrics": metrics})
    return report_and_check("exactness", metrics, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
