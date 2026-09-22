"""The derivative oracle on realistic schemas: bundled schemas (patterns, recursion, thousands of
DFA states) and the tiny distortion schemas, compiled against the trained 1,024-token vocabulary.

``bench/bench_exactness.py`` checks all 50 (about a minute); this test covers the 20 tiny schemas,
the nine bundled schemas whose minimal DFA has at most about 500 states (most of them with
patterns) and the recursive ``category_tree`` (2,532 states), so the default suite stays fast.
"""

from __future__ import annotations

import pytest

from conftest import Trained
from grammar_constrained_decoder.automata.oracle import DerivativeOracle
from grammar_constrained_decoder.decoding.compiler import GrammarCompiler
from grammar_constrained_decoder.training.corpus import (
    BundledSchema,
    load_bundled_schemas,
    load_distortion_schemas,
)

pytestmark = pytest.mark.integration

BUNDLED = (
    "weather_reading", "sensor_batch", "http_request", "flight", "file_metadata", "feature_flag",
    "air_quality", "library_loan", "geo_point", "category_tree",
)  # fmt: skip
SCHEMAS: list[BundledSchema] = [
    *(item for item in load_bundled_schemas() if item.id in BUNDLED),
    *load_distortion_schemas(),
]


def test_selection_covers_recursion_and_patterns() -> None:
    assert len(SCHEMAS) == len(BUNDLED) + 20
    assert any("$ref" in str(item.schema) for item in SCHEMAS)
    assert any("pattern" in str(item.schema) for item in SCHEMAS)


@pytest.mark.parametrize("item", SCHEMAS, ids=[item.id for item in SCHEMAS])
def test_masks_of_bundled_and_tiny_schemas_are_exact(
    trained: Trained, compiler: GrammarCompiler, item: BundledSchema
) -> None:
    vocab = trained.model.tokenizer.vocab
    grammar = compiler.compile(item.schema)
    oracle = DerivativeOracle(grammar.regex)
    equivalence = oracle.check_equivalence(grammar.dfa)
    assert equivalence.ok, equivalence.mismatches
    report = oracle.check_token_index(
        grammar.index, vocab.token_bytes, equivalence, eos_id=vocab.eos_id
    )
    assert report.ok, report.mismatches
    assert report.states_checked == len(grammar.index.reachable_states)
