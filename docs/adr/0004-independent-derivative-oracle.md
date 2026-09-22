# 4. Prove each mask exact with an independent Brzozowski-derivative oracle

- Status: accepted
- Date: 2026-09-22

## Context

A constrained decoder is only as trustworthy as its masks. A mask that is too permissive lets
invalid output through; one that is too strict silently removes valid outputs (and distorts the
distribution further). Testing against the same automaton code that built the masks proves
nothing, and comparing with Python's `re` only covers the regex subset `re` shares with us, not
intersections, byte-level UTF-8 encodings or token boundaries.

## Decision

Implement a second, structurally different construction of the same language: Brzozowski
derivatives over the same regex AST, with the similarity rules (associativity, commutativity and
idempotence of union and intersection) applied by the smart constructors so that the set of
derivatives is finite. The oracle computes its own byte partition and never imports the NFA/DFA
modules. It checks a compiled grammar in two steps:

1. a breadth-first walk over pairs (DFA state, derivative) from the start pair; every pair must
   agree on acceptance and on liveness. Visiting every reachable pair without disagreement is a
   bisimulation, i.e. a proof that the DFA accepts exactly the regex language;
2. at every DFA state reachable at a token boundary, it recomputes the allowed tokens by walking
   a byte trie of the vocabulary through derivatives, pruning non-live derivatives, and compares
   the set and each token's successor with the precomputed index.

## Consequences

- `bench/bench_exactness.py` runs the oracle on 2,000 generated schemas against the real
  1,024-token vocabulary; the acceptance criterion is zero mismatches, enforced in CI.
- Emptiness with intersection is not syntactic, so the oracle explores the whole derivative graph
  of the root before answering liveness; it has its own state cap.
- The oracle validates the automaton pipeline and the index, not the schema-to-regex lowering.
  Lowering is covered separately: soundness (random strings of the language validate with
  `jsonschema`) and completeness (canonical serializations of `hypothesis-jsonschema` instances are
  accepted).
