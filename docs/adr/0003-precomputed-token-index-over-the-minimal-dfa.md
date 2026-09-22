# 3. Precompute a token-transition index per DFA state, cached by schema and tokenizer

- Status: accepted
- Date: 2026-09-22

## Context

At every decoding step the allowed tokens are those whose bytes, fed to the automaton from the
current state, end in a live state. There are three common ways to answer that:

1. walk the vocabulary (or a byte trie of it) through the automaton at every step;
2. precompute, for each automaton state, the allowed tokens and their successor states;
3. approximate with a character-level check of the first byte only.

Option 3 is wrong for multi-byte tokens such as `}}`, `","` or `":{"` that cross several grammar
symbols: the first byte may be fine while a later byte closes one bracket too many. Option 1 costs
a trie walk per generated token (hundreds of microseconds in Python). Option 2 costs memory
proportional to states times allowed tokens, but reduces a step to a slice lookup.

## Decision

Build the index once per (schema, tokenizer, limits) with a vectorized walk: all tokens are padded
into a byte matrix and advanced through the DFA transition table column by column for blocks of
states (numpy fancy indexing). Store it in CSR form: per state, the sorted allowed token ids and
their successor states. Derive two tables from it: the fewest tokens to acceptance per state
(reverse breadth-first search), used for budget-aware masks, and the set of states reachable at
token boundaries, used for the dead-end proof. EOS is not a byte token; it is allowed exactly in
accepting states.

Cache compiled grammars in memory (LRU) and on disk as `.npz` (`allow_pickle=False`) keyed by
`sha256(compiler version, canonical schema hash, vocabulary hash, limits)`, with a checksum over the
arrays; any mismatch is a miss.

## Consequences

- Per-token masking is a cached boolean vector plus `np.where`; the p95 is a few microseconds and
  much smaller than one language-model step (README Results).
- Exactness is defined per token and per state, which is what the oracle recomputes independently.
- Index size is bounded by `max_index_entries`; the largest bundled schema (`calendar_event`,
  3,139 DFA states) needs 335,346 entries, about 2.7 MB of int32 token/successor pairs
  (`bench/bench_performance.py`).
- A tokenizer change invalidates every cached index, by construction of the key.
- The disk cache detects corruption but not a writer who recomputes the checksum; the cache
  directory must be writable only by the service (threat model T4).
