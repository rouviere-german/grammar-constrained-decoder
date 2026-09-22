# 6. Compact canonical output and budget-aware masks

- Status: accepted
- Date: 2026-09-22

## Context

Two decisions shape what "the language" of a schema is and whether decoding always ends well.

**Output form.** JSON allows whitespace anywhere between tokens, any member order, several escape
spellings (`A` or `A`) and several number spellings (`1e2`, `100`, `100.0`). Admitting all of
them multiplies DFA states (member permutations are factorial) and lets a weak model wander
through whitespace. None of it changes the parsed value.

**Termination.** A mask that only checks "can this prefix still be completed?" can let the model
use its whole `max_tokens` budget inside an unbounded string and stop without closing the
document: the output is then truncated and invalid even though every token was allowed.

## Decision

- The language is compact canonical JSON: no insignificant whitespace, members in declaration
  order, the canonical escapes of `json.dumps(ensure_ascii=False)`, integers without leading zeros,
  numbers in fixed-point notation (never exponents, never `-0`). The parsed values still cover
  every instance of the schema whose nesting fits the depth cap; `canonical_dumps` gives the one
  serialization per instance that the completeness test checks.
- Masks are budget-aware: with `r` tokens left the decoder only offers tokens whose successor
  state can reach acceptance in at most `r - 1` more tokens, and EOS only in accepting states.
  Requests whose `max_tokens` is below the schema's minimum are rejected before decoding.

## Consequences

- Every constrained generation ends with EOS in an accepting state within `max_tokens`; the
  validity benchmark measures 100% across 1,000 held-out generations.
- The model cannot emit pretty-printed JSON or reorder members; consumers get compact JSON.
- Budget-aware masking is one more constraint on the local distribution and adds distortion near
  the end of the budget; the distortion study uses a budget large enough not to bind.
- Numbers may carry trailing fractional zeros (`1.50`), which are valid JSON and in range; only
  the canonical form is guaranteed to be accepted, not guaranteed to be produced.
