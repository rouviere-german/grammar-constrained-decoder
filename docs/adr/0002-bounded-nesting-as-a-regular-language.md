# 2. Bounded nesting inlined into one regular language (no pushdown stack)

- Status: accepted
- Date: 2026-09-22

## Context

JSON is context-free: nested objects and arrays need a stack to match brackets. Grammar-constrained
decoders that accept arbitrary JSON Schemas therefore usually run a pushdown automaton or an
incremental parser (Earley, LR) at decoding time, and compute token masks by simulating the parser
on every candidate token, often with caching heuristics. That is flexible, but:

- the per-token mask cost depends on parser state and is hard to bound;
- the set of reachable parser configurations is unbounded, so masks cannot be precomputed and
  exactness can only be tested, not enumerated;
- "no dead ends" (every reachable state can still finish with the model's tokens) is a property
  of an infinite configuration space.

Real response schemas are shallow: the 30 bundled schemas nest at most 3 containers deep, and
recursion (trees, comments) is rare and useful only to a small depth.

## Decision

Cap container nesting at `max_depth` (default 6) and lower the schema to a single regular
expression. Non-recursive nesting beyond the cap is a compile error. Recursive `$ref`s are
unrolled up to the cap; the alternative that would exceed it becomes the empty language, so the
compiled language is exactly the set of instances whose nesting fits. Reference cycles that pass
through no container are cut on re-entry, which yields the least fixed point.

The regex is compiled by Thompson construction, subset construction and Hopcroft minimization into
one canonical minimal DFA, and every per-token decision is precomputed over its finite state set.

## Consequences

- Masks are a table lookup; overhead per token is a few microseconds (see README Results) and does
  not depend on nesting.
- Exactness is checkable exhaustively: the derivative oracle visits every reachable state
  (ADR 0004), and the dead-end check is a finite graph search.
- The DFA grows with depth, string lengths and optional members; resource limits (states, time)
  reject schemas whose automaton would be too large instead of degrading at decode time.
- Deeply recursive data (for example an arbitrarily deep comment thread) cannot be expressed.
  Callers raise `compile_max_depth` within the state limits or restructure the schema. This is a
  documented non-goal.
- Canonical member order (declaration order) keeps object languages linear in the number of
  properties; permutations of optional members would multiply states.
