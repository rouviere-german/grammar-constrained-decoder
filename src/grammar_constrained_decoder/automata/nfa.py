"""Thompson construction: byte-level regex AST -> epsilon-NFA with one start and one accept state.

Construction follows McNaughton-Yamada-Thompson (Thompson, "Regular expression search
algorithm", CACM 1968): every node becomes a fragment with a fresh start and a fresh accept state
joined by epsilon edges. Bounded repetition ``r{m,n}`` is expanded into ``m`` mandatory copies
followed by ``n - m`` optional copies that may each exit early, so the expansion is linear.

Intersection has no Thompson fragment. An ``And`` node is compiled into a minimal DFA by the
``sub_dfa`` callback (product construction in ``dfa.py``) and that DFA is embedded as a fragment:
DFA states become NFA states and accepting states get an epsilon edge to the fragment's accept.
The same embedding is used for large repeated bodies (see ``_Builder``). Every created state is
charged to the :class:`CompileBudget`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

from grammar_constrained_decoder.automata.limits import CompileBudget
from grammar_constrained_decoder.automata.regex import (
    EMPTY,
    EPS,
    Alt,
    And,
    ByteSet,
    Cat,
    Regex,
    Repeat,
    Star,
)

if TYPE_CHECKING:
    from grammar_constrained_decoder.automata.dfa import DFA

__all__ = ["NFA", "thompson"]


@dataclass
class NFA:
    """Epsilon-NFA over bytes. ``edges[s]`` holds ``(byte_mask, target)`` pairs."""

    budget: CompileBudget
    eps: list[list[int]] = field(default_factory=list)
    edges: list[list[tuple[int, int]]] = field(default_factory=list)
    start: int = 0
    accept: int = 0

    def new_state(self) -> int:
        self.budget.charge_nfa()
        self.eps.append([])
        self.edges.append([])
        return len(self.eps) - 1

    @property
    def n_states(self) -> int:
        return len(self.eps)

    def masks(self) -> set[int]:
        return {mask for out in self.edges for mask, _ in out}


_EMBED_THRESHOLD = 12


def _thompson_size(node: Regex, memo: dict[int, int]) -> int:
    """Number of NFA states Thompson's construction would create for ``node`` (capped)."""
    cached = memo.get(node.nid)
    if cached is not None:
        return cached
    if isinstance(node, Cat):
        total = sum(_thompson_size(item, memo) for item in node.items)
    elif isinstance(node, Alt):
        total = 2 + sum(_thompson_size(item, memo) for item in node.items)
    elif isinstance(node, Star):
        total = 2 + _thompson_size(node.item, memo)
    elif isinstance(node, Repeat):
        copies = node.lo + (node.hi - node.lo if node.hi is not None else 1)
        total = 2 + copies * _thompson_size(node.item, memo)
    else:
        total = 2
    total = min(total, 1 << 30)
    memo[node.nid] = total
    return total


class _Builder:
    """Thompson construction with one optimization: the body of a repetition whose Thompson
    fragment would be large is compiled once into its minimal DFA, and every copy embeds that
    DFA. A minimal DFA accepts the same language, so the result is unchanged, but a JSON string
    character (UTF-8 sequences plus escapes, ~150 Thompson states) shrinks to ~15 states per copy.
    """

    def __init__(self, nfa: NFA, sub_dfa: Callable[[Regex], DFA]) -> None:
        self.nfa = nfa
        self.sub_dfa = sub_dfa
        self._sizes: dict[int, int] = {}
        self._dfas: dict[int, DFA] = {}

    def body(self, node: Regex) -> tuple[int, int]:
        """Fragment for a repeated body: an embedded minimal DFA when that is smaller."""
        if _thompson_size(node, self._sizes) <= _EMBED_THRESHOLD:
            return self.fragment(node)
        dfa = self._dfas.get(node.nid)
        if dfa is None:
            dfa = self._dfas[node.nid] = self.sub_dfa(node)
        return self._embed(dfa)

    def fragment(self, node: Regex) -> tuple[int, int]:  # noqa: PLR0911 - one branch per node type
        nfa = self.nfa
        if node is EMPTY:
            return nfa.new_state(), nfa.new_state()
        if node is EPS:
            start, accept = nfa.new_state(), nfa.new_state()
            nfa.eps[start].append(accept)
            return start, accept
        if isinstance(node, ByteSet):
            start, accept = nfa.new_state(), nfa.new_state()
            nfa.edges[start].append((node.mask, accept))
            return start, accept
        if isinstance(node, Cat):
            return self._concat([self.fragment(item) for item in node.items])
        if isinstance(node, Alt):
            start, accept = nfa.new_state(), nfa.new_state()
            for item in node.items:
                inner_start, inner_accept = self.fragment(item)
                nfa.eps[start].append(inner_start)
                nfa.eps[inner_accept].append(accept)
            return start, accept
        if isinstance(node, Star):
            return self._star(node.item)
        if isinstance(node, Repeat):
            return self._repeat(node)
        if isinstance(node, And):
            return self._embed(self.sub_dfa(node))
        raise TypeError(f"unknown regex node {type(node).__name__}")

    def _concat(self, parts: list[tuple[int, int]]) -> tuple[int, int]:
        for (_, left_accept), (right_start, _) in pairwise(parts):
            self.nfa.eps[left_accept].append(right_start)
        return parts[0][0], parts[-1][1]

    def _star(self, item: Regex) -> tuple[int, int]:
        nfa = self.nfa
        start, accept = nfa.new_state(), nfa.new_state()
        inner_start, inner_accept = self.body(item)
        nfa.eps[start].extend((inner_start, accept))
        nfa.eps[inner_accept].extend((inner_start, accept))
        return start, accept

    def _repeat(self, node: Repeat) -> tuple[int, int]:
        nfa = self.nfa
        parts = [self.body(node.item) for _ in range(node.lo)]
        if node.hi is None:
            parts.append(self._star(node.item))
            return self._concat(parts)
        start, accept = nfa.new_state(), nfa.new_state()
        cursor = start
        if parts:
            nfa.eps[start].append(parts[0][0])
            cursor = self._concat(parts)[1]
        for _ in range(node.hi - node.lo):
            nfa.eps[cursor].append(accept)
            inner_start, inner_accept = self.body(node.item)
            nfa.eps[cursor].append(inner_start)
            cursor = inner_accept
        nfa.eps[cursor].append(accept)
        return start, accept

    def _embed(self, dfa: DFA) -> tuple[int, int]:
        nfa = self.nfa
        accept = nfa.new_state()
        if dfa.is_empty:
            return nfa.new_state(), accept
        class_masks = dfa.class_masks()
        mapping: dict[int, int] = {}
        for state in range(dfa.n_states):
            if state != dfa.dead:
                mapping[state] = nfa.new_state()
        for state, node_state in mapping.items():
            merged: dict[int, int] = {}
            for cls, target in enumerate(dfa.table[state].tolist()):
                if target != dfa.dead:
                    merged[target] = merged.get(target, 0) | class_masks[cls]
            for target, mask in sorted(merged.items()):
                nfa.edges[node_state].append((mask, mapping[target]))
            if dfa.accepting[state]:
                nfa.eps[node_state].append(accept)
        return mapping[dfa.start], accept


def thompson(root: Regex, budget: CompileBudget, sub_dfa: Callable[[Regex], DFA]) -> NFA:
    """Build the epsilon-NFA for ``root``. ``sub_dfa`` compiles intersection nodes."""
    nfa = NFA(budget=budget)
    nfa.start, nfa.accept = _Builder(nfa, sub_dfa).fragment(root)
    return nfa
