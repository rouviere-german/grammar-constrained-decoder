"""Brzozowski-derivative oracle, independent of the NFA/DFA pipeline.

The derivative of a language L by a byte b is ``{w : bw in L}`` (Brzozowski, "Derivatives of
regular expressions", JACM 1964). For a regex it is computed syntactically, and with the
similarity rules applied by the smart constructors in ``regex.py`` every regex has finitely many
distinct derivatives. The oracle uses that to answer three questions without touching
``nfa.py``/``dfa.py``:

- **membership**: ``w`` is in L(r) iff the derivative of r by w is nullable;
- **emptiness**: L(d) is non-empty iff some derivative of d is nullable. With intersection in the
  language this is not syntactic, so the oracle explores the whole derivative graph of the root
  (Brzozowski's DFA construction) and computes liveness by reverse reachability;
- **exactness of a compiled grammar**: :meth:`DerivativeOracle.check_equivalence` walks the
  product of the compiled DFA and the derivative graph. Each visited pair must agree on
  acceptance and on liveness; because every pair reachable from the start pair is visited, a clean
  walk is a bisimulation, i.e. a proof that the DFA accepts exactly L(root).
  :meth:`DerivativeOracle.check_token_index` then recomputes, for every state reachable at token
  boundaries, the set of tokens whose bytes leave the derivative non-empty and compares it with
  the precomputed token index, including the successor state of every allowed token.

The oracle computes its own byte partition from the masks in the regex (all masks of every
derivative are unions or intersections of those, so the partition is valid for all of them).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

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
    alt,
    cat,
    inter,
    masks_of,
    rep,
)

if TYPE_CHECKING:
    from grammar_constrained_decoder.automata.dfa import DFA

__all__ = [
    "DerivativeOracle",
    "EquivalenceReport",
    "OracleLimitError",
    "TokenIndexView",
    "TokenReport",
]


class OracleLimitError(RuntimeError):
    """The derivative graph grew past the oracle's state cap."""


class TokenIndexView(Protocol):
    """What the oracle needs from a token index (structural, so the oracle imports nothing)."""

    @property
    def reachable_states(self) -> Sequence[int]: ...

    @property
    def accepting(self) -> Sequence[bool] | npt.NDArray[np.bool_]: ...

    def allowed_tokens(self, state: int) -> Sequence[int] | npt.NDArray[np.int32]: ...

    def next_state(self, state: int, token: int) -> int: ...


@dataclass
class EquivalenceReport:
    pairs_checked: int = 0
    mismatches: list[str] = field(default_factory=list)
    representative: dict[int, Regex] = field(default_factory=dict)
    pair_set: set[tuple[int, int]] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not self.mismatches


@dataclass
class TokenReport:
    states_checked: int = 0
    tokens_checked: int = 0
    allowed_total: int = 0
    mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches


@dataclass
class _TrieNode:
    children: dict[int, _TrieNode] = field(default_factory=dict)
    tokens: list[int] = field(default_factory=list)


def _signature_partition(masks: set[int]) -> tuple[list[int], list[int]]:
    """Group bytes by the set of masks containing them (a different method from dfa.py)."""
    ordered = sorted(masks)
    groups: dict[tuple[bool, ...], int] = {}
    class_of = [0] * 256
    for byte in range(256):
        signature = tuple(bool(mask >> byte & 1) for mask in ordered)
        class_of[byte] = groups.setdefault(signature, len(groups))
    representative = [0] * len(groups)
    for byte in range(255, -1, -1):
        representative[class_of[byte]] = byte
    return class_of, representative


class DerivativeOracle:
    """Explores the derivative graph of ``root`` and answers language questions exactly."""

    def __init__(self, root: Regex, *, max_states: int = 100_000) -> None:
        self.root = root
        self.max_states = max_states
        self.class_of, self.representative = _signature_partition(masks_of(root))
        self._memo: dict[tuple[int, int], Regex] = {}
        self._keep: list[Regex] = []
        self.states: dict[int, Regex] = {}
        self.edges: dict[int, list[int]] = {}
        self.live: set[int] = set()
        self._explore()

    # -- derivatives ---------------------------------------------------------------------------

    def derive(self, node: Regex, byte: int) -> Regex:
        return self._derive_class(node, self.class_of[byte])

    def _derive_class(self, node: Regex, cls: int) -> Regex:
        key = (node.nid, cls)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        result = self._compute(node, cls)
        self._memo[key] = result
        self._keep.append(node)
        return result

    def _compute(self, node: Regex, cls: int) -> Regex:  # noqa: PLR0911 - one rule per node type
        if node is EMPTY or node is EPS:
            return EMPTY
        if isinstance(node, ByteSet):
            return EPS if node.mask >> self.representative[cls] & 1 else EMPTY
        if isinstance(node, Cat):
            head, tail = node.items[0], cat(*node.items[1:])
            derived = cat(self._derive_class(head, cls), tail)
            if head.nullable:
                derived = alt(derived, self._derive_class(tail, cls))
            return derived
        if isinstance(node, Alt):
            return alt(*(self._derive_class(item, cls) for item in node.items))
        if isinstance(node, And):
            return inter(*(self._derive_class(item, cls) for item in node.items))
        if isinstance(node, Star):
            return cat(self._derive_class(node.item, cls), node)
        if isinstance(node, Repeat):
            remaining = rep(
                node.item, max(node.lo - 1, 0), None if node.hi is None else node.hi - 1
            )
            return cat(self._derive_class(node.item, cls), remaining)
        raise TypeError(f"unknown regex node {type(node).__name__}")

    def derive_bytes(self, node: Regex, data: bytes) -> Regex:
        for byte in data:
            node = self.derive(node, byte)
            if node is EMPTY:
                break
        return node

    def accepts(self, data: bytes) -> bool:
        return self.derive_bytes(self.root, data).nullable

    def nonempty(self, node: Regex) -> bool:
        """True when L(node) is non-empty. ``node`` must be a derivative of the root."""
        if node.nid not in self.states:
            raise KeyError("node is not a derivative of the oracle's root")
        return node.nid in self.live

    # -- exploration -----------------------------------------------------------------------------

    def _explore(self) -> None:
        n_classes = len(self.representative)
        self.states[self.root.nid] = self.root
        queue: deque[Regex] = deque([self.root])
        while queue:
            node = queue.popleft()
            successors = []
            for cls in range(n_classes):
                derived = self._derive_class(node, cls)
                successors.append(derived.nid)
                if derived.nid not in self.states:
                    if len(self.states) >= self.max_states:
                        raise OracleLimitError(
                            f"derivative graph exceeds {self.max_states} distinct derivatives"
                        )
                    self.states[derived.nid] = derived
                    queue.append(derived)
            self.edges[node.nid] = successors
        reverse: dict[int, list[int]] = {nid: [] for nid in self.states}
        for source, targets in self.edges.items():
            for target in set(targets):
                reverse[target].append(source)
        pending = [nid for nid, node in self.states.items() if node.nullable]
        self.live = set(pending)
        while pending:
            nid = pending.pop()
            for source in reverse[nid]:
                if source not in self.live:
                    self.live.add(source)
                    pending.append(source)

    @property
    def n_derivatives(self) -> int:
        return len(self.states)

    # -- checks against compiled artifacts -------------------------------------------------------

    def check_equivalence(self, dfa: DFA, *, max_mismatches: int = 5) -> EquivalenceReport:
        """Bisimulation walk over (DFA state, derivative) pairs from the start pair."""
        report = EquivalenceReport()
        joint: dict[tuple[int, int], int] = {}
        for byte in range(256):
            joint.setdefault((int(dfa.byte_class[byte]), self.class_of[byte]), byte)
        start = (dfa.start, self.root.nid)
        parent: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {start: None}
        queue: deque[tuple[int, int]] = deque([start])

        def witness(pair: tuple[int, int]) -> str:
            path: list[int] = []
            link = parent[pair]
            while link is not None:
                previous, byte = link
                path.append(byte)
                link = parent[previous]
            return bytes(reversed(path)).hex()

        while queue:
            pair = queue.popleft()
            state, nid = pair
            node = self.states[nid]
            report.pairs_checked += 1
            report.representative.setdefault(state, node)
            report.pair_set.add(pair)
            if bool(dfa.accepting[state]) != node.nullable:
                report.mismatches.append(f"acceptance differs after bytes {witness(pair)!r}")
            if (state == dfa.dead) != (nid not in self.live):
                report.mismatches.append(f"liveness differs after bytes {witness(pair)!r}")
            if len(report.mismatches) >= max_mismatches:
                break
            if state == dfa.dead:
                continue
            for (dfa_cls, oracle_cls), byte in joint.items():
                target = (int(dfa.table[state, dfa_cls]), self.edges[nid][oracle_cls])
                if target not in parent:
                    parent[target] = (pair, byte)
                    queue.append(target)
        return report

    def check_token_index(
        self,
        index: TokenIndexView,
        token_bytes: Sequence[bytes],
        equivalence: EquivalenceReport,
        *,
        eos_id: int,
        max_mismatches: int = 5,
    ) -> TokenReport:
        """Recompute allowed tokens and successors at every token-reachable DFA state."""
        trie = _TrieNode()
        for token_id, data in enumerate(token_bytes):
            if not data:
                continue
            node = trie
            for byte in data:
                node = node.children.setdefault(byte, _TrieNode())
            node.tokens.append(token_id)
        report = TokenReport()
        n_tokens = sum(1 for data in token_bytes if data)
        for state in index.reachable_states:
            regex = equivalence.representative.get(state)
            if regex is None:
                report.mismatches.append(f"state {state} is not reachable in the oracle walk")
                continue
            expected: dict[int, Regex] = {}
            self._collect(trie, regex, expected)
            actual = list(index.allowed_tokens(state))
            report.states_checked += 1
            report.tokens_checked += n_tokens
            report.allowed_total += len(actual)
            problems = self._compare(
                state, regex, expected, actual, index=index, equivalence=equivalence, eos_id=eos_id
            )
            report.mismatches.extend(problems)
            if len(report.mismatches) >= max_mismatches:
                break
        return report

    def _collect(self, trie: _TrieNode, regex: Regex, out: dict[int, Regex]) -> None:
        stack: list[tuple[_TrieNode, Regex]] = [(trie, regex)]
        while stack:
            node, current = stack.pop()
            for byte, child in node.children.items():
                derived = self.derive(current, byte)
                if derived.nid not in self.live:
                    continue
                for token_id in child.tokens:
                    out[token_id] = derived
                if child.children:
                    stack.append((child, derived))

    def _compare(
        self,
        state: int,
        regex: Regex,
        expected: dict[int, Regex],
        actual: Sequence[int],
        *,
        index: TokenIndexView,
        equivalence: EquivalenceReport,
        eos_id: int,
    ) -> list[str]:
        problems = []
        actual_set = {token for token in actual if token != eos_id}
        if actual_set != set(expected):
            extra = sorted(actual_set - set(expected))[:5]
            missing = sorted(set(expected) - actual_set)[:5]
            problems.append(f"state {state}: extra tokens {extra}, missing tokens {missing}")
        if eos_id in actual:
            problems.append(f"state {state}: EOS is listed as a byte token")
        if bool(index.accepting[state]) != regex.nullable:
            problems.append(f"state {state}: EOS permission (acceptance) differs from nullability")
        for token_id in sorted(actual_set & set(expected)):
            successor = index.next_state(state, token_id)
            if (successor, expected[token_id].nid) not in equivalence.pair_set:
                problems.append(f"state {state}: successor of token {token_id} differs")
                break
        return problems
