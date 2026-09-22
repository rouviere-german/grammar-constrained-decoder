"""Deterministic automata: subset construction, Hopcroft minimization and product intersection.

Pipeline (``compile_regex``): regex -> Thompson NFA -> subset construction over byte equivalence
classes -> Hopcroft minimization -> canonical form.

- **Byte classes.** Bytes that every NFA edge treats alike are merged into one class, so the
  subset construction iterates over tens of classes instead of 256 bytes.
- **Subset construction** (Rabin and Scott 1959) explores only reachable subsets and charges each
  new DFA state to the budget, which bounds the classic exponential blow-up.
- **Hopcroft minimization** (Hopcroft 1971) refines the accepting/non-accepting partition with the
  "process the smaller half" worklist, O(k n log n) for n states and k classes.
- **Canonical form.** After minimization, classes with identical columns are merged and states
  are renumbered in breadth-first order from the start state, visiting classes in order of their
  smallest byte, with the dead state (if any) last. Two minimal DFAs for the same language are
  therefore identical arrays, which the tests use to compare languages.

Matching a string is one table lookup per byte with no backtracking, so match time is linear in
the input whatever the pattern (no ReDoS).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any

import numpy as np
import numpy.typing as npt

from grammar_constrained_decoder.automata.charset import ALL_BYTES, iter_bytes
from grammar_constrained_decoder.automata.limits import CompileBudget
from grammar_constrained_decoder.automata.nfa import NFA, thompson
from grammar_constrained_decoder.automata.regex import And, Regex

__all__ = [
    "DFA",
    "byte_partition",
    "compile_regex",
    "determinize",
    "intersect",
    "minimize",
]

IntArray = npt.NDArray[np.int32]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, eq=False)
class DFA:
    """A complete DFA over byte classes.

    ``table[state, byte_class[b]]`` is the successor of ``state`` on byte ``b``. ``dead`` is the
    absorbing non-accepting state, or -1 when the language has no dead state (every prefix can be
    extended to a match).
    """

    byte_class: IntArray
    table: IntArray
    accepting: BoolArray
    start: int
    dead: int

    @property
    def n_states(self) -> int:
        return int(self.table.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.table.shape[1])

    @property
    def is_empty(self) -> bool:
        """True when the language is empty (the start state is dead)."""
        return self.start == self.dead

    @property
    def n_live_states(self) -> int:
        return self.n_states - (1 if self.dead >= 0 else 0)

    def class_masks(self) -> list[int]:
        masks = [0] * self.n_classes
        for byte, cls in enumerate(self.byte_class.tolist()):
            masks[cls] |= 1 << byte
        return masks

    def step(self, state: int, byte: int) -> int:
        return int(self.table[state, self.byte_class[byte]])

    def run(self, data: bytes, state: int | None = None) -> int:
        """State reached after reading ``data`` (stops early at the dead state)."""
        current = self.start if state is None else state
        table = self.table
        classes = self.byte_class
        for byte in data:
            current = int(table[current, classes[byte]])
            if current == self.dead:
                break
        return current

    def accepts(self, data: bytes) -> bool:
        return bool(self.accepting[self.run(data)])

    def distances_to_accept(self) -> IntArray:
        """Fewest bytes from each state to an accepting state (-1 when unreachable)."""
        n = self.n_states
        reverse: list[list[int]] = [[] for _ in range(n)]
        for state in range(n):
            for target in set(self.table[state].tolist()):
                reverse[target].append(state)
        distance = np.full(n, -1, dtype=np.int32)
        queue: deque[int] = deque()
        for state in np.flatnonzero(self.accepting).tolist():
            distance[state] = 0
            queue.append(state)
        while queue:
            state = queue.popleft()
            for previous in reverse[state]:
                if distance[previous] < 0:
                    distance[previous] = distance[state] + 1
                    queue.append(previous)
        return distance

    def signature(self) -> tuple[bytes, bytes, bytes, int, int]:
        """Canonical fingerprint: equal for two canonical DFAs iff they accept the same language."""
        return (
            self.byte_class.astype(np.int32).tobytes(),
            self.table.astype(np.int32).tobytes(),
            self.accepting.astype(np.bool_).tobytes(),
            self.start,
            self.dead,
        )

    def to_arrays(self, prefix: str = "dfa_") -> dict[str, npt.NDArray[np.int32] | BoolArray]:
        return {
            f"{prefix}byte_class": self.byte_class,
            f"{prefix}table": self.table,
            f"{prefix}accepting": self.accepting,
            f"{prefix}meta": np.array([self.start, self.dead], dtype=np.int32),
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, npt.NDArray[Any]], prefix: str = "dfa_") -> DFA:
        byte_class = np.asarray(arrays[f"{prefix}byte_class"], dtype=np.int32)
        table = np.asarray(arrays[f"{prefix}table"], dtype=np.int32)
        accepting = np.asarray(arrays[f"{prefix}accepting"], dtype=np.bool_)
        meta = np.asarray(arrays[f"{prefix}meta"], dtype=np.int32)
        dfa = cls(byte_class, table, accepting, int(meta[0]), int(meta[1]))
        dfa.validate()
        return dfa

    def validate(self) -> None:
        """Structural checks for DFAs read from disk (shapes, ranges, absorbing dead state)."""
        n, k = self.table.shape if self.table.ndim == 2 else (0, 0)
        problems = []
        if self.byte_class.shape != (256,) or n == 0 or k == 0:
            problems.append("bad shapes")
        elif self.byte_class.min() < 0 or self.byte_class.max() >= k:
            problems.append("byte class out of range")
        elif self.table.min() < 0 or self.table.max() >= n:
            problems.append("transition out of range")
        elif self.accepting.shape != (n,):
            problems.append("accepting shape mismatch")
        elif not 0 <= self.start < n or not -1 <= self.dead < n:
            problems.append("start or dead state out of range")
        elif self.dead >= 0 and (
            self.accepting[self.dead] or not np.all(self.table[self.dead] == self.dead)
        ):
            problems.append("dead state is not absorbing and rejecting")
        if problems:
            raise ValueError(f"invalid DFA: {problems[0]}")


def byte_partition(masks: Iterable[int]) -> tuple[IntArray, list[int]]:
    """Coarsest partition of the 256 bytes that refines every mask.

    Returns the byte -> class map and the class masks, with classes numbered by smallest byte.
    """
    blocks = [ALL_BYTES]
    for mask in sorted(set(masks)):
        if mask in (0, ALL_BYTES):
            continue
        refined: list[int] = []
        for block in blocks:
            inside, outside = block & mask, block & ~mask
            if inside and outside:
                refined.extend((inside, outside))
            else:
                refined.append(block)
        blocks = refined
    blocks.sort(key=lambda block: (block & -block).bit_length())
    byte_class = np.zeros(256, dtype=np.int32)
    for index, block in enumerate(blocks):
        for byte in iter_bytes(block):
            byte_class[byte] = index
    return byte_class, blocks


def _closure(nfa: NFA, state: int) -> frozenset[int]:
    seen = {state}
    stack = [state]
    while stack:
        current = stack.pop()
        for target in nfa.eps[current]:
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return frozenset(seen)


def determinize(nfa: NFA, budget: CompileBudget) -> DFA:
    """Subset construction over byte classes. The result is complete but not minimal."""
    byte_class, class_masks = byte_partition(nfa.masks())
    k = len(class_masks)
    classes_of_mask: dict[int, list[int]] = {}
    for mask in nfa.masks():
        classes_of_mask[mask] = [c for c, cm in enumerate(class_masks) if cm & mask]
    edge_classes: list[list[tuple[list[int], int]]] = [
        [(classes_of_mask[mask], target) for mask, target in out] for out in nfa.edges
    ]
    closures: dict[int, frozenset[int]] = {}

    def closure(state: int) -> frozenset[int]:
        cached = closures.get(state)
        if cached is None:
            cached = closures[state] = _closure(nfa, state)
            budget.tick()
        return cached

    unions: dict[frozenset[int], frozenset[int]] = {}

    def closure_of(targets: set[int]) -> frozenset[int]:
        key = frozenset(targets)
        cached = unions.get(key)
        if cached is None:
            cached = unions[key] = frozenset().union(*(closure(t) for t in key))
        return cached

    start_set = closure(nfa.start)
    index: dict[frozenset[int], int] = {start_set: 0}
    subsets: list[frozenset[int]] = [start_set]
    rows: list[list[int]] = []
    budget.charge_dfa()
    cursor = 0
    while cursor < len(subsets):
        current = subsets[cursor]
        cursor += 1
        moves: dict[int, set[int]] = {}
        for state in current:
            for classes, target in edge_classes[state]:
                for cls in classes:
                    moves.setdefault(cls, set()).add(target)
        row = []
        for cls in range(k):
            targets = moves.get(cls)
            subset: frozenset[int] = closure_of(targets) if targets else frozenset()
            state_id = index.get(subset)
            if state_id is None:
                budget.charge_dfa()
                state_id = index[subset] = len(subsets)
                subsets.append(subset)
            row.append(state_id)
        rows.append(row)
    table = np.asarray(rows, dtype=np.int32).reshape(len(subsets), k)
    accepting = np.array([nfa.accept in subset for subset in subsets], dtype=np.bool_)
    dead = index.get(frozenset(), -1)
    return DFA(byte_class, table, accepting, 0, dead)


def _hopcroft_blocks(dfa: DFA, budget: CompileBudget) -> list[int]:  # noqa: PLR0912
    """Return the block id of every state in the coarsest language-preserving partition."""
    n, k = dfa.n_states, dfa.n_classes
    table = dfa.table.tolist()
    inverse: list[list[list[int]]] = [[[] for _ in range(n)] for _ in range(k)]
    for state in range(n):
        for cls in range(k):
            inverse[cls][table[state][cls]].append(state)
    accepting = dfa.accepting.tolist()
    final = {s for s in range(n) if accepting[s]}
    others = set(range(n)) - final
    blocks: list[set[int]] = [block for block in (final, others) if block]
    block_of = [0] * n
    for index, block in enumerate(blocks):
        for state in block:
            block_of[state] = index
    worklist: deque[tuple[int, int]] = deque()
    pending: set[tuple[int, int]] = set()
    if len(blocks) == 2:
        smaller = 0 if len(blocks[0]) <= len(blocks[1]) else 1
        for cls in range(k):
            worklist.append((smaller, cls))
            pending.add((smaller, cls))
    while worklist:
        splitter, cls = worklist.popleft()
        pending.discard((splitter, cls))
        budget.tick()
        predecessors: dict[int, list[int]] = {}
        for target in list(blocks[splitter]):
            for state in inverse[cls][target]:
                predecessors.setdefault(block_of[state], []).append(state)
        for block_id, members in predecessors.items():
            block = blocks[block_id]
            if len(members) == len(block):
                continue
            moved = set(members)
            block -= moved
            new_id = len(blocks)
            blocks.append(moved)
            for state in moved:
                block_of[state] = new_id
            for split_cls in range(k):
                if (block_id, split_cls) in pending:
                    worklist.append((new_id, split_cls))
                    pending.add((new_id, split_cls))
                else:
                    chosen = new_id if len(moved) <= len(block) else block_id
                    worklist.append((chosen, split_cls))
                    pending.add((chosen, split_cls))
    return block_of


def _canonical(byte_class: IntArray, table: IntArray, accepting: BoolArray, start: int) -> DFA:
    """Merge identical class columns and renumber states breadth-first; dead state last."""
    n, k = table.shape
    columns: dict[bytes, int] = {}
    class_map = np.zeros(k, dtype=np.int32)
    order_by_byte = [int(c) for c in byte_class.tolist()]
    seen_classes: list[int] = []
    for cls in order_by_byte:
        if cls not in seen_classes:
            seen_classes.append(cls)
    for cls in seen_classes:
        key = np.ascontiguousarray(table[:, cls]).tobytes()
        if key not in columns:
            columns[key] = len(columns)
        class_map[cls] = columns[key]
    new_k = len(columns)
    representative = [0] * new_k
    for cls in reversed(seen_classes):
        representative[int(class_map[cls])] = cls
    merged = table[:, representative]

    dead_states = [s for s in range(n) if not accepting[s] and bool(np.all(merged[s] == s))]
    dead = dead_states[0] if dead_states else -1
    order: list[int] = []
    number = {start: 0}
    queue: deque[int] = deque([start])
    if start == dead:
        order.append(start)
    else:
        while queue:
            state = queue.popleft()
            order.append(state)
            for target in merged[state].tolist():
                if target != dead and target not in number:
                    number[target] = len(number)
                    queue.append(target)
    if dead >= 0 and dead not in number:
        number[dead] = len(number)
        order.append(dead)
    new_table = np.empty((len(order), new_k), dtype=np.int32)
    for old in order:
        new_table[number[old]] = [number[t] for t in merged[old].tolist()]
    new_accepting = np.array([bool(accepting[old]) for old in order], dtype=np.bool_)
    new_byte_class = class_map[byte_class]
    return DFA(
        new_byte_class.astype(np.int32),
        new_table,
        new_accepting,
        0,
        number[dead] if dead >= 0 else -1,
    )


def minimize(dfa: DFA, budget: CompileBudget) -> DFA:
    """Hopcroft minimization followed by canonicalization."""
    block_of = _hopcroft_blocks(dfa, budget)
    n_blocks = max(block_of) + 1
    table = np.zeros((n_blocks, dfa.n_classes), dtype=np.int32)
    accepting = np.zeros(n_blocks, dtype=np.bool_)
    for state in range(dfa.n_states):
        block = block_of[state]
        table[block] = [block_of[t] for t in dfa.table[state].tolist()]
        accepting[block] = dfa.accepting[state]
    return _canonical(dfa.byte_class, table, accepting, block_of[dfa.start])


def intersect(left: DFA, right: DFA, budget: CompileBudget) -> DFA:
    """Product construction; the result is minimized and canonical."""
    joint: dict[tuple[int, int], int] = {}
    byte_class = np.zeros(256, dtype=np.int32)
    for byte in range(256):
        pair = (int(left.byte_class[byte]), int(right.byte_class[byte]))
        byte_class[byte] = joint.setdefault(pair, len(joint))
    pairs = sorted(joint, key=joint.__getitem__)
    dead_pair = (-1, -1)
    initial = (left.start, right.start)
    if left.start == left.dead or right.start == right.dead:
        initial = dead_pair
    index: dict[tuple[int, int], int] = {initial: 0}
    states: list[tuple[int, int]] = [initial]
    rows: list[list[int]] = []
    budget.charge_dfa()
    cursor = 0
    while cursor < len(states):
        a, b = states[cursor]
        cursor += 1
        if (a, b) == dead_pair:
            rows.append([index[dead_pair]] * len(pairs))
            continue
        row = []
        for ca, cb in pairs:
            target = (int(left.table[a, ca]), int(right.table[b, cb]))
            if target[0] == left.dead or target[1] == right.dead:
                target = dead_pair
            state_id = index.get(target)
            if state_id is None:
                budget.charge_dfa()
                state_id = index[target] = len(states)
                states.append(target)
            row.append(state_id)
        rows.append(row)
    table = np.asarray(rows, dtype=np.int32).reshape(len(states), len(pairs))
    accepting = np.array(
        [
            a >= 0 and b >= 0 and bool(left.accepting[a]) and bool(right.accepting[b])
            for a, b in states
        ],
        dtype=np.bool_,
    )
    product = DFA(byte_class, table, accepting, 0, index.get(dead_pair, -1))
    return minimize(product, budget)


def _compile(root: Regex, budget: CompileBudget) -> DFA:
    def sub_dfa(node: Regex) -> DFA:
        if not isinstance(node, And):
            return _compile(node, budget)
        parts: Sequence[DFA] = [_compile(item, budget) for item in node.items]
        return reduce(lambda acc, item: intersect(acc, item, budget), parts[1:], parts[0])

    nfa = thompson(root, budget, sub_dfa)
    return minimize(determinize(nfa, budget), budget)


def compile_regex(root: Regex, budget: CompileBudget | None = None) -> DFA:
    """Compile a regex to its canonical minimal DFA, charging ``budget`` along the way."""
    return _compile(root, budget if budget is not None else CompileBudget())
