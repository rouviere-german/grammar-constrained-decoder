"""Precomputed token-transition index over a minimal DFA, and the dead-end proof.

For every live DFA state and every vocabulary token, the token's bytes are walked through the DFA
(vectorized over all tokens with numpy, in chunks of states). A token is *allowed* in a state when
the walk ends in a live state, which is exact for multi-byte tokens such as ``}}`` or ``","``
that cross several grammar symbols: the walk checks every byte, not just the first. The index
stores, per state, the sorted allowed token ids and their successor states (CSR layout).

EOS is not a byte token: it is allowed exactly in accepting states.

Two derived tables make decoding safe:

- ``distance[s]``: fewest tokens needed to reach an accepting state from ``s`` (breadth-first
  search on the reversed token graph). A decoder with ``r`` tokens left only offers tokens whose
  successor has ``distance <= r - 1``, so it always finishes with a valid document within its
  budget.
- the **dead-end check** (:meth:`TokenIndex.check_no_dead_ends`): every state reachable from the
  start at token boundaries must have a finite distance, i.e. it is accepting or some allowed
  token leads towards acceptance. A failure is a compile-time error, never a runtime surprise.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from grammar_constrained_decoder.automata.dfa import DFA
from grammar_constrained_decoder.automata.limits import (
    CompileBudget,
    CompileLimitError,
)
from grammar_constrained_decoder.lm.tokenizer import ByteVocabulary

__all__ = ["DeadEndError", "TokenIndex", "build_token_index"]

IntArray = npt.NDArray[np.int32]
BoolArray = npt.NDArray[np.bool_]
_CHUNK_CELLS = 2_000_000
_MASK_CACHE = 4_096


class DeadEndError(CompileLimitError):
    """Some state reachable at a token boundary cannot reach acceptance through whole tokens."""

    def __init__(self, message: str) -> None:
        super().__init__("dead_end", message)


@dataclass(eq=False)
class TokenIndex:
    """Allowed tokens and successor states for every DFA state (CSR)."""

    offsets: npt.NDArray[np.int64]
    tokens: IntArray
    targets: IntArray
    accepting: BoolArray
    distance: IntArray
    reachable: IntArray
    start: int
    eos_id: int
    vocab_size: int
    _masks: OrderedDict[int, BoolArray] = field(default_factory=OrderedDict, repr=False)
    _max_next: IntArray | None = field(default=None, repr=False)
    # One index is shared by concurrent generations (the compiler's memory cache hands the same
    # object to every API worker thread), so the mask LRU is guarded.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def n_states(self) -> int:
        return int(self.accepting.shape[0])

    @property
    def n_entries(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def reachable_states(self) -> list[int]:
        return [int(state) for state in self.reachable]

    def allowed_tokens(self, state: int) -> IntArray:
        start, end = int(self.offsets[state]), int(self.offsets[state + 1])
        return self.tokens[start:end]

    def next_state(self, state: int, token: int) -> int:
        """Successor after ``token``, or -1 when the token is not allowed in ``state``."""
        start, end = int(self.offsets[state]), int(self.offsets[state + 1])
        slot = start + int(np.searchsorted(self.tokens[start:end], token))
        if slot < end and int(self.tokens[slot]) == token:
            return int(self.targets[slot])
        return -1

    def _max_next_distance(self) -> IntArray:
        cached = self._max_next
        if cached is None:
            best = np.full(self.n_states, -1, dtype=np.int32)
            if self.n_entries:
                owner = np.repeat(
                    np.arange(self.n_states, dtype=np.int32), np.diff(self.offsets).astype(np.int64)
                )
                np.maximum.at(best, owner, self.distance[self.targets])
            self._max_next = cached = best  # idempotent: racing threads compute the same array
        return cached

    def mask(self, state: int, remaining: int | None = None) -> BoolArray:
        """Boolean mask over the vocabulary for ``state``.

        With ``remaining`` tokens left (EOS not counted), only tokens that can still reach
        acceptance in time are allowed. The returned array must not be modified.
        """
        if remaining is not None and remaining - 1 < int(self._max_next_distance()[state]):
            start, end = int(self.offsets[state]), int(self.offsets[state + 1])
            targets = self.targets[start:end]
            keep = self.tokens[start:end][self.distance[targets] <= remaining - 1]
            limited = np.zeros(self.vocab_size, dtype=np.bool_)
            limited[keep] = True
            limited[self.eos_id] = bool(self.accepting[state])
            return limited
        with self._lock:
            cached = self._masks.get(state)
            if cached is not None:
                self._masks.move_to_end(state)
                return cached
        full = np.zeros(self.vocab_size, dtype=np.bool_)
        full[self.allowed_tokens(state)] = True
        full[self.eos_id] = bool(self.accepting[state])
        full.flags.writeable = False
        with self._lock:
            self._masks[state] = full
            self._masks.move_to_end(state)
            while len(self._masks) > _MASK_CACHE:
                self._masks.popitem(last=False)
        return full

    def check_no_dead_ends(self) -> None:
        """Raise :class:`DeadEndError` unless every token-reachable state can finish."""
        stuck = [int(s) for s in self.reachable if self.distance[s] < 0]
        if stuck:
            raise DeadEndError(
                f"{len(stuck)} state(s) reachable at a token boundary cannot reach an accepting "
                "state through vocabulary tokens"
            )

    def to_arrays(self) -> dict[str, npt.NDArray[Any]]:
        return {
            "index_offsets": self.offsets,
            "index_tokens": self.tokens,
            "index_targets": self.targets,
            "index_accepting": self.accepting,
            "index_distance": self.distance,
            "index_reachable": self.reachable,
            "index_meta": np.array([self.start, self.eos_id, self.vocab_size], dtype=np.int64),
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, npt.NDArray[Any]]) -> TokenIndex:
        meta = np.asarray(arrays["index_meta"], dtype=np.int64)
        index = cls(
            offsets=np.asarray(arrays["index_offsets"], dtype=np.int64),
            tokens=np.asarray(arrays["index_tokens"], dtype=np.int32),
            targets=np.asarray(arrays["index_targets"], dtype=np.int32),
            accepting=np.asarray(arrays["index_accepting"], dtype=np.bool_),
            distance=np.asarray(arrays["index_distance"], dtype=np.int32),
            reachable=np.asarray(arrays["index_reachable"], dtype=np.int32),
            start=int(meta[0]),
            eos_id=int(meta[1]),
            vocab_size=int(meta[2]),
        )
        index.validate()
        return index

    def validate(self) -> None:
        n, entries = self.n_states, self.n_entries
        ok = (
            self.offsets.shape == (n + 1,)
            and int(self.offsets[0]) == 0
            and int(self.offsets[-1]) == entries
            and bool(np.all(np.diff(self.offsets) >= 0))
            and self.targets.shape == (entries,)
            and self.distance.shape == (n,)
            and 0 <= self.start < n
            and 0 <= self.eos_id < self.vocab_size
            and (
                entries == 0
                or (int(self.tokens.min()) >= 0 and int(self.tokens.max()) < self.vocab_size)
            )
            and (entries == 0 or (int(self.targets.min()) >= 0 and int(self.targets.max()) < n))
            and (self.reachable.size == 0 or int(self.reachable.max()) < n)
        )
        if not ok:
            raise ValueError("invalid token index arrays")


def _token_matrix(vocab: ByteVocabulary) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    width = max(1, vocab.max_token_length)
    matrix = np.zeros((vocab.size, width), dtype=np.uint8)
    lengths = np.zeros(vocab.size, dtype=np.int64)
    for token, data in enumerate(vocab.token_bytes):
        matrix[token, : len(data)] = np.frombuffer(data, dtype=np.uint8)
        lengths[token] = len(data)
    return matrix, lengths


def _walk_chunks(
    dfa: DFA, vocab: ByteVocabulary, states: IntArray, budget: CompileBudget
) -> Iterator[tuple[IntArray, IntArray]]:
    """Yield ``(states, successors)`` blocks; successors are -1 for disallowed tokens."""
    matrix, lengths = _token_matrix(vocab)
    dead = dfa.dead
    table = dfa.table
    columns = [
        (np.flatnonzero(lengths > position), dfa.byte_class[matrix[lengths > position, position]])
        for position in range(matrix.shape[1])
    ]
    specials = np.array(sorted(vocab.special_ids), dtype=np.int64)
    chunk = max(1, _CHUNK_CELLS // max(1, vocab.size))
    for begin in range(0, states.size, chunk):
        budget.check_time()
        block = states[begin : begin + chunk]
        current = np.repeat(block[:, None], vocab.size, axis=1).astype(np.int32)
        for active, classes in columns:
            current[:, active] = table[current[:, active], classes[None, :]]
        if dead >= 0:
            current[current == dead] = -1
        if specials.size:
            current[:, specials] = -1
        yield block, current


def _distances(n: int, sources: IntArray, targets: IntArray, accepting: BoolArray) -> IntArray:
    order = np.argsort(targets, kind="stable")
    reverse_sources = sources[order]
    reverse_offsets = np.searchsorted(targets[order], np.arange(n + 1))
    distance = np.full(n, -1, dtype=np.int32)
    queue: deque[int] = deque()
    for state in np.flatnonzero(accepting).tolist():
        distance[state] = 0
        queue.append(state)
    while queue:
        state = queue.popleft()
        preds = reverse_sources[reverse_offsets[state] : reverse_offsets[state + 1]]
        fresh = np.unique(preds[distance[preds] < 0])
        if fresh.size:
            distance[fresh] = distance[state] + 1
            queue.extend(fresh.tolist())
    return distance


def _reachable(start: int, offsets: npt.NDArray[np.int64], targets: IntArray, n: int) -> IntArray:
    seen = np.zeros(n, dtype=np.bool_)
    seen[start] = True
    stack = [start]
    while stack:
        state = stack.pop()
        successors = np.unique(targets[offsets[state] : offsets[state + 1]])
        fresh = successors[~seen[successors]]
        if fresh.size:
            seen[fresh] = True
            stack.extend(fresh.tolist())
    return np.flatnonzero(seen).astype(np.int32)


def build_token_index(dfa: DFA, vocab: ByteVocabulary, budget: CompileBudget) -> TokenIndex:
    """Build the index for every live state, then check reachability and dead ends."""
    n = dfa.n_states
    live = np.array([s for s in range(n) if s != dfa.dead], dtype=np.int32)
    counts = np.zeros(n, dtype=np.int64)
    token_parts: list[IntArray] = []
    target_parts: list[IntArray] = []
    source_parts: list[IntArray] = []
    entries = 0
    for block, successors in _walk_chunks(dfa, vocab, live, budget):
        rows, ids = np.nonzero(successors >= 0)
        entries += int(ids.size)
        if entries > budget.limits.max_index_entries:
            raise CompileLimitError(
                "index_entries",
                f"token index needs more than {budget.limits.max_index_entries} entries",
            )
        counts[block] = np.bincount(rows, minlength=block.size)
        token_parts.append(ids.astype(np.int32))
        target_parts.append(successors[rows, ids].astype(np.int32))
        source_parts.append(block[rows])
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    tokens = np.concatenate(token_parts) if token_parts else np.zeros(0, np.int32)
    targets = np.concatenate(target_parts) if target_parts else np.zeros(0, np.int32)
    sources = np.concatenate(source_parts) if source_parts else np.zeros(0, np.int32)
    budget.check_time()
    accepting = dfa.accepting.copy()
    distance = _distances(n, sources, targets, accepting)
    reachable = (
        _reachable(dfa.start, offsets, targets, n)
        if not dfa.is_empty
        else np.zeros(0, dtype=np.int32)
    )
    index = TokenIndex(
        offsets=offsets,
        tokens=tokens,
        targets=targets,
        accepting=accepting,
        distance=distance,
        reachable=reachable,
        start=dfa.start,
        eos_id=vocab.eos_id,
        vocab_size=vocab.size,
    )
    index.check_no_dead_ends()
    return index
