"""Byte-level regular expression AST with hash-consing and normalizing smart constructors.

This is the language the compiler produces from a JSON Schema and the input to both automaton
pipelines: Thompson NFA -> subset construction -> Hopcroft minimization (``nfa.py``/``dfa.py``),
and the independent Brzozowski-derivative oracle (``oracle.py``).

Nodes are immutable and interned: constructing a structurally equal node returns the existing
object, so equality is identity and hashing is O(1). The smart constructors apply the similarity
rules that keep the set of Brzozowski derivatives finite (Brzozowski 1964; Owens, Reppy and Turon,
"Regular-expression derivatives re-examined", JFP 2009):

- concatenation is flattened, ``eps`` is its unit and ``empty`` annihilates it;
- alternation is flattened, deduplicated and sorted (associative, commutative, idempotent), drops
  ``empty`` and merges all byte-set members into a single byte set;
- intersection is flattened, deduplicated and sorted; ``empty`` annihilates it;
- ``star`` and bounded repetition are normalized (``r**`` = ``r*``; a nullable body loses its
  lower bound).

Only constructors are exported; never instantiate the node classes directly.
"""

from __future__ import annotations

import itertools
import weakref
from collections.abc import Iterator

from grammar_constrained_decoder.automata.charset import ALL_BYTES, mask_ranges

__all__ = [
    "EMPTY",
    "EPS",
    "Alt",
    "And",
    "ByteSet",
    "Cat",
    "Empty",
    "Epsilon",
    "Regex",
    "Repeat",
    "Star",
    "alt",
    "byteset",
    "cat",
    "inter",
    "iter_nodes",
    "literal",
    "masks_of",
    "opt",
    "plus",
    "rep",
    "size",
    "star",
    "to_pattern",
]

_ids = itertools.count(1)
_table: weakref.WeakValueDictionary[tuple[object, ...], Regex] = weakref.WeakValueDictionary()


class Regex:
    """Base class of interned regex nodes."""

    __slots__ = ("__weakref__", "_hash", "nid", "nullable")

    nid: int
    nullable: bool
    _hash: int

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return self is other

    def __lt__(self, other: Regex) -> bool:
        return self.nid < other.nid

    def children(self) -> tuple[Regex, ...]:
        return ()

    def __repr__(self) -> str:
        return f"Regex({to_pattern(self)!r})"


def _intern[R: Regex](
    cls: type[R], key: tuple[object, ...], nullable: bool, init: dict[str, object]
) -> R:
    existing = _table.get(key)
    if existing is not None:
        if not isinstance(existing, cls):  # pragma: no cover - the key encodes the class
            raise TypeError("regex intern table is inconsistent")
        return existing
    node = object.__new__(cls)
    for name, value in init.items():
        object.__setattr__(node, name, value)
    object.__setattr__(node, "nid", next(_ids))
    object.__setattr__(node, "nullable", nullable)
    object.__setattr__(node, "_hash", hash(key))
    _table[key] = node
    return node


class Empty(Regex):
    """The empty language (matches nothing)."""

    __slots__ = ()


class Epsilon(Regex):
    """The language containing only the empty string."""

    __slots__ = ()


class ByteSet(Regex):
    """One byte from a non-empty set."""

    __slots__ = ("mask",)
    mask: int


class Cat(Regex):
    """Concatenation of two or more nodes."""

    __slots__ = ("items",)
    items: tuple[Regex, ...]

    def children(self) -> tuple[Regex, ...]:
        return self.items


class Alt(Regex):
    """Union of two or more nodes (sorted by node id, at most one ByteSet)."""

    __slots__ = ("items",)
    items: tuple[Regex, ...]

    def children(self) -> tuple[Regex, ...]:
        return self.items


class And(Regex):
    """Intersection of two or more nodes (sorted by node id)."""

    __slots__ = ("items",)
    items: tuple[Regex, ...]

    def children(self) -> tuple[Regex, ...]:
        return self.items


class Star(Regex):
    """Kleene star of a node that is neither empty nor epsilon."""

    __slots__ = ("item",)
    item: Regex

    def children(self) -> tuple[Regex, ...]:
        return (self.item,)


class Repeat(Regex):
    """``item{lo,hi}`` with ``hi is None`` meaning unbounded; ``lo`` is 0 when item is nullable."""

    __slots__ = ("hi", "item", "lo")
    item: Regex
    lo: int
    hi: int | None

    def children(self) -> tuple[Regex, ...]:
        return (self.item,)


EMPTY: Empty = _intern(Empty, ("empty",), False, {})
EPS: Epsilon = _intern(Epsilon, ("eps",), True, {})


def byteset(mask: int) -> Regex:
    """One byte from ``mask``; the empty mask is the empty language."""
    if mask < 0 or mask > ALL_BYTES:
        raise ValueError("byte mask out of range")
    if mask == 0:
        return EMPTY
    return _intern(ByteSet, ("bytes", mask), False, {"mask": mask})


def literal(data: bytes | str) -> Regex:
    """The single string ``data`` (text is encoded as UTF-8)."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return cat(*(byteset(1 << value) for value in raw))


def cat(*items: Regex) -> Regex:
    flat: list[Regex] = []
    for item in items:
        if item is EMPTY:
            return EMPTY
        if item is EPS:
            continue
        if isinstance(item, Cat):
            flat.extend(item.items)
        else:
            flat.append(item)
    if not flat:
        return EPS
    if len(flat) == 1:
        return flat[0]
    key: tuple[object, ...] = ("cat", *(node.nid for node in flat))
    return _intern(Cat, key, all(node.nullable for node in flat), {"items": tuple(flat)})


def alt(*items: Regex) -> Regex:
    members: set[Regex] = set()
    mask = 0
    pending = list(items)
    while pending:
        item = pending.pop()
        if item is EMPTY:
            continue
        if isinstance(item, Alt):
            pending.extend(item.items)
        elif isinstance(item, ByteSet):
            mask |= item.mask
        else:
            members.add(item)
    if mask:
        members.add(byteset(mask))
    if EPS in members and any(node.nullable for node in members if node is not EPS):
        members.discard(EPS)
    if not members:
        return EMPTY
    if len(members) == 1:
        return next(iter(members))
    ordered = tuple(sorted(members))
    key: tuple[object, ...] = ("alt", *(node.nid for node in ordered))
    return _intern(Alt, key, any(node.nullable for node in ordered), {"items": ordered})


def inter(*items: Regex) -> Regex:
    """Intersection. All-byte-set operands collapse to one byte set."""
    members: set[Regex] = set()
    pending = list(items)
    while pending:
        item = pending.pop()
        if item is EMPTY:
            return EMPTY
        if isinstance(item, And):
            pending.extend(item.items)
        else:
            members.add(item)
    if not members:
        raise ValueError("intersection of zero operands is undefined")
    if EPS in members:
        return EPS if all(node.nullable for node in members) else EMPTY
    byte_sets = [node for node in members if isinstance(node, ByteSet)]
    if len(byte_sets) == len(members):
        mask = ALL_BYTES
        for node in byte_sets:
            mask &= node.mask
        return byteset(mask)
    if len(members) == 1:
        return next(iter(members))
    ordered = tuple(sorted(members))
    key: tuple[object, ...] = ("and", *(node.nid for node in ordered))
    return _intern(And, key, all(node.nullable for node in ordered), {"items": ordered})


def star(item: Regex) -> Regex:
    if item is EMPTY or item is EPS:
        return EPS
    if isinstance(item, Star):
        return item
    if isinstance(item, Repeat) and item.lo == 0 and item.hi is None:
        return star(item.item)
    if isinstance(item, Alt) and EPS in item.items:
        return star(alt(*(node for node in item.items if node is not EPS)))
    return _intern(Star, ("star", item.nid), True, {"item": item})


def rep(item: Regex, lo: int, hi: int | None) -> Regex:
    """Bounded or unbounded repetition ``item{lo,hi}``."""
    if lo < 0 or (hi is not None and hi < lo):
        raise ValueError(f"invalid repetition bounds {{{lo},{hi}}}")
    if hi == 0 or item is EPS:
        return EPS
    if item is EMPTY:
        return EPS if lo == 0 else EMPTY
    if item.nullable:
        lo = 0
    if lo == 1 and hi == 1:
        return item
    if lo == 0 and hi is None:
        return star(item)
    if lo == 0 and hi == 1:
        return alt(EPS, item)
    key: tuple[object, ...] = ("rep", item.nid, lo, hi)
    return _intern(Repeat, key, lo == 0, {"item": item, "lo": lo, "hi": hi})


def opt(item: Regex) -> Regex:
    return alt(EPS, item)


def plus(item: Regex) -> Regex:
    return rep(item, 1, None)


def iter_nodes(root: Regex) -> Iterator[Regex]:
    """Every distinct node reachable from ``root`` (each once)."""
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.nid in seen:
            continue
        seen.add(node.nid)
        yield node
        stack.extend(node.children())


def masks_of(root: Regex) -> set[int]:
    """All byte-set masks appearing in ``root``."""
    return {node.mask for node in iter_nodes(root) if isinstance(node, ByteSet)}


def size(root: Regex) -> int:
    """Number of distinct nodes (the DAG size, not the expanded tree size)."""
    return sum(1 for _ in iter_nodes(root))


def _byteset_pattern(mask: int) -> str:
    if mask == ALL_BYTES:
        return "."
    ranges = mask_ranges(mask)
    if len(ranges) == 1 and ranges[0][0] == ranges[0][1]:
        return _byte_repr(ranges[0][0])
    parts = [
        _byte_repr(lo) if lo == hi else f"{_byte_repr(lo)}-{_byte_repr(hi)}" for lo, hi in ranges
    ]
    return "[" + "".join(parts) + "]"


def _byte_repr(value: int) -> str:
    char = chr(value)
    if 0x21 <= value <= 0x7E and char not in "\\[]().*+?{}|^$-&":
        return char
    return f"\\x{value:02x}"


def to_pattern(root: Regex) -> str:
    """Human-readable rendering for debugging and error messages (not re-parseable)."""
    return _render(root, 0)


def _render(node: Regex, precedence: int) -> str:  # noqa: PLR0911 - one branch per node type
    if node is EMPTY:
        return "∅"
    if node is EPS:
        return "ε"
    if isinstance(node, ByteSet):
        return _byteset_pattern(node.mask)
    if isinstance(node, Cat):
        text = "".join(_render(item, 2) for item in node.items)
        return f"({text})" if precedence > 2 else text
    if isinstance(node, Alt):
        text = "|".join(_render(item, 1) for item in node.items)
        return f"({text})" if precedence > 0 else text
    if isinstance(node, And):
        text = "&".join(_render(item, 1) for item in node.items)
        return f"({text})" if precedence > 0 else text
    if isinstance(node, Star):
        return _render(node.item, 3) + "*"
    if isinstance(node, Repeat):
        bound = f"{node.lo}," if node.hi is None else f"{node.lo},{node.hi}"
        return _render(node.item, 3) + "{" + bound + "}"
    raise TypeError(f"unknown regex node {type(node).__name__}")
