"""Random instances of a schema (for the synthetic corpus) and random strings of a DFA language.

``InstanceSampler`` draws plausible values: property-name hints pick realistic strings (names,
cities, short phrases), numbers are drawn inside their bounds, optional members appear with a
fixed probability, and recursion depth is steered towards leaves near the nesting limit. Every
candidate string is checked against the node's constraints; when a hint does not fit (length or
``pattern``), the value is drawn by a random walk over the minimal DFA of the string's language,
so every sampled instance is valid by construction. The corpus data contract re-validates each one
with ``jsonschema``.

``sample_dfa`` walks a DFA byte by byte: at each state it stops with some probability when
accepting, otherwise picks a random byte among those leading to live states, and near the length
cap follows shortest paths to acceptance. It is used for pattern strings and by tests that check
soundness (every string of the compiled language parses and validates).
"""

from __future__ import annotations

import math
from decimal import Decimal
from functools import lru_cache
from typing import Any

import numpy as np

from grammar_constrained_decoder.automata.charset import iter_bytes
from grammar_constrained_decoder.automata.dfa import DFA, compile_regex
from grammar_constrained_decoder.automata.pattern import ANY_CHAR, compile_pattern, utf8_class
from grammar_constrained_decoder.automata.regex import EMPTY, inter, rep
from grammar_constrained_decoder.schema.canonical import matches
from grammar_constrained_decoder.schema.ir import (
    ROOT_REF,
    AnyOf,
    Array,
    Boolean,
    Enum,
    Integer,
    Node,
    Null,
    Number,
    Object,
    Ref,
    SchemaIR,
    String,
)

__all__ = ["InstanceSampler", "sample_dfa", "string_dfa"]

FIRST_NAMES = (
    "Ada",
    "Alan",
    "Grace",
    "Linus",
    "Margaret",
    "Dennis",
    "Barbara",
    "Edsger",
    "Frances",
    "Ken",
    "Radia",
    "Tim",
    "Hedy",
    "Claude",
    "Katherine",
    "John",
    "Mary",
    "Niklaus",
    "Sophie",
    "Yuki",
)
LAST_NAMES = (
    "Lovelace",
    "Turing",
    "Hopper",
    "Torvalds",
    "Hamilton",
    "Ritchie",
    "Liskov",
    "Dijkstra",
    "Allen",
    "Thompson",
    "Perlman",
    "Berners",
    "Lamarr",
    "Shannon",
    "Johnson",
    "Backus",
    "Keller",
    "Wirth",
    "Wilson",
    "Tanaka",
)
CITIES = (
    "Berlin",
    "Paris",
    "Rome",
    "Madrid",
    "Amsterdam",
    "Warsaw",
    "Lisbon",
    "Vienna",
    "Prague",
    "Oslo",
    "Dublin",
    "Munich",
    "Lyon",
    "Milan",
    "Seville",
    "Utrecht",
    "Krakow",
    "Porto",
)
COMPANIES = ("Acme", "Globex", "Initech", "Umbrella", "Hooli", "Vandelay", "Stark", "Wayne")
WORDS = (
    "quick",
    "report",
    "update",
    "system",
    "server",
    "network",
    "garden",
    "coffee",
    "river",
    "yellow",
    "silent",
    "bright",
    "window",
    "market",
    "signal",
    "winter",
    "summer",
    "planet",
    "engine",
    "record",
    "button",
    "travel",
    "orange",
    "simple",
    "pocket",
    "ticket",
    "letter",
    "memory",
    "vector",
    "stream",
    "backup",
    "folder",
    "invoice",
    "budget",
    "school",
    "museum",
)
_NAME_KEYS = ("name", "player", "user", "assignee", "author", "attendee", "customer", "member")
_CITY_KEYS = ("city", "destination", "location", "hotel")
_PHRASE_KEYS = (
    "title",
    "summary",
    "text",
    "message",
    "notes",
    "description",
    "reference",
    "label",
    "item",
    "skill",
    "achievement",
    "tag",
)


@lru_cache(maxsize=512)
def string_dfa(min_length: int, max_length: int | None, pattern: str | None) -> DFA:
    """DFA over raw UTF-8 for strings with the given length bounds and pattern."""
    if max_length is not None and min_length > max_length:
        return compile_regex(EMPTY)
    body = rep(utf8_class(ANY_CHAR), min_length, max_length)
    if pattern is not None:
        body = inter(body, compile_pattern(pattern))
    return compile_regex(body)


def sample_dfa(
    dfa: DFA,
    rng: np.random.Generator,
    *,
    soft_limit: int = 24,
    stop_probability: float = 0.3,
    printable_bias: bool = True,
) -> bytes | None:
    """A random member of L(dfa), or None when the language is empty."""
    if dfa.is_empty:
        return None
    distance = dfa.distances_to_accept()
    masks = dfa.class_masks()
    out = bytearray()
    state = dfa.start
    while True:
        if dfa.accepting[state] and (len(out) >= soft_limit or rng.random() < stop_probability):
            return bytes(out)
        options = [
            (cls, int(target))
            for cls, target in enumerate(dfa.table[state].tolist())
            if target != dfa.dead and distance[target] >= 0
        ]
        if len(out) >= soft_limit:
            best = min(int(distance[target]) for _, target in options)
            options = [(cls, target) for cls, target in options if distance[target] == best]
        if not options:
            return bytes(out)
        cls, target = options[int(rng.integers(len(options)))]
        candidates = list(iter_bytes(masks[cls]))
        if printable_bias:
            printable = [b for b in candidates if 0x21 <= b <= 0x7E]
            candidates = printable or candidates
        out.append(candidates[int(rng.integers(len(candidates)))])
        state = target


class InstanceSampler:
    """Draws random valid instances of a parsed schema."""

    def __init__(self, ir: SchemaIR, rng: np.random.Generator, *, max_depth: int = 6) -> None:
        self.ir = ir
        self.rng = rng
        self.max_depth = max_depth

    def sample(self) -> Any:
        return self.node(self.ir.root, 0, "")

    def pick(self, options: tuple[str, ...]) -> str:
        return options[int(self.rng.integers(len(options)))]

    def node(self, node: Node, depth: int, key: str) -> Any:  # noqa: PLR0911 - one branch per type
        if isinstance(node, Ref):
            target = self.ir.root if node.name == ROOT_REF else self.ir.definitions[node.name]
            return self.node(target, depth, key)
        if isinstance(node, Null):
            return None
        if isinstance(node, Boolean):
            return bool(self.rng.random() < 0.5)
        if isinstance(node, Integer):
            return self.integer(node)
        if isinstance(node, Number):
            return self.number(node)
        if isinstance(node, String):
            return self.string(node, key)
        if isinstance(node, Enum):
            return node.values[int(self.rng.integers(len(node.values)))]
        if isinstance(node, AnyOf):
            order = self.rng.permutation(len(node.branches)).tolist()
            for index in order:
                value = self.node(node.branches[index], depth, key)
                if matches(SchemaIR(node.branches[index], self.ir.definitions, ""), value):
                    return value
            return self.node(node.branches[order[0]], depth, key)
        if isinstance(node, Array):
            return self.array(node, depth, key)
        return self.obj(node, depth)

    def integer(self, node: Integer) -> int:
        low = (
            node.minimum
            if node.minimum is not None
            else (node.maximum - 1000 if node.maximum is not None else 0)
        )
        high = node.maximum if node.maximum is not None else low + 1000
        if low > high:
            raise ValueError("empty integer range")
        value = int(self.rng.integers(low, high + 1))
        if self.rng.random() < 0.3 and high - low > 20:
            rounded = value - value % 10
            value = rounded if low <= rounded <= high else value
        return value

    def number(self, node: Number) -> float:
        low = float(node.minimum) if node.minimum is not None else -1000.0
        high = float(node.maximum) if node.maximum is not None else low + 1000.0
        for _ in range(64):
            value = round(float(self.rng.uniform(low, high)), int(self.rng.integers(0, 3)))
            if node.maximum is None and node.minimum is None:
                value = round(float(self.rng.uniform(-1000, 1000)), 2)
            number = Decimal(repr(value))
            low_ok = node.minimum is None or (
                number > node.minimum if node.exclusive_minimum else number >= node.minimum
            )
            high_ok = node.maximum is None or (
                number < node.maximum if node.exclusive_maximum else number <= node.maximum
            )
            if low_ok and high_ok and math.isfinite(value):
                return value
        raise ValueError("could not sample a number in range")

    def string(self, node: String, key: str) -> str:
        hint = self.hint(key)
        if hint is not None and self.fits(node, hint):
            return hint
        data = sample_dfa(string_dfa(node.min_length, node.max_length, node.pattern), self.rng)
        if data is None:
            raise ValueError("string language is empty")
        return data.decode("utf-8")

    def fits(self, node: String, value: str) -> bool:
        return matches(SchemaIR(node, {}, ""), value)

    def hint(self, key: str) -> str | None:  # noqa: PLR0911 - one return per hint family
        lowered = key.lower()
        if "email" in lowered:
            first, last = self.pick(FIRST_NAMES).lower(), self.pick(LAST_NAMES).lower()
            domain = self.pick(("example", "mail", "post"))
            return f"{first}.{last}@{domain}.{self.pick(('com', 'org', 'net'))}"
        if "company" in lowered:
            return self.pick(COMPANIES)
        if any(word in lowered for word in _NAME_KEYS):
            return f"{self.pick(FIRST_NAMES)} {self.pick(LAST_NAMES)}"
        if any(word in lowered for word in _CITY_KEYS):
            return self.pick(CITIES)
        if "street" in lowered:
            return f"{int(self.rng.integers(1, 200))} {self.pick(WORDS).capitalize()} Street"
        if any(word in lowered for word in _PHRASE_KEYS):
            count = int(self.rng.integers(1, 5))
            return " ".join(self.pick(WORDS) for _ in range(count)).capitalize()
        return None

    def array(self, node: Array, depth: int, key: str) -> list[Any]:
        high = node.max_items if node.max_items is not None else node.min_items + 4
        high = min(high, node.min_items + 4)
        if depth + 2 >= self.max_depth:
            high = node.min_items
        length = int(self.rng.integers(node.min_items, high + 1))
        singular = key[:-1] if key.endswith("s") else key
        return [self.node(node.items, depth + 1, singular) for _ in range(length)]

    def obj(self, node: Object, depth: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for prop in node.properties:
            if prop.required or self.rng.random() < 0.7:
                result[prop.name] = self.node(prop.schema, depth + 1, prop.name)
        return result
