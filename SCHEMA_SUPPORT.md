# JSON Schema support matrix

The compiler accepts a subset of JSON Schema (Draft 2020-12 vocabulary) whose instances form a
regular language when nesting is bounded. Everything else is rejected with an explicit
`SchemaError` naming the keyword and a JSON Pointer; nothing is silently ignored except the
annotation keywords listed below. `tests/unit/test_schema_ir.py::test_support_matrix_matches_the_parser`
keeps this table and the parser in sync, and the conformance tests cited in the last column prove
each row.

Output is **compact canonical JSON**: no insignificant whitespace, object members in schema
declaration order, the canonical string escapes of `json.dumps(ensure_ascii=False)`, integers
without leading zeros and numbers in fixed-point notation (no exponent, never `-0`). This holds
for `enum`/`const` members too: `{"enum": [1e20, 1e-7, -0.0]}` compiles to
`100000000000000000000`, `0.0000001` and `0.0`
(`tests/unit/test_lower.py::test_numeric_enum_and_const_literals_use_fixed_point`). Object values
inside an `enum` are emitted with their keys sorted (their canonical literal), not in declaration
order.

| Keyword | Status | Semantics in the compiled language | Evidence |
|---|---|---|---|
| `type` | supported | one type or a list of distinct types (a list is a union) | `tests/unit/test_schema_ir.py::test_type_lists_become_unions` |
| `enum` | supported | canonical literal of each value; may be combined with `type` only | `tests/unit/test_lower.py::test_enum_and_const_emit_canonical_literals` |
| `const` | supported | one canonical literal; may be combined with `type` only | `tests/unit/test_lower.py::test_enum_and_const_emit_canonical_literals` |
| `anyOf` | supported | union of the branches; siblings other than annotations are rejected | `tests/unit/test_lower.py::test_any_of_is_a_union` |
| `$ref` | supported | local only: `#`, `#/$defs/<name>`, `#/definitions/<name>`; recursion unrolled to `max_depth` | `tests/unit/test_lower.py::test_recursive_refs_are_unrolled_to_the_depth_limit` |
| `minimum` | supported | integer and number lower bound | `tests/unit/test_numeric.py::test_decimal_ranges_are_exact` |
| `maximum` | supported | integer and number upper bound | `tests/unit/test_numeric.py::test_integer_ranges_are_exact` |
| `exclusiveMinimum` | supported | numeric form (Draft 6+); boolean form rejected | `tests/unit/test_numeric.py::test_decimal_ranges_are_exact` |
| `exclusiveMaximum` | supported | numeric form (Draft 6+); boolean form rejected | `tests/unit/test_numeric.py::test_decimal_ranges_are_exact` |
| `minLength` | supported | counts code points (escapes count as one) | `tests/unit/test_lower.py::test_string_lengths_count_code_points_not_bytes` |
| `maxLength` | supported | counts code points; at most `max_repeat` (256) | `tests/unit/test_lower.py::test_string_lengths_count_code_points_not_bytes` |
| `pattern` | supported | safe ECMA-262 subset, parsed in-repo, matched by DFA against the decoded value | `tests/unit/test_pattern.py::test_patterns_inside_json_strings_match_the_decoded_value` |
| `items` | supported | required for arrays (a single schema) | `tests/unit/test_lower.py::test_array_bounds_are_exact` |
| `minItems` | supported | at most `max_repeat` | `tests/unit/test_lower.py::test_array_bounds_are_exact` |
| `maxItems` | supported | at most `max_repeat`; omitted means unbounded | `tests/unit/test_lower.py::test_array_bounds_are_exact` |
| `uniqueItems` | supported | only `false` (the default) | `tests/unit/test_schema_ir.py::test_invalid_keyword_values_are_rejected` |
| `properties` | supported | members emitted in declaration order | `tests/unit/test_lower.py::test_optional_members_in_declaration_order` |
| `required` | supported | every name must be declared in `properties` | `tests/unit/test_lower.py::test_optional_members_in_declaration_order` |
| `additionalProperties` | supported | only `false`; omitting it also means closed | `tests/unit/test_schema_ir.py::test_open_objects_are_rejected` |
| `title` | annotation | ignored | `tests/unit/test_schema_ir.py::test_annotations_are_ignored` |
| `description` | annotation | ignored | `tests/unit/test_schema_ir.py::test_annotations_are_ignored` |
| `$comment` | annotation | ignored | `tests/unit/test_schema_ir.py::test_annotations_are_ignored` |
| `examples` | annotation | ignored | `tests/unit/test_schema_ir.py::test_annotations_are_ignored` |
| `default` | annotation | ignored (never emitted) | `tests/unit/test_schema_ir.py::test_annotations_are_ignored` |
| `deprecated` | annotation | ignored | `tests/unit/test_schema_ir.py::test_annotations_are_ignored` |
| `readOnly` | annotation | ignored | `tests/unit/test_schema_ir.py::test_annotations_are_ignored` |
| `writeOnly` | annotation | ignored | `tests/unit/test_schema_ir.py::test_annotations_are_ignored` |
| `$schema` | root-only | ignored at the root (must be a string of at most 512 characters), rejected elsewhere | `tests/unit/test_schema_ir.py::test_root_only_keywords` |
| `$id` | root-only | ignored at the root (must be a string of at most 512 characters), rejected elsewhere | `tests/unit/test_schema_ir.py::test_root_identifiers_must_be_strings` |
| `$defs` | root-only | definitions for local `$ref` | `tests/unit/test_schema_ir.py::test_root_only_keywords` |
| `definitions` | root-only | older spelling of `$defs` | `tests/unit/test_schema_ir.py::test_root_only_keywords` |
| `oneOf` | rejected | exclusive disjunction needs language difference | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `allOf` | rejected | merge the constraints instead | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `not` | rejected | negation | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `if` | rejected | conditionals | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `then` | rejected | conditionals | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `else` | rejected | conditionals | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `format` | rejected | use `pattern` | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `multipleOf` | rejected | not a bounded-width regular language in general | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `patternProperties` | rejected | only declared properties | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `propertyNames` | rejected | only declared properties | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `minProperties` | rejected | property counts | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `maxProperties` | rejected | property counts | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `dependentRequired` | rejected | dependencies | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `dependentSchemas` | rejected | dependencies | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `dependencies` | rejected | dependencies (older spelling) | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `unevaluatedProperties` | rejected | unevaluated keywords | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `unevaluatedItems` | rejected | unevaluated keywords | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `prefixItems` | rejected | tuple validation | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `additionalItems` | rejected | tuple validation | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `contains` | rejected | contains | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `minContains` | rejected | contains | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `maxContains` | rejected | contains | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `contentEncoding` | rejected | content keywords | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `contentMediaType` | rejected | content keywords | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `contentSchema` | rejected | content keywords | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `$anchor` | rejected | use `$defs` | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `$dynamicRef` | rejected | dynamic references | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `$dynamicAnchor` | rejected | dynamic references | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `$recursiveRef` | rejected | dynamic references | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `$recursiveAnchor` | rejected | dynamic references | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `$vocabulary` | rejected | vocabularies | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |
| `nullable` | rejected | OpenAPI; use a type list with `null` | `tests/unit/test_schema_ir.py::test_rejected_keywords_raise_explicit_errors` |

Any keyword not in this table is rejected as unknown.

## Other rules

- Every subschema must constrain its value (`type`, `enum`, `const`, `anyOf` or `$ref`): `{}`
  and boolean schemas are rejected, because "any JSON value" has no finite automaton at unbounded
  depth and would dwarf every other schema at bounded depth.
- Objects are closed: the language contains only declared members.
- Nesting depth (objects plus arrays) is capped at `max_depth` (default 6). Deeper non-recursive
  nesting is an error; recursive `$ref` alternatives beyond the cap are cut, so the compiled
  language contains exactly the instances whose nesting stays within the cap.
- The schema document itself must be plain JSON: nesting at most `max_json_depth` (64) levels,
  finite numbers only, and no string or object key containing a lone surrogate (it has no UTF-8
  encoding). These are checked first, without recursion, and rejected with reason
  `schema_depth` or `invalid`.
- Reference chains: at most `max_ref_depth` (64) `$ref` expansions may be open at once while the
  language is built, so a long chain of definitions is rejected (`ref_depth`) instead of
  exhausting the Python stack. A reference cycle that passes through no container takes the
  least fixed point (a cycle with nothing else is the empty language, rejected as
  `empty_language`).
- Limits (all configurable, all enforced during compilation): document depth, schema nodes,
  properties per object, enum size, literal length, pattern length, repetition counts, number
  digits, reference depth, NFA states, DFA states, NFA-state entries held by subset construction
  (its memory), token-index entries and a wall-clock timeout. See `automata/limits.py`.

## Pattern subset

Supported: literals, `.`, `[...]`/`[^...]` with ranges, `\d \D \w \W \s \S`, `\t \n \r \f \v \0`,
`\xHH`, `\uHHHH`, `\u{H...}`, escaped syntax characters, `(...)`, `(?:...)`, `|`, `* + ? {m} {m,}
{m,n}` (lazy forms accepted), `^` at the start and `$` at the end of a top-level branch. Search
semantics: an unanchored side matches anywhere, as JSON Schema requires.

Rejected: backreferences, lookaround, named groups, `\b \B`, `\p{...}`, `\cX`, octal escapes,
anchors in other positions, lone `{ } ]`, quantifiers without an operand, patterns over 256
characters and repetition counts over `max_repeat`.

### Dialect: ECMA-262, not Python `re`

JSON Schema specifies ECMA-262 regular expressions, and the compiler follows ECMA on the
supported subset. Python's `re` (which the stock `jsonschema` validator uses for `pattern`)
differs on a few points, so a document can be valid for one and invalid for the other:

| construct | here (ECMA) | Python `re` on `str` |
|---|---|---|
| `\d`, `\w` | ASCII digits, ASCII word characters | any Unicode digit, any Unicode word character |
| `\D`, `\W` | everything else, so they match `٣` (Arabic-Indic three) or `é` | exclude Unicode digits and letters |
| `\s`, `\S` | the ECMA whitespace set (tab to CR, space, NBSP, U+1680, U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F, U+3000, U+FEFF) | Python's Unicode whitespace set, which also includes U+001C to U+001F and U+0085 but not U+FEFF |
| `.` | any code point except `\n`, `\r`, U+2028 and U+2029 | any character except `\n` |
| `$` | end of the string only | end of the string or before a final `\n` |

The service validates outputs with its `SafeValidator`, which matches `pattern` with these DFAs
(ECMA semantics, never Python `re`), so it agrees with the compiled language. The bundled schemas
use only ASCII classes and anchors, where the two dialects agree except for `$` before a final
newline; `bench/bench_validity.py` checks every output with both validators and reports the
disagreements (0 on constrained output is a threshold).
