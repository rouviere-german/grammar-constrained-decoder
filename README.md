# grammar-constrained-decoder

Compiles a subset of JSON Schema into a minimal DFA with a precomputed token-transition index, so a
locally trained language model emits schema-valid JSON on every generation. An independent
Brzozowski-derivative oracle checks that every token mask is exact. Everything (corpus, byte-level
BPE tokenizer, Kneser-Ney 5-gram model, compiler, decoder, API) runs offline on a CPU in seconds,
with no model downloads and no torch.

## Problem

"Please answer in JSON" plus a retry loop is unreliable and wastes tokens, and most engineers use
constrained decoding as a black box. Doing it correctly means:

- **token boundaries**: tokens such as `}}`, `","` or `":{"` span several grammar symbols, so a
  first-character check is wrong; every byte of every token has to be checked;
- **nesting** without an unbounded stack, and **state explosion** in the automaton;
- **hostile schemas**: compilation takes attacker-controlled input (inline schemas), so it needs
  hard limits, and a schema `pattern` must never reach a backtracking regex engine;
- **termination**: a mask that only checks "this prefix is still completable" can burn the whole
  token budget inside a string and stop mid-document;
- **distortion**: masking and renormalizing token by token changes the model's distribution,
  which is rarely measured.

This repository implements each of those from first principles and tests the invariants with
property-based tests, an independent oracle and benchmarks.

## Architecture

```mermaid
flowchart LR
    subgraph train["training (seconds, offline)"]
        schemas[(30 bundled schemas<br/>18 train / 12 held out)] --> corpus["synthetic corpus<br/>+ data contracts"]
        corpus --> bpe["byte-level BPE<br/>1,024 tokens"]
        corpus --> kn["interpolated Kneser-Ney<br/>5-gram (numpy)"]
        bpe --> reg[("content-addressed registry<br/>npz + JSON + manifest")]
        kn --> reg
    end
    subgraph compile["compile (per schema, cached)"]
        schema["JSON Schema"] --> ir["subset parser<br/>explicit rejections"]
        ir --> regex["regex AST<br/>bounded nesting inlined"]
        regex --> nfa["Thompson NFA"] --> dfa["subset construction<br/>+ Hopcroft minimization"]
        dfa --> index["token index<br/>allowed tokens + successors"]
        index --> proof["dead-end proof<br/>min tokens to accept"]
        regex -. "independent check" .-> oracle["Brzozowski derivative oracle"]
        oracle -. "bisimulation + per-state masks" .-> index
    end
    subgraph serve["decode / serve"]
        api["FastAPI /v1/generate<br/>JWT, caps, SSE"] --> decoder["greedy / top-p / beam<br/>logits masked with -inf"]
        reg --> decoder
        proof --> decoder
        api --> sreg[("schema registry<br/>hash-chained audit")]
    end
```

**Compiler.** `schema/ir.py` parses the supported subset (see [SCHEMA_SUPPORT.md](SCHEMA_SUPPORT.md))
and rejects everything else with a JSON Pointer and a reason. `schema/lower.py` produces one
byte-level regex for compact canonical JSON: members in declaration order, canonical string escapes,
integer and decimal ranges lowered to digit-level languages (`schema/numeric.py`), string lengths
counted in code points, patterns from the in-repo ECMA subset parser (`automata/pattern.py`)
translated into the JSON-escaped encoding and intersected with the length language. Nesting is
bounded (default depth 6); recursive `$ref`s are unrolled to that depth ([ADR 0002](docs/adr/0002-bounded-nesting-as-a-regular-language.md)).
`automata/nfa.py` and `automata/dfa.py` implement Thompson's construction, subset construction over
byte equivalence classes, product automata for intersection and Hopcroft minimization with a
canonical renumbering, all charged to a budget of NFA states, DFA states and wall-clock time.

**Token index.** For every live DFA state, every token's bytes are walked through the transition
table (vectorized with numpy), recording allowed tokens and successor states
([ADR 0003](docs/adr/0003-precomputed-token-index-over-the-minimal-dfa.md)). A reverse breadth-first
search gives the fewest tokens from each state to acceptance; the compiler refuses a grammar if any
state reachable at a token boundary cannot finish. Compiled grammars are cached in memory and on
disk (npz, keyed by schema hash, tokenizer hash and limits).

**Decoder.** Greedy, nucleus (top-p) and beam search over the language model's log-probabilities,
with disallowed tokens set to `-inf`. EOS is allowed only in accepting states, and with `r` tokens
left only tokens that can still reach acceptance in `r - 1` more are offered, so every generation
ends in a valid document within `max_tokens` ([ADR 0006](docs/adr/0006-canonical-output-and-budget-aware-masks.md)).

**Oracle.** `automata/oracle.py` rebuilds the language with Brzozowski derivatives, independently of
the NFA/DFA code, walks the product of DFA states and derivatives (a bisimulation, so a clean walk
proves the DFA exact) and recomputes every mask and successor of the token index
([ADR 0004](docs/adr/0004-independent-derivative-oracle.md)).

**Model.** An interpolated modified Kneser-Ney 5-gram over BPE token ids, trained in numpy on a
synthetic corpus of 2,160 prompted JSON documents plus 600 template sentences
([ADR 0005](docs/adr/0005-kneser-ney-n-gram-model.md), [MODEL_CARD.md](MODEL_CARD.md)). It is weak on
purpose: claims here concern validity, exactness, overhead and distortion, not semantic quality.

**Service.** `POST /v1/generate` (JSON or Server-Sent Events) with a registered `schema_id` or an
inline `schema`; `POST /v1/schemas` (admin scope, audited); `GET /v1/schemas[/{id}]`; `/healthz`,
`/readyz` (model loaded and hash-verified, registry audit chain valid) and `/metrics`
(`tokens_per_second`, `mask_latency_seconds`, `index_cache_hit_ratio`, `compile_seconds`,
`compile_rejections_total`, `output_validity_ratio`, `prompt_length_psi`).

Layout:

```text
src/grammar_constrained_decoder/
  automata/   charset, regex AST, pattern parser, NFA, DFA (Hopcroft), derivative oracle, limits
  schema/     subset parser (IR), numeric ranges, lowering, canonical serialization, sampler, validation
  lm/         byte-level BPE tokenizer, Kneser-Ney 5-gram
  decoding/   token index, compiler + cache, decoder, distortion study
  training/   corpus generation, pipeline (train, register, verified load)
  serving/    API, schema registry, concurrency caps, metrics, service
  core/ ml/   shared baseline: settings, logging, auth, rate limits, audit log, registry, drift
bench/        validity, exactness, distortion, performance, micro-benchmarks
tests/        unit + integration (property-based with hypothesis)
```

## Quickstart

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.12. No GPU, no network after `uv sync`.

```bash
uv sync --frozen
uv run python tasks.py demo    # trains, compiles, decodes with/without constraints, streams SSE
uv run python tasks.py check   # ruff, mypy --strict, pytest + coverage, bandit, pip-audit, baseline
uv run python tasks.py bench   # every bench/ script against bench/thresholds.json
```

On Windows with a small repository drive, keep the environment elsewhere first:
`$env:UV_PROJECT_ENVIRONMENT = "C:/Users/<you>/.venvs/grammar-constrained-decoder"`.

Run the service:

```bash
uv run grammar-constrained-decoder train --workdir var/model          # ~3 s, writes var/model/registry
export APP_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
TOKEN="$(uv run grammar-constrained-decoder issue-token --sub alice --scope generate)"
uv run grammar-constrained-decoder serve &
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"prompt": "Q: a geographic point\ngeo_point:\n", "schema_id": "geo_point", "max_tokens": 64}' \
     http://127.0.0.1:8000/v1/generate
curl -sN -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}, "stream": true}' \
     http://127.0.0.1:8000/v1/generate
```

Compile a schema file and print its automaton statistics:
`uv run grammar-constrained-decoder compile my_schema.json`.

Containers: `docker compose up --build` builds a non-root, read-only image that trains the model
at build time and stores registry state in a named volume. Container build, smoke test (health and
readiness), trivy scan and SBOM run in CI (`.github/workflows/ci.yml`, job `docker`). The build,
non-root check (`uid=10001(app)`), HEALTHCHECK, `/readyz` and one authenticated `/v1/generate` call
were also checked locally with Docker 29.6.1 (image: 87.8 MB content, 374 MB unpacked); trivy and
the SBOM were not run locally.

## Results

All numbers below were copied from real runs of the bench scripts on one machine: Windows 10
(10.0.19045), Intel64 Family 6 Model 158 (6 logical CPUs), Python 3.12.14, CPU only, no network.
Each script trains the corpus, tokenizer and 5-gram model from scratch into a temporary directory
first (about 3 s) and fails with `--check` if a metric crosses `bench/thresholds.json`. CI runs
all five (job `bench`); locally `uv run python tasks.py bench` does the same. Wall-clock times
(run one after another): validity 64 s, exactness 157 s, distortion 9 s, performance 18 s,
micro-benchmarks 10 s.

### Validity: constrained vs unconstrained vs 3 retries

`uv run python bench/bench_validity.py --check`: 1,000 nucleus-sampled generations (top-p 0.95,
temperature 1.0, `max_tokens` 256) on the 12 held-out schemas the model never saw; validity means
`json.loads` succeeds and `jsonschema` (Draft 2020-12) accepts the value. Every output is also
checked with the service's `SafeValidator` (ECMA `pattern` semantics by DFA); the two validators
disagreed on 0 outputs. For unconstrained output, "parses" is the share that is JSON at all and
"viable" the mean share of an output's bytes before it leaves every valid document (the prefix
the grammar would still have accepted).

| held-out schema | n | constrained | unconstrained | + 3 retries | parses | viable | greedy c/u |
|---|---:|---:|---:|---:|---:|---:|---|
| invoice | 84 | 100.0% | 0.0% | 0.0% | 16.7% | 5.1% | ok/x |
| vital_signs | 84 | 100.0% | 0.0% | 0.0% | 11.9% | 5.3% | ok/x |
| shipment | 84 | 100.0% | 0.0% | 0.0% | 11.9% | 5.3% | ok/x |
| job_posting | 84 | 100.0% | 0.0% | 0.0% | 14.3% | 6.3% | ok/x |
| category_tree | 83 | 100.0% | 0.0% | 0.0% | 10.8% | 5.5% | ok/x |
| feature_flag | 83 | 100.0% | 0.0% | 0.0% | 13.2% | 6.8% | ok/x |
| game_score | 83 | 100.0% | 0.0% | 0.0% | 12.0% | 5.6% | ok/x |
| air_quality | 83 | 100.0% | 0.0% | 0.0% | 10.8% | 4.7% | ok/x |
| library_loan | 83 | 100.0% | 0.0% | 0.0% | 15.7% | 7.3% | ok/x |
| support_ticket | 83 | 100.0% | 0.0% | 0.0% | 9.6% | 7.1% | ok/x |
| hotel_booking | 83 | 100.0% | 0.0% | 0.0% | 9.6% | 5.4% | ok/x |
| geo_point | 83 | 100.0% | 0.0% | 0.0% | 20.5% | 4.4% | ok/x |

The shipped model is a weak baseline on purpose, so the same script adds a stronger one: for each
held-out schema, a 5-gram fitted on 120 sampled documents of that schema alone (same tokenizer
and code, not the shipped model), 20 samples per schema in the same three modes:

| held-out schema | n | constrained | unconstrained | + 3 retries | parses | viable |
|---|---:|---:|---:|---:|---:|---:|
| invoice | 20 | 100.0% | 0.0% | 0.0% | 5.0% | 23.7% |
| vital_signs | 20 | 100.0% | 5.0% | 30.0% | 50.0% | 36.0% |
| shipment | 20 | 100.0% | 0.0% | 0.0% | 35.0% | 17.9% |
| job_posting | 20 | 100.0% | 0.0% | 5.0% | 0.0% | 42.4% |
| category_tree | 20 | 100.0% | 15.0% | 50.0% | 25.0% | 60.8% |
| feature_flag | 20 | 100.0% | 0.0% | 10.0% | 25.0% | 29.3% |
| game_score | 20 | 100.0% | 0.0% | 0.0% | 0.0% | 47.8% |
| air_quality | 20 | 100.0% | 40.0% | 90.0% | 60.0% | 65.9% |
| library_loan | 20 | 100.0% | 0.0% | 0.0% | 60.0% | 20.5% |
| support_ticket | 20 | 100.0% | 0.0% | 0.0% | 0.0% | 25.8% |
| hotel_booking | 20 | 100.0% | 10.0% | 15.0% | 30.0% | 13.2% |
| geo_point | 20 | 100.0% | 0.0% | 15.0% | 15.0% | 33.2% |

```text
generations: 1000.0
constrained_valid_ratio: 1.0
unconstrained_valid_ratio: 0.0
unconstrained_3_retries_valid_ratio: 0.0
unconstrained_json_parse_ratio: 0.131
unconstrained_required_keys_ratio: 0.0015
unconstrained_viable_prefix_ratio: 0.0573
constrained_tokens_per_generation: 57.3
unconstrained_tokens_per_generation: 31.0
retry_tokens_per_generation: 124.9
greedy_constrained_valid_schemas: 12.0
greedy_unconstrained_valid_schemas: 0.0
seen_schema_unconstrained_valid_ratio: 0.0
constrained_validator_disagreements: 0.0
validator_disagreements: 0.0
single_schema_generations: 240.0
single_schema_constrained_valid_ratio: 1.0
single_schema_unconstrained_valid_ratio: 0.0583
single_schema_3_retries_valid_ratio: 0.1792
single_schema_unconstrained_json_parse_ratio: 0.2542
single_schema_unconstrained_required_keys_ratio: 0.2094
single_schema_unconstrained_viable_prefix_ratio: 0.3471
single_schema_retry_tokens_per_generation: 231.2
held_out_seconds: 39.5
single_schema_seconds: 12.9
```

Read the shipped-model rows as a statement about the 5-gram as much as about masking: with a
four-token memory it mixes fields from different schemas, so it produced no valid document even
on the 18 schemas it was trained on (0 of 360 samples). 13.1% of its outputs parse as JSON, yet
across all outputs only 0.15% of the target schema's required top-level keys are present: the
objects it closes are built from other schemas' fields. On average only the first 5.7% of an
output's bytes are still a prefix of some valid document. Three retries spend four times the
tokens for nothing.

The single-schema model knows the target format and does better, but not well: 5.8% of its
unconstrained samples are valid, 17.9% after up to three retries (231 tokens per request on
average), 25% parse as JSON and on average a third of each output is a viable prefix. High-entropy
values (random identifiers, digits) push a four-token context off the document structure. With
the mask, both models are valid in 100% of samples, by construction rather than by luck. A
baseline that is right most of the time would need a model with a longer memory than an n-gram,
which this repository does not include (see Non-goals), so these numbers show what masking
guarantees, not how much it adds to a capable model.

### Mask exactness, completeness, dead ends, DFA vs `re`

`uv run python bench/bench_exactness.py --check` (2,000 derandomized hypothesis schemas, the real
1,024-token vocabulary; 5 generated schemas have an empty language and are rejected at compile
time with `empty_language`, no generated schema is rejected for any other reason, and the other
1,995 are checked; then the same oracle check on all 30 bundled and 20 tiny schemas):

```text
schemas: 1995.0
rejected_empty_language: 5.0
unexpected_rejections: 0.0
dfa_states: 59295.0
token_reachable_states: 57300.0
token_decisions: 58617900.0
allowed_tokens: 2003247.0
equivalence_pairs: 62288.0
mask_mismatches: 0.0
equivalence_mismatches: 0.0
instances: 5331.0
completeness_failures: 0.0
instance_generation_skipped: 0.0
exactness_seconds: 99.7
dead_end_schemas_checked: 50.0
dead_end_failures: 0.0
corpus_oracle_schemas: 50.0
corpus_dfa_states: 33474.0
corpus_token_reachable_states: 33424.0
corpus_token_decisions: 34192752.0
corpus_equivalence_mismatches: 0.0
corpus_mask_mismatches: 0.0
corpus_seconds: 48.6
re_comparisons: 40000.0
re_disagreements: 0.0
mean_dfa_states: 29.7
total_seconds: 155.2
```

- **Exactness:** for every one of the 1,995 generated grammars, the derivative oracle's
  bisimulation walk (62,288 DFA-state/derivative pairs) found no disagreement. The oracle then
  found the states reachable at a token boundary itself (57,300, the same set as the index lists)
  and at each of them the allowed-token sets and successors matched the index: 58.6 million
  (state, token) decisions, 0 mismatches.
- **Realistic schemas:** the same check on the 30 bundled and 20 tiny schemas (patterns,
  recursion, up to 3,139 DFA states; 33,474 DFA states in total): 33,424 token-reachable states,
  34.2 million decisions, 0 mismatches.
- **Completeness:** 5,331 `hypothesis-jsonschema` instances, canonically serialized, all accepted.
- **No dead ends:** all 30 bundled and 20 tiny schemas compile with the dead-end proof.
- **DFA vs `re`:** 500 generated patterns (trusted, generated by the script) x 40 strings, full
  match and search: 40,000 comparisons, 0 disagreements. The generated patterns (unanchored
  bodies, `re.ASCII` classes, an alphabet without Unicode digits, Unicode spaces or `
`) stay
  where ECMA and Python agree; the dialect differences are listed in
  [SCHEMA_SUPPORT.md](SCHEMA_SUPPORT.md).

### Distortion of masked sampling

`uv run python bench/bench_distortion.py --check`: 20 tiny schemas whose token languages are
enumerated exactly. `P(. | valid)` is the model conditioned on validity (ideal constrained
sampler), `Q` the distribution masked, locally renormalized sampling actually draws from. KL is
KL(P(. | valid) || Q) in nats; "TV samples" compares 300 real decoder samples with the exact `Q`.
Enumeration and sampling use a 32-token budget; the longest valid string among the 20 schemas is
20 bytes (`{"a":false,"b":null}`), so the budget excludes no tokenization.

| schema | strings | token seqs | valid mass | KL string | KL token | greedy = mode | TV samples |
|---|---:|---:|---:|---:|---:|---|---:|
| bool | 2 | 12 | 4.52e-09 | 0.1798 | 2.7150 | no | 0.038 |
| null | 1 | 5 | 2.26e-09 | 0.0000 | 2.2663 | yes | 0.000 |
| color | 3 | 10 | 1.92e-19 | 1.4518 | 2.4557 | no | 0.047 |
| answer | 3 | 7 | 2.61e-17 | 0.9876 | 3.1228 | no | 0.025 |
| digit | 10 | 10 | 3.84e-07 | 0.0016 | 0.0016 | yes | 0.111 |
| percent | 100 | 152 | 1.95e-06 | 1.1777 | 1.2426 | no | 0.227 |
| offset | 11 | 11 | 2.38e-07 | 0.1166 | 0.1166 | yes | 0.049 |
| ab_word | 7 | 7 | 2.43e-13 | 4.0235 | 4.0235 | no | 0.034 |
| abc_code | 12 | 12 | 1.52e-15 | 6.3915 | 6.3915 | no | 0.068 |
| ok_flag | 2 | 48 | 6.74e-12 | 0.0013 | 0.0956 | yes | 0.033 |
| small_id | 5 | 40 | 1.11e-04 | 0.0109 | 0.0203 | yes | 0.057 |
| bool_pair | 7 | 157 | 2.99e-13 | 3.7686 | 3.7692 | no | 0.048 |
| bits | 14 | 14 | 9.78e-16 | 5.3332 | 5.3332 | no | 0.050 |
| maybe_bool | 3 | 17 | 6.78e-09 | 0.1410 | 2.5866 | no | 0.019 |
| optional_pair | 6 | 1029 | 6.28e-08 | 5.6833 | 5.7073 | no | 0.014 |
| priority | 3 | 16 | 2.91e-17 | 0.5984 | 2.6192 | no | 0.022 |
| digit_string | 11 | 11 | 2.46e-13 | 5.9507 | 5.9507 | no | 0.070 |
| log_level | 2 | 768 | 2.97e-06 | 0.0750 | 0.1006 | no | 0.000 |
| small_ints | 21 | 21 | 3.01e-13 | 5.5228 | 5.5228 | no | 0.074 |
| teen | 11 | 22 | 3.71e-07 | 0.0328 | 0.2642 | yes | 0.064 |

```text
schemas: 20.0
kl_string_mean: 2.0724
kl_string_median: 0.793
kl_string_max: 6.3915
kl_token_mean: 2.7153
greedy_is_mode_ratio: 0.3
sampled_tv_max: 0.227
kl_string_exceeds_kl_token: 0.0
```

Masking is not a neutral operation: the mean string-level KL is 2.07 nats, and greedy constrained
decoding returned the most likely valid string (under the model conditioned on validity) for only
6 of 20 schemas. Six of the seven schemas with a string-level KL above 3.7 nats get less than
1e-12 of the unconstrained model's mass on valid output: when the model rarely goes where the
grammar allows, the mask forces it through continuations it considers unlikely, and local
renormalization skews most. Low valid mass alone does not predict it (`color`, at 1.9e-19, has a
KL of 1.45). The TV column is a
check of the decoder against the exact `Q` (sampling noise: 300 draws, up to 100 strings), not a
distortion measure.

### Overhead, compile time and hostile schemas

`uv run python bench/bench_performance.py --check` (mask overhead over constrained greedy and
top-p decoding of every held-out schema; cold compiles of all 30 bundled schemas; stress cases
with a 2 s compile timeout):

```text
mask_tokens_measured: 6984.0
mask_p50_us: 6.7
mask_p95_us: 19.3
mask_p99_us: 30.6
lm_step_p50_us: 33.9
compile_cold_median_s: 0.189
compile_cold_max_s: 0.862
compile_memory_hit_ms: 0.363
compile_disk_hit_ms: 26.0
stress_cases: 12.0
stress_accepted: 0.0
stress_max_seconds: 0.57
```

Compile time vs schema size (excerpt of the 30-row table, smallest, median-sized and largest
schemas; cold compile = parse, lower, NFA, DFA, minimize, token index, dead-end proof):

| schema | canonical bytes | DFA states | index entries | cold compile (s) |
|---|---:|---:|---:|---:|
| category_tree | 199 | 2532 | 216718 | 0.645 |
| feature_flag | 330 | 147 | 15619 | 0.031 |
| library_loan | 342 | 85 | 1201 | 0.014 |
| user_profile | 370 | 645 | 108685 | 0.212 |
| calendar_event | 491 | 3139 | 335346 | 0.803 |
| invoice | 599 | 2645 | 315949 | 0.862 |
| order | 726 | 842 | 111328 | 0.259 |

Compile cost tracks DFA states times vocabulary size (the index walk), not schema bytes: a
recursive 199-byte `category_tree` costs almost as much as the 599-byte `invoice`.

| stress case | rejection reason | seconds |
|---|---|---:|
| deep_nesting_200 | schema_depth | 0.0 |
| enum_10000 | enum_size | 0.011 |
| pattern_5000_chars | pattern_length | 0.0 |
| exponential_dfa | dfa_states | 0.168 |
| nested_counters | subset_size | 0.57 |
| max_length_1e9 | repeat_bound | 0.0 |
| bound_1e40 | number_digits | 0.0 |
| object_500_properties | properties | 0.001 |
| any_of_5000 | schema_size | 0.023 |
| backreference | pattern | 0.001 |
| ref_chain_600 | ref_depth | 0.006 |
| lone_surrogate_enum | invalid | 0.0 |

Every stress case is rejected with a typed reason well inside the 2 s timeout. `nested_counters`
used to run until the deadline (2.09 s in the previous run of this bench), allocating subsets all
the way; subset construction is now charged for the NFA-state entries it holds, and the pattern
stops at the 1,000,000-entry cap in 0.57 s. The deadline itself is cooperative: it is checked at
every new DFA state and every few dozen cheaper steps, so a compile that does reach it stops
within one check interval
(`tests/unit/test_compiler.py::test_timeouts_fire_within_one_check_interval` lifts every other
cap and asserts at most 0.25 s past a 0.5 s deadline).

Per-token mask overhead (fetching the budget-aware mask and applying it to the logits) has a
median of 6.7 us and a p95 of 19.3 us, against a median of 33.9 us for one step of the numpy
5-gram (no neural model is measured here). The micro-benchmarks
(`uv run python bench/bench_baseline.py --check`, p95): cached mask fetch and apply 3.7 us, one
LM step 57.9 us, compile from the memory cache 358 us, cold compile of `invoice` 0.93 s, DFA match
of a 10 KB ReDoS-style input 3.3 ms (linear, no backtracking).

## Security

Threats, controls and the tests that prove them are in [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md);
every claim below links to evidence in [CLAIMS.md](CLAIMS.md).

- **Validity is not safety.** The decoder guarantees schema-valid JSON, not true, authorized or
  harmless content. A prompt can still steer which valid document is produced. Authorize every
  effect of the output downstream, as for any untrusted input.
- **Hostile schemas.** Every client-supplied schema (inline or being registered) is vetted
  before any token is generated, whether or not decoding is constrained: parsed only by the
  in-repo parser (no eval, local `$ref` only, unsupported keywords rejected explicitly) and
  compiled under the service limits. The document itself must be plain JSON (bounded nesting,
  finite numbers, no lone surrogates). Compilation enforces limits on depth, schema size,
  properties, enum size, literal and pattern length, repetition counts, number digits, `$ref`
  expansion depth, NFA states, DFA states, the memory of subset construction and token-index
  entries, plus a cooperative wall-clock timeout checked at every new DFA state; every rejection
  is a typed `400`/`413` with a reason and a metric, and rejected schemas are remembered by
  digest for five minutes. Patterns are matched by DFA in linear time; output validation swaps
  `jsonschema`'s `pattern` implementation for the DFA (compiled under the same limits) and uses a
  reference registry that never retrieves anything, so Python's `re` never sees a client pattern
  and no `http://` or `file://` reference is ever fetched or read. Validation runs in a worker
  thread for streamed and non-streamed responses alike.
- **Access.** HS256 JWTs (pinned algorithm, required claims, bounded lifetime, fail closed without
  a secret) with a `generate` scope for decoding and a `schemas:admin` scope for registration.
  `/metrics` needs an operator API key. Per-subject rate limits, non-blocking global and
  per-client concurrency caps, and caps on `max_tokens`, prompt length and inline schema size.
- **Integrity.** Schema registrations and every rejected attempt (compile rejections, oversize
  or malformed schemas, id conflicts) are appended to a hash-chained audit log; a stored schema
  that no longer matches its audited hash is refused and fails `/readyz`.
  Model counts and tokenizer load from npz (`allow_pickle=False`) and JSON only after the registry
  and the run manifest hashes verify.
- **Privacy.** Prompts and outputs are never logged: log lines carry lengths, truncated SHA-256
  digests, the schema label, token counts and timings.
- CI: ruff, mypy --strict, pytest with an 85% coverage floor, bandit, semgrep with repo rules,
  pip-audit, gitleaks, hadolint, trivy and a syft CycloneDX SBOM; base images pinned by digest.
  See [SECURITY.md](SECURITY.md) for reporting.

## Non-goals

- **Full JSON Schema.** `oneOf`, `allOf`, `not`, conditionals, `format`, `multipleOf`,
  `patternProperties`, open objects, `uniqueItems: true`, `prefixItems`/`contains`, remote refs and
  unbounded recursion are rejected ([SCHEMA_SUPPORT.md](SCHEMA_SUPPORT.md)). Several are regular in
  principle (`oneOf` via language difference) but would need complement automata and are cut.
- **Arbitrary JSON formatting.** Output is compact canonical JSON in declaration order; pretty
  printing, reordered members and exponent notation are never generated.
- **Semantic quality.** The 5-gram model forgets which schema it is writing after four tokens;
  constrained output is valid but often semantically odd (a timestamp in a name field). No claim is
  made about content quality, and no neural model is included.
- **A strong unconstrained baseline.** The strongest baseline here, a 5-gram trained on each
  held-out schema alone, is valid in only a few percent of unconstrained samples; a baseline that
  is right 30-70% of the time would need a model with longer memory than an n-gram, which is out
  of scope (no neural models, no downloads). The validity comparison therefore shows what masking
  guarantees, not how much it adds to a capable model.
- **Python `re` pattern semantics.** `pattern` follows ECMA-262 as JSON Schema requires; where
  Python's `re` differs (Unicode `\d`/`\w`, `$` before a final newline, see
  [SCHEMA_SUPPORT.md](SCHEMA_SUPPORT.md)), a document can pass one validator and fail the other.
- **Distortion correction.** The distortion of locally renormalized masking is measured, not
  corrected; grammar-aligned sampling (for example importance weighting or ASAp) is future work.
- **Distributed serving.** Rate limits, concurrency caps and caches are per process; there is no
  batching across requests and no GPU path.
- **Signed artifacts.** The model registry, index cache and schema registry are hash-verified,
  not signed; an attacker who can rewrite a manifest or recompute a cache checksum is out of
  scope (threat model T3, T4).
- **Token revocation.** JWTs are verified statelessly; there is no revocation list or key rotation.

## References

- K. Thompson, "Regular expression search algorithm", CACM 1968 (NFA construction).
- M. O. Rabin and D. Scott, "Finite automata and their decision problems", 1959 (subset construction).
- J. Hopcroft, "An n log n algorithm for minimizing states in a finite automaton", 1971.
- J. A. Brzozowski, "Derivatives of regular expressions", JACM 1964; S. Owens, J. Reppy and
  A. Turon, "Regular-expression derivatives re-examined", JFP 2009.
- R. Kneser and H. Ney, "Improved backing-off for m-gram language modeling", ICASSP 1995;
  S. F. Chen and J. Goodman, "An empirical study of smoothing techniques for language modeling", 1998.
- B. T. Willard and R. Louf, "Efficient Guided Generation for Large Language Models", 2023
  (FSM-indexed vocabulary masks, the idea implemented here from scratch).
- K. Park et al., "Grammar-Aligned Decoding", NeurIPS 2024 (distortion of constrained sampling).
- The UTF-8 range splitting follows the construction used by RE2 and Rust's `regex-syntax`.
