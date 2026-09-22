# 5. A trained-in-repo Kneser-Ney 5-gram model instead of a neural language model

- Status: accepted
- Date: 2026-09-22

## Context

The project must run offline on a CPU-only laptop, with no model downloads and no torch, and the
whole train-evaluate cycle must fit in CI. The decoder needs a language model that returns a full
next-token distribution over the tokenizer's vocabulary, deterministically.

## Decision

Train a byte-level BPE tokenizer (1,024 tokens, Hugging Face `tokenizers`) and an interpolated
modified Kneser-Ney 5-gram model (Chen and Goodman 1998) in numpy on a synthetic corpus generated
from 18 bundled schemas plus template sentences. Twelve further schemas are held out and never
seen in training. Counts are stored as npz arrays in the content-addressed registry together with
a run manifest (seed, corpus hashes, tokenizer hash, package versions).

## Consequences

- Training takes a few seconds; every bench and the demo retrain from scratch, so results do not
  depend on leftover state.
- The model is deliberately weak: on held-out schemas its unconstrained output is almost never
  valid, which makes the constrained/unconstrained comparison meaningful. It is also a poor judge
  of content: constrained outputs are valid but often semantically odd (dates in name fields).
- Claims are therefore limited to validity, exactness, overhead and distortion. Nothing here says
  anything about the semantic quality of constrained output from a strong model.
- The decoder only depends on a `logprobs(context) -> vector` protocol, so a neural model could be
  plugged in without changing the grammar machinery; doing so is out of scope.
