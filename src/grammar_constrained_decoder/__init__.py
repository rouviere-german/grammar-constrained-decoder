"""Grammar-constrained decoding: JSON Schema subset -> regex -> minimal DFA -> token masks.

Sub-packages:
- ``automata``: byte-level regex AST, safe pattern parser, Thompson NFA, subset construction,
  Hopcroft minimization and the independent Brzozowski-derivative oracle.
- ``schema``: the supported JSON Schema subset, its lowering to one regular language, canonical
  serialization, instance sampling and DFA-backed validation.
- ``lm``: byte-level BPE tokenizer and interpolated Kneser-Ney n-gram model.
- ``decoding``: token-transition index, compiler with limits and caches, masked decoding and the
  distortion study.
- ``training``: synthetic corpus and the verified train/load pipeline.
- ``serving``: FastAPI app with SSE, JWT scopes, audited schema registry and metrics.
- ``core`` and ``ml``: the shared engineering baseline (settings, logging, auth, rate limits,
  audit log, model registry, manifests, drift).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
