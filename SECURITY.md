# Security policy

## Supported versions

Only the latest commit on `main` is supported. This is a portfolio project, not a product with
release branches.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting ("Report a vulnerability" under the repository's Security tab). Include:

- the affected commit and component (schema compiler, token index, decoder, API, training);
- steps to reproduce, ideally as a failing test or a schema file;
- the impact you expect.

You can expect an acknowledgement within 7 days and a fix or a written assessment within 30 days.
Credit is given in the fix commit unless you ask otherwise.

Especially welcome: a schema accepted by the compiler whose constrained output fails `jsonschema`
validation, a mask that differs from the derivative oracle, a schema that takes longer than the
compile timeout to reject, or a way to make Python's `re` evaluate a client-supplied pattern.

## Scope

In scope: the code in this repository, its container image and its CI configuration.
Out of scope: denial of service by traffic volume, findings that need a compromised host, and
issues in third-party dependencies that are already publicly tracked (please report those
upstream).

## What the guarantees mean

Constrained decoding guarantees that generated text is a canonical serialization of an instance
of the schema. **Validity is not safety**: the content is still produced by a language model and
can be wrong or adversarial (a prompt can steer which valid document comes out). Treat every
output as untrusted input and authorize its effects downstream.

## How this repository is hardened

- Threats, controls and the tests that prove them: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
- Every security claim is linked to evidence in [CLAIMS.md](CLAIMS.md).
- Schemas are parsed only by the in-repo parser; patterns are compiled to DFAs (no backtracking);
  Python's `re` never sees a client-supplied pattern; compilation has state caps and a timeout.
- Generation endpoints need a scoped HS256 JWT; schema registration needs the admin scope and is
  recorded in a hash-chained audit log; prompts and outputs are never logged.
- Model artifacts are JSON and npz (`allow_pickle=False`), loaded only after hash verification.
- CI runs ruff, mypy --strict, pytest with a coverage floor, bandit, semgrep (registry packs plus
  `.semgrep/rules.yml`), pip-audit, gitleaks, hadolint, a trivy image scan and a syft CycloneDX
  SBOM.
- Secrets come only from the environment; only `.env.example` is committed.
