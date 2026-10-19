# syntax=docker/dockerfile:1
#
# Multi-stage, non-root image. Base images are pinned by digest (resolved 2026-09-13); Dependabot's
# docker ecosystem proposes digest bumps. python:3.12-slim is multi-arch, so the index digest works
# on amd64 and arm64.
#
# The builder stage also trains the corpus, tokenizer and 5-gram model (seconds on one CPU, no
# downloads) and bakes the verified registry into the image read-only at /opt/model. Runtime state
# (schema registry, audit log, index cache) goes to APP_DATA_DIR, which must be writable: a tmpfs
# in the CI smoke test, a named volume in docker-compose.yml.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

FROM ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
# Dependencies first, so source edits do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
# The registry writes files atomically via mkstemp, which creates them 0600. The runtime user is not
# root, so the baked registry is made world-readable (still root-owned, so read-only for the app).
RUN uv sync --frozen --no-dev --no-editable \
    && /opt/venv/bin/python -m grammar_constrained_decoder train --workdir /opt/model --seed 0 \
    && chmod -R a+rX,go-w /opt/model

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="grammar-constrained-decoder" \
      org.opencontainers.image.licenses="MIT"
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent \
       --shell /usr/sbin/nologin app \
    && install -d -o app -g app -m 0750 /data
# /data exists and belongs to the service user so that a fresh named volume mounted there (as in
# docker-compose.yml) inherits that ownership; the image itself defaults to a tmpfs path.
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/model/registry /opt/model/registry
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_MODEL_DIR=/opt/model/registry \
    APP_DATA_DIR=/tmp/gcd
WORKDIR /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-m", "grammar_constrained_decoder.core.healthcheck"]
ENTRYPOINT ["python", "-m", "grammar_constrained_decoder"]
CMD ["serve"]
