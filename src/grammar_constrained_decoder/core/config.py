"""Typed configuration loaded from environment variables (prefix ``APP_``).

Only ``.env.example`` is committed. Secrets are ``SecretStr`` so they never appear in ``repr``
or logs. API keys are configured as SHA-256 digests, so the process never holds raw keys.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = ["Settings", "get_settings"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Service settings. Every field can be set with ``APP_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        # ``KEY=`` (as in .env.example) means "not set": an empty secret must not fail validation.
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
    )

    # --- service ---
    service_name: str = Field(
        default="grammar-constrained-decoder", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$"
    )
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    enable_docs: bool = False

    # --- HTTP security ---
    api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    rate_limit_per_second: float = Field(default=5.0, gt=0)
    rate_limit_burst: int = Field(default=20, ge=1)
    rate_limit_max_clients: int = Field(default=10_000, ge=1)
    max_body_bytes: int = Field(default=1_048_576, ge=1)
    metrics_require_auth: bool = True

    # --- observability ---
    tracing_exporter: Literal["none", "console"] = "none"

    # --- JWT authentication (generation API) ---
    jwt_secret: SecretStr | None = None
    jwt_issuer: str = Field(default="grammar-constrained-decoder", min_length=1, max_length=128)
    jwt_audience: str = Field(default="gcd-api", min_length=1, max_length=128)
    jwt_leeway_seconds: int = Field(default=30, ge=0, le=300)
    jwt_max_ttl_seconds: int = Field(default=86_400, ge=60, le=2_592_000)

    # --- generation limits ---
    max_tokens_cap: int = Field(default=256, ge=1, le=4096)
    max_prompt_chars: int = Field(default=2_000, ge=1, le=100_000)
    max_prompt_tokens: int = Field(default=512, ge=1, le=100_000)
    max_inline_schema_bytes: int = Field(default=32_768, ge=64, le=1_048_576)
    max_concurrent_generations: int = Field(default=4, ge=1, le=256)
    max_concurrent_per_client: int = Field(default=2, ge=1, le=256)

    # --- compiler limits ---
    compile_timeout_seconds: float = Field(default=5.0, gt=0, le=120)
    compile_max_dfa_states: int = Field(default=20_000, ge=16, le=1_000_000)
    compile_max_depth: int = Field(default=6, ge=1, le=32)

    # --- storage ---
    data_dir: Path = Path("var")
    model_dir: Path = Path("var/model/registry")

    @field_validator("api_key_sha256", mode="before")
    @classmethod
    def _parse_key_hashes(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("api_key_sha256")
    @classmethod
    def _check_key_hashes(cls, value: list[str]) -> list[str]:
        for digest in value:
            if not _SHA256_HEX.fullmatch(digest):
                raise ValueError("api_key_sha256 entries must be lowercase hex SHA-256 digests")
        return value

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("cors_allow_origins")
    @classmethod
    def _check_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if not _ORIGIN.fullmatch(origin):
                raise ValueError(
                    "cors_allow_origins must be explicit scheme://host[:port] origins "
                    "('*' is refused)"
                )
        return value

    @field_validator("jwt_secret")
    @classmethod
    def _check_jwt_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value().encode("utf-8")) < 32:
            raise ValueError("jwt_secret must be at least 32 bytes (HS256 needs a 256-bit key)")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Tests construct ``Settings(...)`` directly instead."""
    return Settings()
